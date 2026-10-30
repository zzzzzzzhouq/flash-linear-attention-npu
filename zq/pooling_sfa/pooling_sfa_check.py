#!/usr/bin/env python3
"""PoolingSFA 部署自检脚本（仅依赖 Python 标准库）。

子命令：
  env       检查硬件/包/算子注册/schema/MemFabric/vendor/ulimit/clang，可选校验模型 config
  service   向 vLLM OpenAI 兼容接口发固定请求（或批量 prompt 文件），校验服务与输出
  ab        对两个端点发同一请求，比较输出一致性与耗时（融合 vs baseline/非融合）
  compare   对比两份批量结果 JSONL，统计整网精度一致率
  log       扫描服务日志，检查融合命中标记与常见致命错误

用法示例：
  python3 pooling_sfa_check.py env --model-config /models/GLM-5.2/config.json
  python3 pooling_sfa_check.py service --base-url http://127.0.0.1:8000 --model glm --prompt "你好"
  python3 pooling_sfa_check.py service --base-url http://127.0.0.1:8001 --model glm --prompt-file prompts.txt --output fused.jsonl
  python3 pooling_sfa_check.py compare --file-a baseline.jsonl --file-b fused.jsonl --report accuracy.json
  python3 pooling_sfa_check.py ab --base-url-a http://127.0.0.1:8000 --base-url-b http://127.0.0.1:8001 --model glm
  python3 pooling_sfa_check.py log --log-file /var/log/vllm/glm.log

退出码：0 通过；1 存在失败项；2 参数/环境错误。
"""

from __future__ import annotations

import argparse
import difflib
import glob
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
import urllib.error
import urllib.request

try:
    import resource
except ImportError:  # Windows 等非 Unix 平台
    resource = None

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []

# 兼容非 UTF-8 控制台（如 Windows GBK）：打印日志片段时不因编码异常中断
try:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))


def has_failure() -> bool:
    return any(status == FAIL for status, _, _ in results)


def summarize(title: str) -> int:
    total = len(results)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print(f"\n===== {title}：{total} 项，PASS {total - n_fail - n_warn}，WARN {n_warn}，FAIL {n_fail} =====")
    if n_fail:
        for status, name, detail in results:
            if status == FAIL:
                print(f"  FAIL: {name} | {detail}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------

def _check_platform() -> None:
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        record(PASS, "CPU 架构", machine)
    else:
        record(WARN, "CPU 架构", f"{machine}（KV offload 辅助扩展按 aarch64 编译，x86 上可能无法运行）")
    record(PASS, "Python", platform.python_version())


def _check_torch() -> tuple[object | None, object | None]:
    torch = None
    torch_npu = None
    try:
        import torch  # type: ignore

        record(PASS, "torch", getattr(torch, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "torch 导入", f"{type(exc).__name__}: {exc}")
        return None, None

    try:
        import torch_npu  # type: ignore  # noqa: F401

        record(PASS, "torch_npu", getattr(torch_npu, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "torch_npu 导入", f"{type(exc).__name__}: {exc}")

    for mod in ("vllm",):
        try:
            m = __import__(mod)
            record(PASS, mod, getattr(m, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001
            record(FAIL, f"{mod} 导入", f"{type(exc).__name__}: {exc}")
    return torch, torch_npu


def _check_vllm_ascend(torch) -> object | None:
    try:
        import vllm_ascend  # type: ignore

        pkg_dir = os.path.realpath(os.path.dirname(vllm_ascend.__file__))
        record(PASS, "vllm_ascend 路径", pkg_dir)
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "vllm_ascend 导入", f"{type(exc).__name__}: {exc}")
        return None

    # 设备类型与 custom op 开关
    device_type = None
    try:
        from vllm_ascend.utils import get_ascend_device_type  # type: ignore

        device_type = str(get_ascend_device_type())
        record(PASS, "Ascend 设备类型", device_type)
        if "A5" in device_type.upper():
            record(FAIL, "硬件适用性", "A5（Ascend950）上 custom op 被强制关闭，本路径不支持")
    except Exception as exc:  # noqa: BLE001
        record(WARN, "Ascend 设备类型", f"读取失败：{type(exc).__name__}: {exc}")

    try:
        from vllm_ascend.utils import enable_custom_op  # type: ignore

        enabled = bool(enable_custom_op())
        record(PASS if enabled else FAIL, "enable_custom_op()", str(enabled))
        if not enabled:
            record(FAIL, "custom op 可用性",
                   "被关闭：常见原因是 A5 设备、VLLM_BATCH_INVARIANT=1，或 vllm_ascend_C 加载失败")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "enable_custom_op()", f"{type(exc).__name__}: {exc}")

    # 算子注册与 schema
    if torch is not None:
        try:
            import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "vllm_ascend_C 导入", f"{type(exc).__name__}: {exc}")
        op = None
        try:
            op = getattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap", None)
        except Exception:  # noqa: BLE001
            op = None
        if op is None:
            record(FAIL, "PoolingSFA 算子注册", "torch.ops._C_ascend 中不存在 npu_fused_sparse_attention_overlap")
        else:
            try:
                schema = str(op.default._schema)
            except Exception:  # noqa: BLE001
                schema = ""
            tensor_count = schema.count("Tensor")
            has_membership = "selection_membership_map" in schema
            ok_schema = has_membership and tensor_count == 12
            record(PASS, "PoolingSFA 算子注册", "已注册")
            record(PASS if ok_schema else FAIL, "算子 schema（15 位置参数版）",
                   f"张量参数 {tensor_count} 个（期望 12）；selection_membership_map={'有' if has_membership else '缺失'}")
            print(f"    schema: {schema}")
    return pkg_dir


def _check_vendor(pkg_dir: str | None) -> None:
    if pkg_dir is None:
        return
    vendor = os.path.join(pkg_dir, "_cann_ops_custom", "vendors", "custom_transformer")
    if not os.path.isdir(vendor):
        record(FAIL, "vendor 算子包", f"目录不存在：{vendor}")
        return
    set_env = os.path.join(vendor, "bin", "set_env.bash")
    record(PASS if os.path.isfile(set_env) else WARN, "vendor set_env.bash",
           set_env if os.path.isfile(set_env) else "缺失，服务启动前需手工设置 ASCEND_CUSTOM_OPP_PATH")

    kernels = glob.glob(os.path.join(
        vendor, "op_impl", "ai_core", "tbe", "kernel", "*",
        "fused_sparse_attention_overlap", "*.o"))
    if kernels:
        record(PASS, "融合算子 kernel 产物", f"{len(kernels)} 个 .o，例如 {os.path.basename(kernels[0])}")
    else:
        record(FAIL, "融合算子 kernel 产物", "未找到 fused_sparse_attention_overlap/*.o")

    opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if vendor in opp_path:
        record(PASS, "ASCEND_CUSTOM_OPP_PATH", "已包含 vendor 目录")
    else:
        record(WARN, "ASCEND_CUSTOM_OPP_PATH", f"未包含 vendor 目录（当前：{opp_path or '空'}）")


def _check_memfabric() -> None:
    if importlib.util.find_spec("memfabric_hybrid") is None:
        record(FAIL, "MemFabric", "未安装 memfabric_hybrid")
        return
    record(PASS, "MemFabric", "memfabric_hybrid 可导入")
    lib = os.environ.get("MEMFABRIC_HYBRID_EXTEND_LIB_PATH", "")
    record(PASS if lib else WARN, "MEMFABRIC_HYBRID_EXTEND_LIB_PATH", lib or "未设置，确认已 source /usr/local/memfabric_hybrid/set_env.sh")


def _check_ulimit() -> None:
    if resource is None:
        record(WARN, "锁页内存 ulimit -l", "当前平台不支持 resource 模块（请在 Linux 目标机执行本项）")
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        fmt = lambda v: "unlimited" if v == resource.RLIM_INFINITY else f"{v / (1 << 20):.0f} MiB"  # noqa: E731
        status = PASS if soft == resource.RLIM_INFINITY or soft >= 1 << 30 else WARN
        record(status, "锁页内存 ulimit -l", f"soft={fmt(soft)}, hard={fmt(hard)}")
        if status == WARN:
            print("    提示：KV 池规模较大时需提高 ulimit -l，否则可能报 offload.empty failed")
    except Exception as exc:  # noqa: BLE001
        record(WARN, "锁页内存 ulimit -l", f"读取失败：{type(exc).__name__}: {exc}")


def _check_clang() -> None:
    clang = shutil.which("clang++")
    record(PASS if clang else FAIL, "clang++", clang or "未找到（运行时需 JIT 编译 kv_offload_decode）")


def _check_model_config(path: str, topk_buffer_size: int, block_size: int) -> None:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "模型 config 读取", f"{type(exc).__name__}: {exc}")
        return

    # 兼容 hf config 嵌套
    text_cfg = cfg.get("text_config", cfg)
    index_topk = text_cfg.get("index_topk")
    records = text_cfg.get("compress_ratios")

    record(PASS if index_topk else FAIL, "模型 index_topk",
           str(index_topk) if index_topk else "缺失：该模型不是稀疏注意力模型，无法使用本路径")
    record(PASS if not records else FAIL, "模型 compress_ratios",
           "无（符合要求）" if not records else "存在：本路径不支持压缩 KV")
    if index_topk:
        if topk_buffer_size < int(index_topk):
            record(FAIL, "topk_buffer_size", f"{topk_buffer_size} < index_topk({index_topk})")
        elif topk_buffer_size % block_size != 0:
            record(FAIL, "topk_buffer_size", f"{topk_buffer_size} 不是 block_size({block_size}) 的整数倍")
        else:
            record(PASS, "topk_buffer_size", f"{topk_buffer_size}（index_topk={index_topk}, block_size={block_size}）")


def cmd_env(args: argparse.Namespace) -> int:
    print("===== PoolingSFA 环境自检 =====")
    _check_platform()
    torch, _ = _check_torch()
    pkg_dir = _check_vllm_ascend(torch)
    _check_vendor(pkg_dir)
    _check_memfabric()
    _check_ulimit()
    _check_clang()
    if args.model_config:
        _check_model_config(args.model_config, args.topk_buffer_size, args.block_size)

    # 环境变量提示（不判失败）
    for key in ("VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_ASCEND_ENABLE_FLASHCOMM1"):
        record(PASS, f"env {key}", os.environ.get(key, "（未设置）"))
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("0", "", "false", "False"):
        record(FAIL, "VLLM_BATCH_INVARIANT", "已开启：会强制关闭 custom op")

    print("\n提示：还需人工确认 --additional-config 中 kv_offload_decode_config.enabled/use_fused_overlap=true，"
          "且 enable_dsa_cp/enable_sparse_sfa_c8 未开启。")
    return summarize("环境自检")


# ---------------------------------------------------------------------------
# OpenAI 请求
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: float, api_key: str | None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")
        return json.loads(body)


def _extract_text(response: dict, api: str) -> str:
    try:
        if api == "chat":
            return response["choices"][0]["message"]["content"] or ""
        return response["choices"][0].get("text", "") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"响应结构异常：{str(response)[:500]}") from exc


def _request_once(args: argparse.Namespace, base_url: str, prompt: str | None = None) -> tuple[str, float, dict]:
    url = base_url.rstrip("/") + ("/v1/chat/completions" if args.api == "chat" else "/v1/completions")
    prompt = args.prompt if prompt is None else prompt
    if args.api == "chat":
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 1,
            "stream": False,
        }
    else:
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 1,
            "stream": False,
        }
    if args.seed is not None:
        payload["seed"] = args.seed
    start = time.time()
    response = _post_json(url, payload, args.timeout, args.api_key)
    elapsed = time.time() - start
    return _extract_text(response, args.api), elapsed, response


def _usage(response: dict) -> str:
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return "usage 未返回"
    return f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"


def _read_prompts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def cmd_service(args: argparse.Namespace) -> int:
    print(f"===== 服务验证：{args.base_url} =====")

    if args.prompt_file:
        try:
            prompts = _read_prompts(args.prompt_file)
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "prompt 文件读取", f"{type(exc).__name__}: {exc}")
            return summarize("服务验证")
        if not prompts:
            record(FAIL, "prompt 文件", "没有有效行")
            return summarize("服务验证")

        records = []
        latencies = []
        for index, prompt in enumerate(prompts):
            try:
                text, elapsed, response = _request_once(args, args.base_url, prompt)
                latencies.append(elapsed)
                ok = bool(text.strip())
                records.append({
                    "index": index,
                    "prompt": prompt,
                    "output": text,
                    "latency_ms": round(elapsed * 1000, 3),
                    "ok": ok,
                    "usage": response.get("usage"),
                })
                if not ok:
                    record(FAIL, f"第 {index} 条输出为空", prompt[:60])
            except Exception as exc:  # noqa: BLE001
                records.append({"index": index, "prompt": prompt, "output": "", "ok": False,
                                "error": f"{type(exc).__name__}: {exc}"})
                record(FAIL, f"第 {index} 条请求失败", f"{type(exc).__name__}: {exc}")
            if (index + 1) % 20 == 0 or index + 1 == len(prompts):
                print(f"    进度：{index + 1}/{len(prompts)}")

        ok_count = sum(1 for r in records if r.get("ok"))
        record(PASS if ok_count == len(records) else FAIL, "批量请求",
               f"成功 {ok_count}/{len(records)}")
        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies) // 2] * 1000
            p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] * 1000
            record(PASS, "单请求耗时",
                   f"p50={p50:.1f} ms, p95={p95:.1f} ms, max={max(latencies) * 1000:.1f} ms")

        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    for row in records:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                record(PASS, "结果落盘", args.output)
            except Exception as exc:  # noqa: BLE001
                record(FAIL, "结果落盘", f"{type(exc).__name__}: {exc}")
        if args.log_file:
            _scan_log(args.log_file, require_fused=True)
        return summarize("服务验证（批量）")

    try:
        text, elapsed, response = _request_once(args, args.base_url)
    except urllib.error.URLError as exc:
        record(FAIL, "请求", f"连接失败：{exc}")
        return summarize("服务验证")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "请求", f"{type(exc).__name__}: {exc}")
        return summarize("服务验证")

    record(PASS, "HTTP 请求", f"{elapsed * 1000:.1f} ms")
    record(PASS if text.strip() else FAIL, "输出非空", f"{len(text)} 字符")
    print(f"    输出摘要：{text[:200]!r}")
    print(f"    {_usage(response)}")
    if args.log_file:
        _scan_log(args.log_file, require_fused=True)
    else:
        print("    提示：请用 `log` 子命令或人工确认日志包含 [fused_overlap_offload][decode]")
    return summarize("服务验证")


def cmd_ab(args: argparse.Namespace) -> int:
    print("===== A/B 对比 =====")
    outcomes = {}
    for label, url in (("A", args.base_url_a), ("B", args.base_url_b)):
        try:
            text, elapsed, response = _request_once(args, url)
            outcomes[label] = (text, elapsed, response)
            record(PASS, f"端点 {label}", f"{elapsed * 1000:.1f} ms，输出 {len(text)} 字符")
        except Exception as exc:  # noqa: BLE001
            record(FAIL, f"端点 {label}", f"{type(exc).__name__}: {exc}")

    if len(outcomes) == 2:
        text_a, time_a, _ = outcomes["A"]
        text_b, time_b, _ = outcomes["B"]
        same = text_a == text_b
        ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
        record(PASS if same else (PASS if ratio >= args.min_similarity else FAIL),
               "输出一致性",
               f"完全一致={same}, 相似度={ratio:.4f}, 阈值={args.min_similarity}")
        if not same:
            print(f"    A: {text_a[:160]!r}")
            print(f"    B: {text_b[:160]!r}")
            print("    提示：bf16 + 不同 TP 布局可能带来数值差异；严格判定请比较 logprobs 或业务评测集。")
        print(f"    耗时：A={time_a * 1000:.1f} ms, B={time_b * 1000:.1f} ms, "
              f"B/A={time_b / time_a:.2f}x" if time_a else "")
    return summarize("A/B 对比")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmd_compare(args: argparse.Namespace) -> int:
    print(f"===== 批量结果对比：A={args.file_a} B={args.file_b} =====")
    try:
        rows_a = _read_jsonl(args.file_a)
        rows_b = _read_jsonl(args.file_b)
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "结果文件读取", f"{type(exc).__name__}: {exc}")
        return summarize("批量结果对比")

    if len(rows_a) != len(rows_b):
        record(FAIL, "条数一致", f"A={len(rows_a)}, B={len(rows_b)}")
        return summarize("批量结果对比")
    record(PASS, "条数一致", str(len(rows_a)))

    pairs = []
    exact = 0
    worst = []
    for row_a, row_b in zip(rows_a, rows_b):
        text_a = row_a.get("output", "")
        text_b = row_b.get("output", "")
        same = text_a == text_b
        ratio = 1.0 if same else difflib.SequenceMatcher(None, text_a, text_b).ratio()
        exact += int(same)
        pairs.append((row_a.get("index"), ratio))
        worst.append((ratio, row_a.get("index"), text_a[:120], text_b[:120]))

    match_ratio = exact / len(rows_a) if rows_a else 0.0
    avg_ratio = sum(r for _, r in pairs) / len(pairs) if pairs else 0.0
    record(PASS if match_ratio >= args.min_match_ratio else FAIL, "完全一致率",
           f"{match_ratio:.4f}（阈值 {args.min_match_ratio}），平均相似度 {avg_ratio:.4f}")
    below = sum(1 for _, r in pairs if r < args.min_similarity)
    record(PASS if below == 0 else FAIL, f"相似度低于 {args.min_similarity} 的条数", str(below))

    worst.sort(key=lambda x: x[0])
    if worst and worst[0][0] < 1.0:
        print("    最不一致的 5 条：")
        for ratio, index, text_a, text_b in worst[:5]:
            print(f"      #{index} ratio={ratio:.4f}")
            print(f"        A: {text_a!r}")
            print(f"        B: {text_b!r}")
    if args.report:
        try:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump({
                    "count": len(rows_a),
                    "exact_match_ratio": match_ratio,
                    "average_similarity": avg_ratio,
                    "below_threshold": below,
                }, f, ensure_ascii=False, indent=2)
            record(PASS, "报告落盘", args.report)
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "报告落盘", f"{type(exc).__name__}: {exc}")
    return summarize("批量结果对比")


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------

FATAL_PATTERNS = (
    "fused_overlap offload requires custom op",
    "Failed to stat",
    "offload launch load library failed",
    "offload.empty failed",
    "KV Offload Decode is only supported in PD",
    "KV Offload Decode don't support compress",
    "KV offload decode currently requires TP without context parallelism",
    "does not support the sparse SFA C8 main cache",
    "external fused_overlap planner requires one TopK head",
    "Traceback (most recent call last)",
)

FUSED_MARKER = "[fused_overlap_offload][decode]"
POOL_MARKER = "KVOffloadDecodeManager starts CPU KV pool initialization"


def _scan_log(path: str, require_fused: bool = True) -> None:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "日志读取", f"{type(exc).__name__}: {exc}")
        return

    fused = [ln.strip() for ln in lines if FUSED_MARKER in ln]
    pool = [ln.strip() for ln in lines if POOL_MARKER in ln]
    fatals = [(pat, ln.strip()) for ln in lines for pat in FATAL_PATTERNS if pat in ln]

    record(PASS if pool else WARN, "CPU KV 池初始化标记",
           f"{len(pool)} 条" + (f"，示例：{pool[-1][:180]}" if pool else "（未出现）"))
    record(PASS if fused else FAIL, "融合算子命中标记",
           f"{len(fused)} 条" + (f"，示例：{fused[-1][:180]}" if fused else "（未出现）"))
    if not fused:
        print("    排查：①是否只有 prefill 请求；②use_fused_overlap 是否为 true；"
              "③是否加载了旧包；④日志是否被轮转")
    if fatals:
        for pat, line in fatals[:5]:
            record(FAIL, f"致命日志：{pat}", line[:200])
    else:
        record(PASS, "致命错误扫描", "未命中已知模式")


def cmd_log(args: argparse.Namespace) -> int:
    print(f"===== 日志扫描：{args.log_file} =====")
    _scan_log(args.log_file, require_fused=not args.no_require_fused)
    return summarize("日志扫描")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PoolingSFA 部署自检脚本")
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="检查环境、包、算子注册、vendor、MemFabric、ulimit")
    env.add_argument("--model-config", default=None, help="模型 config.json 路径（可选）")
    env.add_argument("--topk-buffer-size", type=int, default=4096, help="计划使用的 topk_buffer_size")
    env.add_argument("--block-size", type=int, default=128, help="KV block size（本路径要求 128）")
    env.set_defaults(func=cmd_env)

    def add_request_args(p: argparse.ArgumentParser, with_log: bool = False,
                         with_base_url: bool = True, with_batch: bool = False) -> None:
        if with_base_url:
            p.add_argument("--base-url", default="http://127.0.0.1:8000", help="服务地址")
        p.add_argument("--model", required=True, help="served-model-name")
        p.add_argument("--prompt", default="用一句话解释注意力机制。", help="测试 prompt")
        p.add_argument("--api", choices=("chat", "completions"), default="chat", help="OpenAI 接口类型")
        p.add_argument("--max-tokens", type=int, default=32, help="最大输出 token 数")
        p.add_argument("--temperature", type=float, default=0.0, help="采样温度（验收建议 0）")
        p.add_argument("--seed", type=int, default=0, help="随机种子；服务不支持时可传 -1 去掉")
        p.add_argument("--timeout", type=float, default=120.0, help="单请求超时（秒）")
        p.add_argument("--api-key", default=None, help="可选 API Key")
        if with_batch:
            p.add_argument("--prompt-file", default=None, help="批量 prompt 文件（每行一条）")
            p.add_argument("--output", default=None, help="批量结果 JSONL 输出路径")
        if with_log:
            p.add_argument("--log-file", default=None, help="服务日志路径（可选，用于自动检查融合标记）")

    service = sub.add_parser("service", help="发固定请求或批量请求并校验")
    add_request_args(service, with_log=True, with_batch=True)
    service.set_defaults(func=cmd_service)

    ab = sub.add_parser("ab", help="两个端点 A/B 对比")
    add_request_args(ab, with_base_url=False)
    ab.add_argument("--base-url-a", required=True, help="端点 A（如 baseline）")
    ab.add_argument("--base-url-b", required=True, help="端点 B（如融合路径）")
    ab.add_argument("--min-similarity", type=float, default=0.95, help="输出相似度阈值")
    ab.set_defaults(func=cmd_ab)

    compare = sub.add_parser("compare", help="对比两份批量结果 JSONL（整网精度）")
    compare.add_argument("--file-a", required=True, help="A 结果（如 baseline.jsonl）")
    compare.add_argument("--file-b", required=True, help="B 结果（如 fused.jsonl）")
    compare.add_argument("--min-match-ratio", type=float, default=0.99, help="完全一致率阈值")
    compare.add_argument("--min-similarity", type=float, default=0.95, help="单条相似度阈值")
    compare.add_argument("--report", default=None, help="汇总报告 JSON 输出路径")
    compare.set_defaults(func=cmd_compare)

    log = sub.add_parser("log", help="扫描服务日志")
    log.add_argument("--log-file", required=True, help="服务日志路径")
    log.add_argument("--no-require-fused", action="store_true", help="不要求出现融合标记（仅看错误）")
    log.set_defaults(func=cmd_log)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "seed", 0) is not None and getattr(args, "seed", 0) < 0:
        args.seed = None
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 未处理异常：{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
