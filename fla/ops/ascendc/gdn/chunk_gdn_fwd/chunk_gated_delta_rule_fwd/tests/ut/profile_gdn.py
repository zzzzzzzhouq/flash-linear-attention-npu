#!/usr/bin/env python3
"""采集 chunk_gated_delta_rule_fwd（A5 Prepare/FwdH/FwdO 三算子路径）的核内计时。

做法：申请一个 ``torch.zeros(AscendTimer::TOTAL_BUFFER_WORDS, int64, npu)`` 计时缓冲，
通过 stable/ctypes 入口的 ``timer`` 参数传进 public op，运行一次后同步，导出

  <out>/rank_<rank>_npu_time.csv            原始 int64 缓冲（一行，parse_timer_csv.py 的输入）
  <out>/rank_<rank>_<slice>_parsed.csv      逐核逐计时项 CSV（prepare / fwd_h / fwd_o）
  <out>/rank_<rank>_<slice>.json            Chrome trace（仅在给定 --cycles-per-us 时）
  <out>/metadata.json                       本次采集的 shape / dtype / 切片 / 频率

入口说明：stable ABI（``torch_custom/.../stable_chunk_gated_delta_rule_fwd.cpp``）目前把
timer 固定传 ``std::nullopt``，所以这里走 ctypes 直调 ``aclnnChunkGatedDeltaRuleFwd``
（``fla_npu.ops.ascendc._aclnn_ctypes.npu_chunk_gated_delta_rule_fwd(..., timer=timer)``）。

用法::

    python profile_gdn.py --device 0 --out profiling_results/gdn_fwd_timer
    python profile_gdn.py --device 0 --cycles-per-us 1800   # 核实过时钟才给，用于 us + trace
    python profile_gdn.py --device 0 --no-timer             # 基线（不传 timer，验证零开销路径）
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import parse_timer_csv  # noqa: E402  （同目录，负责常量与 CSV 解析）
import trace_parser  # noqa: E402


def build_inputs(torch, device, batch, tokens, k_heads, v_heads, k_dim, v_dim):
    """构造 A5 Prepare 路径支持的一组输入（BF16 + K=V=128 + chunk=64 + GVA<=4）。

    注意 public op 的 g/beta 是 BTH（[B, T, Hv]），q/k/v 按 layout=BNSD 给 [B, H, T, D]。
    """
    q = torch.randn(batch, k_heads, tokens, k_dim, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(batch, k_heads, tokens, k_dim, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(batch, v_heads, tokens, v_dim, dtype=torch.bfloat16, device=device) * 0.1
    # g 是 log 衰减（负值），beta 在 (0,1)；两者都可以是 fp32。
    g = -torch.rand(batch, tokens, v_heads, dtype=torch.float32, device=device) * 0.05
    beta = torch.rand(batch, tokens, v_heads, dtype=torch.float32, device=device) * 0.5 + 0.25
    return q, k, v, g, beta


def write_raw(path, words):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(words)


def main():
    parser = argparse.ArgumentParser(description="采集 GDN 三算子路径的 AscendTimerV2 计时")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", default="profiling_results/gdn_fwd_timer")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--k-heads", type=int, default=2)
    parser.add_argument("--v-heads", type=int, default=4)
    parser.add_argument("--k-dim", type=int, default=128)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--use-exp2", action="store_true")
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--no-timer", action="store_true", help="不传 timer，跑基线")
    parser.add_argument("--cycles-per-us", type=float, default=None,
                        help="已核实的时钟频率；给了才导出 us 与 Perfetto trace")
    parser.add_argument("--shared-clock", action="store_true",
                        help="确认 AIC/AIV 共同时钟后再加，用于 trace")
    args = parser.parse_args()

    import torch  # noqa: F401  （必须先 import torch 再 import torch_npu）
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc._aclnn_ctypes import npu_chunk_gated_delta_rule_fwd

    device = f"npu:{args.device}"
    torch.npu.set_device(device)

    q, k, v, g, beta = build_inputs(torch, device, args.batch, args.tokens,
                                    args.k_heads, args.v_heads, args.k_dim, args.v_dim)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timer = None
    if not args.no_timer:
        timer = torch.zeros(parse_timer_csv.TOTAL_BUFFER_WORDS, dtype=torch.int64, device=device)

    o, final_state, g_cumsum, a_out, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd = \
        npu_chunk_gated_delta_rule_fwd(
            q, k, v, g, beta,
            output_final_state=False,
            chunk_size=64,
            scale=args.scale,
            use_exp2=args.use_exp2,
            use_qk_l2norm_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            state_v_first=args.state_v_first,
            return_intermediate_states=False,
            disable_recompute=True,
            layout="BNSD",
            timer=timer,
        )
    torch.npu.synchronize()

    finite = bool(torch.isfinite(o.float()).all().item())
    metadata = {
        "op": "npu_chunk_gated_delta_rule_fwd",
        "path": "A5 Prepare/FwdH/FwdO (UsePreparePath)",
        "device": device,
        "shape": {
            "batch": args.batch, "tokens": args.tokens,
            "k_heads": args.k_heads, "v_heads": args.v_heads,
            "k_dim": args.k_dim, "v_dim": args.v_dim, "chunk_size": 64,
        },
        "flags": {
            "use_exp2": args.use_exp2,
            "state_v_first": args.state_v_first,
            "timer_enabled": timer is not None,
        },
        "timer_words": parse_timer_csv.TOTAL_BUFFER_WORDS,
        "slice_words": parse_timer_csv.SLICE_WORDS,
        "output_finite": finite,
        "cycles_per_us": args.cycles_per_us,
    }

    if timer is None:
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 未启用 timer（基线）；输出有限性检查：{finite}")
        return

    if not bool(timer.any().item()):
        print("[WARNING] timer 全 0：确认 kernel 是否为本分支构建的版本（未接线的旧包不会写计时）",
              file=sys.stderr)

    raw_path = out_dir / f"rank_{args.rank}_npu_time.csv"
    write_raw(raw_path, timer.cpu().tolist())
    print(f"[INFO] raw -> {raw_path}")

    combined_events = []
    for slice_name in ("prepare", "fwd_h", "fwd_o"):
        slice_index = parse_timer_csv.resolve_slice(slice_name)
        parsed_path = out_dir / f"rank_{args.rank}_{slice_name}_parsed.csv"
        try:
            parse_timer_csv.parse_and_dump(str(raw_path), str(parsed_path),
                                           args.cycles_per_us, slice_index)
        except ValueError as exc:
            print(f"[INFO] {slice_name}: {exc}")
            continue
        if args.cycles_per_us is not None and args.shared_clock:
            base_cycle = trace_parser.find_min_start(parsed_path)
            events = trace_parser.generate_trace_json(parsed_path, args.rank, base_cycle,
                                                      args.cycles_per_us, True)
            trace_path = out_dir / f"rank_{args.rank}_{slice_name}.json"
            trace_path.write_text(
                json.dumps({"traceEvents": events, "displayTimeUnit": "us"},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[INFO] {slice_name}: {len(events)} trace events -> {trace_path}")
            # 合并视图：每个 slice 一个独立进程（各自 origin），避免伪造跨 launch 的相对时间。
            slice_pid = f"rank{args.rank}_{slice_name}"
            for event in events:
                event["pid"] = slice_pid
            combined_events.extend(events)
        elif args.cycles_per_us is not None and not args.shared_clock:
            print(f"[INFO] {slice_name}: 未给 --shared-clock，不生成 trace")

    if combined_events:
        combined_path = out_dir / f"rank_{args.rank}_gdn_timer.json"
        combined_path.write_text(
            json.dumps({"traceEvents": combined_events, "displayTimeUnit": "us"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] combined: {len(combined_events)} trace events -> {combined_path}")

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 完成，结果目录 {out_dir}")


if __name__ == "__main__":
    main()
