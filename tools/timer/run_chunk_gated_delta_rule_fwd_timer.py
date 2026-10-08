"""chunk_gated_delta_rule_fwd A5 Phase6 融合 kernel timer 采集脚本。

用法（在装有 timer 分支构建的 wheel 的 NPU 环境）:

  python run_chunk_gated_delta_rule_fwd_timer.py                    # 默认 shape
  python run_chunk_gated_delta_rule_fwd_timer.py --tokens 11274     # 指定 T
  python run_chunk_gated_delta_rule_fwd_timer.py --key-heads 1 --value-heads 8

流程（与参考仓 vllm-ascend fused_sparse_attention_overlap 的 timer 用法一致）:
  1. 按 tools/timer/parse_timer_csv.py 的 TOTAL_BUFFER_SIZE 分配 int64 NPU 张量;
  2. 调用 fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_fwd(..., timer=buf);
  3. torch.npu.synchronize() 后把缓冲落盘为 raw CSV;
  4. parse_timer_csv 解析为每核每阶段表并打印汇总;
  5. trace_parser 生成 Perfetto 可打开的 trace JSON。

路径说明（重要）:
  A5/950 上 UsePreparePath 为 true 的 shape（4D BF16 + K=V=128 + chunk=64 +
  hv/hq<=4）会拆成 Prepare/FwdH/FwdO 三个独立 launch，不进入 Phase6 融合
  kernel，timer 打不到点。默认参数取 value-heads/key-heads = 8 (GVA>4) 强制
  走 Phase6；如需分析其它 shape，可用 --value-dim 256、--chunk-size 128 或
  --dtype fp16 同样绕开 prepare 路径。
"""

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TIMER_DIR = Path(__file__).resolve().parent

try:
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd
except ImportError:
    sys.path.insert(0, str(_REPO_ROOT / "torch_custom" / "fla_npu"))
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd

sys.path.insert(0, str(_TIMER_DIR))
import parse_timer_csv  # noqa: E402
import trace_parser  # noqa: E402

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401


def build_args():
    parser = argparse.ArgumentParser(description="chunk_gated_delta_rule_fwd timer 采集")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--key-heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--key-dim", type=int, default=128, choices=(128,))
    parser.add_argument("--value-dim", type=int, default=128, choices=(128, 256))
    parser.add_argument("--chunk-size", type=int, default=64, choices=(64, 128))
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--layout", default="BNSD", choices=("BNSD", "BSND", "NTD", "TND"))
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--initial-state", action="store_true", help="传入初始状态并输出最终状态 (B30/tiling key 301)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=1, help="计时采集次数（每次重新清零 timer 并落盘）")
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "timer"))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_inputs(args, dtype, device):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    b, t, hk, hv, k, v = args.batch, args.tokens, args.key_heads, args.value_heads, args.key_dim, args.value_dim
    if args.layout in ("BNSD",):
        q = torch.rand(b, hk, t, k, dtype=torch.float32, generator=generator).to(dtype).npu()
        kk = torch.rand(b, hk, t, k, dtype=torch.float32, generator=generator).to(dtype).npu()
        vv = torch.rand(b, hv, t, v, dtype=torch.float32, generator=generator).to(dtype).npu()
    else:  # BSND / TND：物理头维在 dim1
        q = torch.rand(b, t, hk, k, dtype=torch.float32, generator=generator).to(dtype).npu()
        kk = torch.rand(b, t, hk, k, dtype=torch.float32, generator=generator).to(dtype).npu()
        vv = torch.rand(b, t, hv, v, dtype=torch.float32, generator=generator).to(dtype).npu()
    g = torch.nn.functional.logsigmoid(
        torch.randn(b, t, hv, dtype=torch.float32, generator=generator)).npu()
    beta = torch.rand(b, t, hv, dtype=torch.float32, generator=generator).sigmoid().to(dtype).npu()
    initial_state = None
    if args.initial_state:
        initial_state = torch.randn(b, hv, k, v, dtype=torch.float32, generator=generator).to(dtype).npu()
    return q, kk, vv, g, beta, initial_state


def dump_timer_csv(timer, path):
    data = timer.detach().cpu().tolist()
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(data)


def main():
    args = build_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = "npu"

    gva = args.value_heads // args.key_heads
    if args.dtype == "bf16" and args.value_dim == 128 and args.chunk_size == 64 and gva <= 4:
        print("[WARNING] 当前参数组合在 A5 上会走 Prepare/FwdH/FwdO 三算子路径，"
              "Phase6 融合 kernel 不执行，timer 不会有数据。")
        print("[WARNING] 建议任选其一绕开：--value-heads 8 --key-heads 1 (GVA>4)、"
              "--chunk-size 128、--value-dim 256、--dtype fp16。")

    q, k, v, g, beta, initial_state = make_inputs(args, dtype, device)
    print(f"[INFO] shape: B={args.batch} T={args.tokens} Hk={args.key_heads} Hv={args.value_heads} "
          f"K={args.key_dim} V={args.value_dim} chunk={args.chunk_size} dtype={args.dtype} "
          f"layout={args.layout} initial_state={initial_state is not None}")

    total = parse_timer_csv.TOTAL_BUFFER_SIZE
    print(f"[INFO] timer buffer: {total} int64 ({total * 8 / 1024 / 1024:.2f} MB)")
    timer = torch.zeros(total, dtype=torch.int64, device=device)

    call_kwargs = dict(
        initial_state=initial_state,
        output_final_state=initial_state is not None,
        chunk_size=args.chunk_size,
        scale=args.scale,
        layout=args.layout,
    )

    for _ in range(args.warmup):
        npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **call_kwargs)
    torch.npu.synchronize()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "rank_0_npu_time.csv"

    for it in range(args.iters):
        timer.zero_()
        npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, timer=timer, **call_kwargs)
        torch.npu.synchronize()
        suffix = "" if args.iters == 1 else f"_{it}"
        it_raw_path = raw_path if args.iters == 1 else out_dir / f"rank_0_npu_time{suffix}.csv"
        dump_timer_csv(timer, it_raw_path)
        print(f"[INFO] raw timer CSV -> {it_raw_path}")

        parsed_path = it_raw_path.with_name(it_raw_path.stem + "_parsed.csv")
        parse_timer_csv.parse_and_dump(str(it_raw_path), str(parsed_path))
        parse_timer_csv.print_summary(str(parsed_path))
        trace_parser.generate_trace_files(str(parsed_path), out_dir, rank=0)
    print(f"[INFO] 全部产物已写入 {out_dir}")


if __name__ == "__main__":
    main()
