"""Run the real GDN A5 forward path and export three independent timer slices."""

import argparse
import json
from pathlib import Path

from gdn_timer_layout import KERNEL_NAMES, TIMER_BUFFER_NUMEL, TIMER_SLICE_NUMEL
from parse_timer_csv import parse_and_dump
from trace_parser import generate_trace_json


def _clone_outputs(outputs):
    return tuple(None if tensor is None else tensor.clone() for tensor in outputs)


def _compare_outputs(reference, actual):
    for index, (expected, observed) in enumerate(zip(reference, actual)):
        if expected is None:
            if observed is not None:
                raise RuntimeError(f"output {index}: diagnostic call returned an unexpected tensor")
            continue
        import torch
        torch.testing.assert_close(observed, expected, rtol=0, atol=0, msg=lambda msg: f"output {index}: {msg}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--k-heads", type=int, default=16)
    parser.add_argument("--v-heads", type=int, default=32)
    parser.add_argument("--k-dim", type=int, default=128, choices=(128,))
    parser.add_argument("--v-dim", type=int, default=128, choices=(128,))
    parser.add_argument("--chunk-size", type=int, default=64, choices=(64,))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-exp2", action="store_true")
    parser.add_argument("--initial-state", action="store_true", help="cover H_V0_CAST with a BF16 initial state")
    parser.add_argument("--cycles-per-us", type=float)
    parser.add_argument("--shared-clock", action="store_true", help="assert a verified common clock for each kernel")
    parser.add_argument("--out", type=Path, default=Path("profiling_results/gdn_stage_timer"))
    args = parser.parse_args()
    if (args.cycles_per_us is None) != (not args.shared_clock):
        parser.error("trace export requires both --cycles-per-us and --shared-clock, or neither")
    if min(args.batch, args.tokens, args.k_heads, args.v_heads) <= 0:
        parser.error("batch/tokens/head counts must be positive")
    if args.v_heads % args.k_heads != 0 or args.v_heads // args.k_heads > 4:
        parser.error("the A5 path requires Hv/Hk in {1,2,3,4}")

    import torch
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc._aclnn_ctypes import npu_chunk_gated_delta_rule_fwd

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    device = torch.device(f"npu:{args.device}")
    q = torch.randn(args.batch, args.k_heads, args.tokens, args.k_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(args.batch, args.v_heads, args.tokens, args.v_dim, dtype=torch.bfloat16, device=device)
    g = torch.randn(args.batch, args.tokens, args.v_heads, dtype=torch.float32, device=device)
    beta = torch.rand(args.batch, args.tokens, args.v_heads, dtype=torch.bfloat16, device=device)
    initial_state = None
    if args.initial_state:
        initial_state = torch.randn(
            args.batch, args.v_heads, args.k_dim, args.v_dim, dtype=torch.bfloat16, device=device
        )

    call_args = dict(
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=args.chunk_size,
        use_exp2=args.use_exp2,
        return_intermediate_states=True,
        layout="BNSD",
    )
    reference = _clone_outputs(npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **call_args))
    torch.npu.synchronize()

    timer = torch.zeros(TIMER_BUFFER_NUMEL, dtype=torch.int64, device=device)
    observed = npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, timer=timer, **call_args)
    torch.npu.synchronize()
    _compare_outputs(reference, observed)

    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "rank_0_gdn_timer.npy"
    raw = timer.cpu().numpy().reshape(len(KERNEL_NAMES), TIMER_SLICE_NUMEL)
    import numpy as np
    np.save(raw_path, raw, allow_pickle=False)

    outputs = {"raw": str(raw_path), "kernels": {}}
    for kernel in KERNEL_NAMES:
        parsed = args.out / f"rank_0_{kernel}_timer_parsed.csv"
        parse_and_dump(raw_path, parsed, kernel, args.cycles_per_us)
        outputs["kernels"][kernel] = {"csv": str(parsed)}
        if args.cycles_per_us is not None:
            trace = args.out / f"rank_0_{kernel}_timer.json"
            generate_trace_json(parsed, kernel, trace, args.cycles_per_us, args.shared_clock)
            outputs["kernels"][kernel]["trace"] = str(trace)

    metadata = {
        "status": "success",
        "ordinary_vs_timer_outputs": "exact_match",
        "shape": {
            "batch": args.batch,
            "tokens": args.tokens,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "k_dim": args.k_dim,
            "v_dim": args.v_dim,
            "chunk_size": args.chunk_size,
        },
        "use_exp2": args.use_exp2,
        "initial_state": args.initial_state,
        "timer_dtype": "int64",
        "timer_slice_numel": TIMER_SLICE_NUMEL,
        "timer_buffer_numel": TIMER_BUFFER_NUMEL,
        "cycles_per_us": args.cycles_per_us,
        "shared_clock_asserted": args.shared_clock,
        "launch_timelines_independent": True,
        "outputs": outputs,
    }
    metadata_path = args.out / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] exact output comparison passed; metadata: {metadata_path}")


if __name__ == "__main__":
    main()
