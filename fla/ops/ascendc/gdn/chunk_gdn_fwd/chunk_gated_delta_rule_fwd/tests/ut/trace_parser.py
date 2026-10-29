#!/usr/bin/env python3
"""把 parse_timer_csv.py 产出的逐核 CSV 转成 Chrome trace（Perfetto）JSON。

每个切片单独出一份 trace：``rank_<r>_<slice>_parsed.csv`` -> ``rank_<r>_<slice>.json``。
仅在核实了 cycles_per_us 与 AIC/AIV 共同时钟后才使用 ``--shared-clock``。
"""

import argparse
import csv
import json
from pathlib import Path

from parse_timer_csv import (
    KERNEL_TIMING_NAME,
    is_accumulated,
    resolve_slice,
    validate_frequency,
)

SLICE_ALIASES = ("prepare", "fwd_h", "fwd_o", "phase6")
ANCHOR_NAME = "ANCHOR"


def read_csv_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def collect_event_names(fieldnames):
    events = []
    fields = set(fieldnames)
    for col in fieldnames:
        if not col.endswith("_start_cycle"):
            continue
        name = col[: -len("_start_cycle")]
        if name == KERNEL_TIMING_NAME:
            continue
        if name == ANCHOR_NAME:
            continue
        if is_accumulated(name):
            continue
        if f"{name}_end_cycle" in fields:
            events.append(name)
    return events


def find_min_start(csv_path):
    """Return the minimum valid start cycle within one rank's CSV.

    Each rank runs on a separate NPU whose hardware cycle counter has its own
    absolute origin, so every rank must be zeroed against its own minimum start.
    ANCHOR is intentionally NOT used here: it is recorded at the body entry on
    every core without a rendezvous, so subtracting it per core only gives a
    "time inside this core's body" view. That view is opt-in (--align-anchor).
    """
    if not csv_path.exists():
        return 0
    fieldnames, rows = read_csv_rows(csv_path)
    min_start = None
    for event in collect_event_names(fieldnames):
        start_col = f"{event}_start_cycle"
        end_col = f"{event}_end_cycle"
        for row in rows:
            start = int(row[start_col])
            end = int(row[end_col])
            if start > 0 and end >= start:
                min_start = start if min_start is None else min(min_start, start)
    return min_start if min_start is not None else 0


def anchor_spread_us(fieldnames, rows, cycles_per_us):
    """Spread of per-core ANCHOR: the direct proof (or refutation) of a common base."""
    col = f"{ANCHOR_NAME}_start_cycle"
    if col not in set(fieldnames):
        return None
    vals = [int(row[col]) for row in rows if int(row[col]) > 0]
    if not vals:
        return None
    return (max(vals) - min(vals)) / cycles_per_us


def _interval(row, name):
    start_col, end_col = f"{name}_start_cycle", f"{name}_end_cycle"
    if start_col not in row or end_col not in row:
        return None
    start, end = int(row[start_col]), int(row[end_col])
    return (start, end) if start > 0 and end >= start else None


def check_dependencies(fieldnames, rows, cycles_per_us, tol_us=0.5):
    """Assert the producer/consumer order that the kernel's flags guarantee.

    Every check is a necessary condition of the code, independent of how the
    timestamps happen to line up:
      * AIC KKT         after AIV stage1 upload (NotifyAicStage1Done)
      * AIC solve issue after AIV stage3 epilogue (WaitAivStage3Done)
      * AIC W/U issue   after AIV stage6 WU_K upload (WaitAivStage6Done)
    Returns (violations, lines) where each line is a printable report row.
    """
    groups = {}
    for row in rows:
        groups.setdefault(int(row["group_id"]), {})[(row["core_type"], int(row["sub_id"]))] = row
    violations, lines = [], []

    def first(rows_of_group, event):
        got = [_interval(r, event) for r in rows_of_group.values() if r is not None]
        got = [g for g in got if g]
        return got[0] if got else None

    def span(rows_of_group, event):
        got = [_interval(r, event) for r in rows_of_group.values() if r is not None]
        got = [g for g in got if g]
        return (min(g[0] for g in got), max(g[1] for g in got)) if got else None

    checks = (
        ("KKT vs stage1", "AIC", "S2_KKT_0", "AIV", "S1A_KUPLOAD_0", "start"),
        ("ISSUE vs stage3", "AIC", "S4_ISSUE_0", "AIV", "S3B_EPI_0", "start"),
        ("W_ISSUE vs WU_K", "AIC", "S7_W_0", "AIV", "S6_KBG_0", "end"),
    )
    for group_id, members in sorted(groups.items()):
        aic = {k: v for k, v in members.items() if k[0] == "AIC"}
        aiv = {k: v for k, v in members.items() if k[0] == "AIV"}
        if not aic or not aiv:
            continue
        for label, _, cevent, _, pevent, mode in checks:
            consumer, producer = first(aic, cevent), span(aiv, pevent)
            if consumer is None or producer is None:
                continue
            ref = producer[0] if mode == "start" else producer[1]
            delta = (consumer[0] - ref) / cycles_per_us
            ok = delta >= -tol_us
            lines.append(f"  g{group_id:02d} {label:18s} {delta:+8.2f} us  {'OK' if ok else 'VIOLATION'}")
            if not ok:
                violations.append((group_id, label, delta))
    return violations, lines


def generate_trace_json(csv_path, rank, base_cycle, cycles_per_us=None, shared_clock=False,
                        align_anchor=False):
    validate_frequency(cycles_per_us)
    if cycles_per_us is None or not shared_clock:
        raise ValueError("Perfetto requires verified cycles_per_us and shared_clock=True")
    fieldnames, rows = read_csv_rows(csv_path)
    events = collect_event_names(fieldnames)
    trace_events = []

    anchor_col = f"{ANCHOR_NAME}_start_cycle"
    for row in rows:
        group_id = int(row["group_id"])
        core_type = row["core_type"]
        sub_id = int(row["sub_id"])
        tid = f"group_{group_id}_{core_type}_{sub_id}"
        pid = f"rank{rank}"
        # ANCHOR 记录在 body 起点，且不强制全核会合；只有显式 --align-anchor 时
        # 才用它做每核原点（视图变成“各自 body 内的时间”），默认仍用全局 min-start。
        row_anchor = int(row[anchor_col]) if (align_anchor and anchor_col in row) else 0
        base = row_anchor if row_anchor > 0 else base_cycle

        for event in events:
            start_col = f"{event}_start_cycle"
            end_col = f"{event}_end_cycle"
            start = int(row[start_col])
            end = int(row[end_col])
            if (start, end) == (0, 0):
                continue
            if start <= 0 or end < start:
                # start 可以早于 base（ANCHOR 之前的 Init 就是这种情况），
                # 导出后表现为负 ts，Perfetto 能正常显示。
                raise ValueError(f"invalid trace interval: {event}")

            ts = (start - base) / cycles_per_us
            dur = (end - start) / cycles_per_us
            trace_events.append({
                "ph": "X",
                "cat": "aic_op" if core_type == "AIC" else "aiv_op",
                "pid": pid,
                "tid": tid,
                "name": event,
                "ts": round(ts, 6),
                "dur": round(dur, 6),
                "args": {
                    "rank": rank,
                    "group_id": group_id,
                    "core_type": core_type,
                    "sub_id": sub_id,
                    "start_cycle": start,
                    "end_cycle": end,
                },
            })
            if start == end:
                trace_events[-1].update(ph="i", s="t")
                trace_events[-1].pop("dur")

    return trace_events


def main():
    parser = argparse.ArgumentParser(description="Generate Chrome trace JSON from parsed GDN timer CSV files")
    parser.add_argument("--timer-dir", default="./timer", help="Directory containing rank_X_<slice>_parsed.csv")
    parser.add_argument("--output-dir", default="./timer_json", help="Directory for generated trace JSON")
    parser.add_argument("--rank-start", type=int, default=0, help="Start rank, inclusive")
    parser.add_argument("--rank-end", type=int, default=1, help="End rank, inclusive")
    parser.add_argument("--slice", action="append", default=None, choices=sorted(SLICE_ALIASES),
                        help="slice to export, repeatable (default prepare/fwd_h/fwd_o)")
    parser.add_argument("--output", default="all_rank_trace.json", help="Merged trace JSON file name")
    parser.add_argument("--cycles-per-us", type=float, required=True, help="verified clock frequency")
    parser.add_argument("--shared-clock", action="store_true", help="assert verified common clock")
    parser.add_argument("--tol-us", type=float, default=0.5, help="allowed dependency violation (us)")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when a dependency is violated")
    parser.add_argument("--align-anchor", action="store_true",
                        help="use each core's ANCHOR as its origin (per-core body view)")
    args = parser.parse_args()
    validate_frequency(args.cycles_per_us)
    if not args.shared_clock:
        parser.error("trace requires --shared-clock")

    slices = args.slice if args.slice else ["prepare", "fwd_h", "fwd_o"]
    timer_dir = Path(args.timer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for slice_name in slices:
        resolve_slice(slice_name)  # 早失败：切片名必须能映射到 KernelSlice
        all_events = []
        for rank in range(args.rank_start, args.rank_end + 1):
            csv_path = timer_dir / f"rank_{rank}_{slice_name}_parsed.csv"
            if not csv_path.exists():
                print(f"[SKIP] rank_{rank}/{slice_name}: 文件不存在 {csv_path}")
                continue
            base_cycle = find_min_start(csv_path)
            fieldnames, rows = read_csv_rows(csv_path)
            spread = anchor_spread_us(fieldnames, rows, args.cycles_per_us)
            violations, lines = check_dependencies(fieldnames, rows, args.cycles_per_us, args.tol_us)
            anchor_tag = "per-core ANCHOR" if (spread is not None and args.align_anchor) else "min-start"
            print(f"[CHECK] rank_{rank}/{slice_name}: {'PASS' if not violations else 'FAIL'}"
                  f" (base={anchor_tag})")
            if spread is not None:
                print(f"[ANCHOR] cores={len(rows)}  body-entry spread={spread:.3f} us"
                      f"  (start skew + counter base; not a sync proof)")
            for line in lines:
                print(line)
            if violations:
                print(f"[CHECK] {len(violations)} violation(s); first = {violations[0]}")
                if args.strict:
                    raise SystemExit(2)
            rank_events = generate_trace_json(csv_path, rank, base_cycle, args.cycles_per_us,
                                              args.shared_clock, args.align_anchor)
            all_events.extend(rank_events)
            rank_output = output_dir / f"rank_{rank}_{slice_name}.json"
            with open(rank_output, "w", encoding="utf-8") as f:
                json.dump({"traceEvents": rank_events, "displayTimeUnit": "us"}, f, indent=2, ensure_ascii=False, allow_nan=False)
            print(f"[INFO] rank_{rank}/{slice_name}: {len(rank_events)} events -> {rank_output}")

        merged_output = output_dir / f"{slice_name}_{args.output}"
        with open(merged_output, "w", encoding="utf-8") as f:
            json.dump({"traceEvents": all_events, "displayTimeUnit": "us"}, f, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"[INFO] merged {slice_name}: {len(all_events)} events -> {merged_output}")


if __name__ == "__main__":
    main()
