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
        if is_accumulated(name):
            continue
        if f"{name}_end_cycle" in fields:
            events.append(name)
    return events


def find_min_start(csv_path):
    """Return the minimum valid start cycle within a single rank's CSV.

    Each rank runs on a separate NPU whose hardware cycle counter has its own
    absolute origin, so every rank must be zeroed against its own minimum start
    rather than a shared global base; otherwise ranks with a higher counter
    origin appear to begin tens of milliseconds into the timeline.
    """
    if not csv_path.exists():
        return 0
    min_start = None
    fieldnames, rows = read_csv_rows(csv_path)
    for event in collect_event_names(fieldnames):
        start_col = f"{event}_start_cycle"
        end_col = f"{event}_end_cycle"
        for row in rows:
            start = int(row[start_col])
            end = int(row[end_col])
            if start > 0 and end >= start:
                min_start = start if min_start is None else min(min_start, start)
    return min_start if min_start is not None else 0


def generate_trace_json(csv_path, rank, base_cycle, cycles_per_us=None, shared_clock=False):
    validate_frequency(cycles_per_us)
    if cycles_per_us is None or not shared_clock:
        raise ValueError("Perfetto requires verified cycles_per_us and shared_clock=True")
    fieldnames, rows = read_csv_rows(csv_path)
    events = collect_event_names(fieldnames)
    trace_events = []

    for row in rows:
        group_id = int(row["group_id"])
        core_type = row["core_type"]
        sub_id = int(row["sub_id"])
        tid = f"group_{group_id}_{core_type}_{sub_id}"
        pid = f"rank{rank}"

        for event in events:
            start_col = f"{event}_start_cycle"
            end_col = f"{event}_end_cycle"
            start = int(row[start_col])
            end = int(row[end_col])
            if (start, end) == (0, 0):
                continue
            if start <= 0 or end < start or start < base_cycle:
                raise ValueError(f"invalid trace interval: {event}")

            ts = (start - base_cycle) / cycles_per_us
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
            rank_events = generate_trace_json(csv_path, rank, base_cycle, args.cycles_per_us,
                                              args.shared_clock)
            all_events.extend(rank_events)
            rank_output = output_dir / f"rank_{rank}_{slice_name}.json"
            with open(rank_output, "w", encoding="utf-8") as f:
                json.dump(rank_events, f, indent=2, ensure_ascii=False, allow_nan=False)
            print(f"[INFO] rank_{rank}/{slice_name}: {len(rank_events)} events -> {rank_output}")

        merged_output = output_dir / f"{slice_name}_{args.output}"
        with open(merged_output, "w", encoding="utf-8") as f:
            json.dump(all_events, f, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"[INFO] merged {slice_name}: {len(all_events)} events -> {merged_output}")


if __name__ == "__main__":
    main()
