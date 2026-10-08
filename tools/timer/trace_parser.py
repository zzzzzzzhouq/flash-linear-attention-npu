"""Generate Chrome trace JSON (Perfetto 兼容) from parsed timer CSV files.

输入为 parse_timer_csv.py 产出的 *_parsed.csv，输出 chrome trace 事件数组，
可直接用 chrome://tracing 或 https://ui.perfetto.dev 加载查看各核时间线。

用法:
  python trace_parser.py --timer-dir timer --output-dir timer_json
"""

import argparse
import csv
import json
from pathlib import Path

CYCLE_TO_TIME_BASE = 1000.0  # 1000 cycle = 1 us


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
        if name == "KERNEL_TIMING":
            continue
        if not (name.startswith("AIC") or name.startswith("AIV") or name.startswith("FWH")
                or name.startswith("FWO") or name.startswith("SOLVE") or name.startswith("SYNC")
                or name.startswith("RECOMPUTE") or name.startswith("WRITE") or name.startswith("FWD")
                or name.startswith("HO_") or name.startswith("FINAL")):
            continue
        if f"{name}_end_cycle" in fields:
            events.append(name)
    return events


def find_min_start(csv_path):
    """返回本 rank 内所有事件的最小有效 start cycle，用作时间轴零点。"""
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
            if start > 0 and end > start:
                min_start = start if min_start is None else min(min_start, start)
    return min_start if min_start is not None else 0


def generate_trace_json(csv_path, rank, base_cycle):
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
            if start <= 0 or end <= start:
                continue

            ts = (start - base_cycle) / CYCLE_TO_TIME_BASE
            dur = (end - start) / CYCLE_TO_TIME_BASE
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

    return trace_events


def generate_trace_files(parsed_csv_path, output_dir, rank=0):
    """由 *_parsed.csv 生成单 rank trace JSON，返回事件数。"""
    csv_path = Path(parsed_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cycle = find_min_start(csv_path)
    rank_events = generate_trace_json(csv_path, rank, base_cycle)
    rank_output = output_dir / f"rank_{rank}_npu_time.json"
    with open(rank_output, "w", encoding="utf-8") as f:
        json.dump(rank_events, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"[INFO] trace: {len(rank_events)} events -> {rank_output}")
    return len(rank_events)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Chrome trace JSON from parsed chunk_gated_delta_rule_fwd timer CSV")
    parser.add_argument("--timer-dir", default="./timer", help="包含 *_parsed.csv 的目录")
    parser.add_argument("--output-dir", default="./timer_json", help="trace JSON 输出目录")
    parser.add_argument("--rank-start", type=int, default=0, help="起始 rank (包含)")
    parser.add_argument("--rank-end", type=int, default=0, help="结束 rank (包含)")
    parser.add_argument("--output", default="all_rank_trace.json", help="合并 trace JSON 文件名")
    args = parser.parse_args()

    timer_dir = Path(args.timer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_paths = [
        timer_dir / f"rank_{rank}_npu_time_parsed.csv"
        for rank in range(args.rank_start, args.rank_end + 1)
    ]
    all_events = []

    for rank, csv_path in zip(range(args.rank_start, args.rank_end + 1), parsed_paths):
        if not csv_path.exists():
            print(f"[SKIP] rank_{rank}: 文件不存在 {csv_path}")
            continue
        base_cycle = find_min_start(csv_path)
        rank_events = generate_trace_json(csv_path, rank, base_cycle)
        all_events.extend(rank_events)
        rank_output = output_dir / f"rank_{rank}_npu_time.json"
        with open(rank_output, "w", encoding="utf-8") as f:
            json.dump(rank_events, f, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"[INFO] rank_{rank}: {len(rank_events)} events -> {rank_output}")

    merged_output = output_dir / args.output
    with open(merged_output, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"[INFO] merged: {len(all_events)} events -> {merged_output}")


if __name__ == "__main__":
    main()
