"""Generate a Chrome/MindStudio trace for one kernel, never across launches."""

import argparse
import csv
import json
from pathlib import Path

from gdn_timer_layout import KERNEL_SLICES
from parse_timer_csv import collect_event_names, validate_frequency


def read_csv_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def generate_trace_json(csv_path, kernel, output_path, cycles_per_us, shared_clock=False):
    validate_frequency(cycles_per_us)
    if cycles_per_us is None or not shared_clock:
        raise ValueError("trace requires verified cycles_per_us and shared_clock=True")
    if kernel not in KERNEL_SLICES:
        raise ValueError(f"unknown kernel {kernel!r}")
    fieldnames, rows = read_csv_rows(csv_path)
    events = collect_event_names(fieldnames)
    valid_starts = [
        int(row[f"{name}_start_cycle"])
        for row in rows
        for name in events
        if int(row[f"{name}_start_cycle"]) > 0
    ]
    if not valid_starts:
        raise ValueError(f"{kernel}: no valid events for trace")
    base_cycle = min(valid_starts)
    trace_events = []
    for row in rows:
        group_id = int(row["group_id"])
        core_type = row["core_type"]
        sub_id = int(row["sub_id"])
        for name in events:
            start = int(row[f"{name}_start_cycle"])
            end = int(row[f"{name}_end_cycle"])
            if start == 0 and end == 0:
                continue
            if start < base_cycle or end < start:
                raise ValueError(f"{kernel}: invalid trace interval {name}")
            event = {
                "ph": "X",
                "cat": "aic_op" if core_type == "AIC" else "aiv_op",
                "pid": f"gdn_{kernel}",
                "tid": f"group_{group_id}_{core_type}_{sub_id}",
                "name": name,
                "ts": round((start - base_cycle) / cycles_per_us, 6),
                "dur": round((end - start) / cycles_per_us, 6),
                "args": {
                    "kernel": kernel,
                    "group_id": group_id,
                    "core_type": core_type,
                    "sub_id": sub_id,
                    "start_cycle": start,
                    "end_cycle": end,
                },
            }
            trace_events.append(event)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(trace_events, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"[INFO] {kernel}: {len(trace_events)} events -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--kernel", choices=tuple(KERNEL_SLICES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycles-per-us", type=float, required=True)
    parser.add_argument("--shared-clock", action="store_true")
    args = parser.parse_args()
    generate_trace_json(args.csv, args.kernel, args.out, args.cycles_per_us, args.shared_clock)


if __name__ == "__main__":
    main()
