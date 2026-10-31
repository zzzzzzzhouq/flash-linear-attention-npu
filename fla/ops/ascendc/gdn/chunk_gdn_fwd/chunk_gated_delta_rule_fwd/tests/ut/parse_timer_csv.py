"""Parse one GDN kernel's AscendTimerV2 slice into per-core stage CSV."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from gdn_timer_layout import (
    CORES_PER_GROUP,
    DYNAMIC_ITER_PER_CORE_ALIGN,
    DYNAMIC_ITER_TOTAL_SIZE,
    DYNAMIC_TIMING_NAMES,
    DYNAMIC_TYPE_COUNT,
    FIXED_TIMING_COUNT,
    FIXED_TIMING_NAMES,
    KERNEL_SLICES,
    MAX_DYNAMIC_ITER,
    N_CORE_COUNT,
    N_TIMING_COUNTER,
    N_TIMING_COUNTER_PER_CORE,
    N_TIMING_COUNTER_PER_CORE_ALIGN,
    TIMER_BUFFER_NUMEL,
    TIMER_SLICE_NUMEL,
)


def validate_frequency(cycles_per_us):
    if cycles_per_us is not None and (not math.isfinite(cycles_per_us) or cycles_per_us <= 0):
        raise ValueError("cycles_per_us must be a verified finite positive number")


def load_raw_data(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.int64, copy=False).reshape(-1)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError("npz must contain exactly one timer array")
            return archive[archive.files[0]].astype(np.int64, copy=False).reshape(-1)
    values = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            values.extend(int(item.strip()) for item in row if item.strip())
    return np.asarray(values, dtype=np.int64)


def select_kernel_slice(raw, kernel):
    if kernel not in KERNEL_SLICES:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {tuple(KERNEL_SLICES)}")
    if raw.size == TIMER_BUFFER_NUMEL:
        begin = KERNEL_SLICES[kernel] * TIMER_SLICE_NUMEL
        return raw[begin:begin + TIMER_SLICE_NUMEL]
    if raw.size == TIMER_SLICE_NUMEL:
        return raw
    raise ValueError(
        f"timer has {raw.size} elements; expected one slice ({TIMER_SLICE_NUMEL}) "
        f"or three slices ({TIMER_BUFFER_NUMEL})"
    )


def get_core_info(core_id):
    group_id, sub_group_id = divmod(core_id, CORES_PER_GROUP)
    if sub_group_id == 0:
        return group_id, "AIC", 0
    return group_id, "AIV", sub_group_id - 1


def get_base_index(core_id):
    return core_id * N_TIMING_COUNTER_PER_CORE_ALIGN


def parse_dynamic_name(name):
    prefix, separator, iter_text = name.rpartition("_")
    if not separator or not iter_text.isdigit() or prefix not in DYNAMIC_TIMING_NAMES:
        raise ValueError(f"invalid dynamic event name {name!r}")
    return DYNAMIC_TIMING_NAMES.index(prefix), int(iter_text)


def collect_event_names(fieldnames):
    return [
        column[:-len("_start_cycle")]
        for column in fieldnames
        if column.endswith("_start_cycle") and column != "KERNEL_TIMING_start_cycle"
    ]


def _dynamic_max_iters(dynamic_iter_data):
    result = [0] * DYNAMIC_TYPE_COUNT
    for core_id in range(N_CORE_COUNT):
        offset = core_id * DYNAMIC_ITER_PER_CORE_ALIGN
        for type_id in range(DYNAMIC_TYPE_COUNT):
            count = int(dynamic_iter_data[offset + type_id])
            if count < 0 or count > MAX_DYNAMIC_ITER:
                raise ValueError(f"invalid iter count at core {core_id}, type {type_id}: {count}")
            result[type_id] = max(result[type_id], count)
    return result


def parse_and_dump(raw_path, output_csv_path, kernel="prepare", cycles_per_us=None):
    validate_frequency(cycles_per_us)
    raw = select_kernel_slice(load_raw_data(raw_path), kernel)
    base = raw[:N_TIMING_COUNTER]
    iters = raw[N_TIMING_COUNTER:N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE]
    if not np.any(base):
        raise ValueError(f"{kernel}: no timer records; check the actual route and loaded kernel")

    max_iters = _dynamic_max_iters(iters)
    names = list(FIXED_TIMING_NAMES)
    logical_indices = list(range(FIXED_TIMING_COUNT))
    for type_id, (name, count) in enumerate(zip(DYNAMIC_TIMING_NAMES, max_iters)):
        for iter_id in range(count):
            names.append(f"{name}_{iter_id}")
            logical_indices.append(FIXED_TIMING_COUNT + type_id * MAX_DYNAMIC_ITER + iter_id)

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["core_id", "group_id", "core_type", "sub_id"]
        for name in names:
            header.extend((f"{name}_start_cycle", f"{name}_end_cycle", f"{name}_duration_cycles"))
            if cycles_per_us is not None:
                header.append(f"{name}_duration(us)")
        header.extend(f"{name}_IterCount" for name in DYNAMIC_TIMING_NAMES)
        writer.writerow(header)

        for core_id in range(N_CORE_COUNT):
            group_id, core_type, sub_id = get_core_info(core_id)
            base_index = get_base_index(core_id)
            core_base = base[base_index:base_index + N_TIMING_COUNTER_PER_CORE]
            iter_offset = core_id * DYNAMIC_ITER_PER_CORE_ALIGN
            active = bool(np.any(core_base) or np.any(iters[iter_offset:iter_offset + DYNAMIC_TYPE_COUNT]))
            kernel_start, kernel_end = map(int, core_base[:2])
            if active and not 0 < kernel_start <= kernel_end:
                raise ValueError(f"{kernel}: incomplete KERNEL_TIMING at core {core_id}")
            row = [core_id, group_id, core_type, sub_id]
            for name, logical_index in zip(names, logical_indices):
                event_index = base_index + logical_index * 2
                start, end = map(int, base[event_index:event_index + 2])
                if (start == 0) != (end == 0):
                    raise ValueError(f"{kernel}: half-written interval at core {core_id}: {name}")
                if start and not 0 < start <= end:
                    raise ValueError(f"{kernel}: invalid interval at core {core_id}: {name}")
                if start and (start < kernel_start or end > kernel_end):
                    raise ValueError(f"{kernel}: event outside KERNEL_TIMING at core {core_id}: {name}")
                duration = end - start
                row.extend((start, end, duration))
                if cycles_per_us is not None:
                    row.append(f"{duration / cycles_per_us:.6f}")
            row.extend(map(int, iters[iter_offset:iter_offset + DYNAMIC_TYPE_COUNT]))
            writer.writerow(row)
    print(f"[INFO] {kernel}: {output_csv_path}")
    return output_csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="rank_0_gdn_timer.npy, one-slice npy, or raw CSV")
    parser.add_argument("--kernel", choices=tuple(KERNEL_SLICES), required=True)
    parser.add_argument("--out", type=Path, required=True, help="parsed CSV path")
    parser.add_argument("--cycles-per-us", type=float)
    args = parser.parse_args()
    parse_and_dump(args.raw, args.out, args.kernel, args.cycles_per_us)


if __name__ == "__main__":
    main()
