#!/usr/bin/env python3
"""解析 AscendTimerV2 计时缓冲（GDN Prepare / FwdH / FwdO 三算子路径）。

与 ``fla/ops/ascendc/common/kernel_utils/gdn_timer/AscendTimerV2.hpp`` 严格对齐：
枚举名、``MAX_DYNAMIC_ITER``、``N_CORE_COUNT``、切片布局全部从该头文件解析，
Python 侧不手抄常量。

缓冲布局（每个 launch 一份，int64 计）：

  timer[SliceOffsetWords(slice) + 0                       : + N_TIMING_COUNTER)
        每核固定项 + 每核动态项的 [start_cycle, end_cycle] 对
  timer[SliceOffsetWords(slice) + N_TIMING_COUNTER
        : + SLICE_WORDS)
        每核动态项的实际迭代次数

用法::

    python parse_timer_csv.py --dir timer --rank-start 0 --rank-end 0
    python parse_timer_csv.py --dir timer --slice fwd_h --slice prepare
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

TIMER_HEADER = Path(__file__).resolve().parents[8] / (
    "fla/ops/ascendc/common/kernel_utils/gdn_timer/AscendTimerV2.hpp"
)

_ENUM_BLOCK_RE = r"enum\s+{kind}\s*:\s*int32_t\s*\{{(.*?)\}};"
_IDENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*[^,]*)?,?\s*(?://.*)?$")
_CONST_RE = r"static\s+constexpr\s+int\s+{name}\s*=\s*(\d+)\s*;"


def _parse_enum(text, kind):
    match = re.search(_ENUM_BLOCK_RE.format(kind=kind), text, re.DOTALL)
    if match is None:
        raise RuntimeError(f"AscendTimerV2.hpp 中未找到 enum {kind}")
    names = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if "=" in line.split("//")[0]:
            # 显式赋值的条目在本头里只出现在固定项（KERNEL_TIMING_IDX = 0）。
            line = line.split("//")[0].strip()
        item = _IDENT_RE.match(line)
        if item is None:
            continue
        name = item.group(1)
        if name.endswith("_COUNT") or name.endswith("_COUNT_ENUM"):
            continue
        names.append(name)
    if not names:
        raise RuntimeError(f"enum {kind} 解析结果为空")
    return names


def _parse_const(text, name):
    match = re.search(_CONST_RE.format(name=name), text)
    if match is None:
        raise RuntimeError(f"AscendTimerV2.hpp 中未找到常量 {name}")
    return int(match.group(1))


def _load_header_constants():
    text = TIMER_HEADER.read_text(encoding="utf-8")
    fixed_names = _parse_enum(text, "FixedTiming")
    dynamic_names = _parse_enum(text, "DynamicTimingType")
    slice_names = _parse_enum(text, "KernelSlice")
    max_dynamic_iter = _parse_const(text, "MAX_DYNAMIC_ITER")
    core_count = _parse_const(text, "N_CORE_COUNT")
    return fixed_names, dynamic_names, slice_names, max_dynamic_iter, core_count


(
    FIXED_TIMING_LIST,
    DYNAMIC_TIMING_LIST,
    SLICE_NAME_LIST,
    MAX_DYNAMIC_ITER,
    N_CORE_COUNT,
) = _load_header_constants()

FIXED_TIMING_NAMES = {idx: name for idx, name in enumerate(FIXED_TIMING_LIST)}
DYNAMIC_TIMING_NAMES = {idx: name for idx, name in enumerate(DYNAMIC_TIMING_LIST)}
FIXED_TIMING_COUNT = len(FIXED_TIMING_LIST)
DYNAMIC_TYPE_COUNT_ENUM = len(DYNAMIC_TIMING_LIST)

# 由头文件中的原始常量推导（与 C++ 侧表达式一一对应）。
N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER
N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2
N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) // 16 * 16
N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT

DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM
DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) // 16 * 16
DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT

SLICE_WORDS = N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE
KERNEL_SLICE_COUNT = len(SLICE_NAME_LIST)
TOTAL_BUFFER_WORDS = SLICE_WORDS * KERNEL_SLICE_COUNT

# 单切片内部布局（解析时把选中切片搬到这段地址空间里比较）。
TOTAL_BUFFER_SIZE = SLICE_WORDS

KERNEL_TIMING_NAME = FIXED_TIMING_NAMES[0]
CORES_PER_GROUP = 3
# 按实际方案填 Accumulate 阶段/动态类型前缀，合成 end 不进入 trace。
ACCUMULATED_EVENTS = set()

SLICE_ALIASES = {
    "phase6": 0,
    "prepare": 1,
    "fwd_h": 2,
    "fwd_o": 3,
}


def resolve_slice(name):
    """把切片名（枚举名或简称）映射到 KernelSlice 序号。"""
    key = str(name).strip().lower()
    if key in SLICE_ALIASES:
        return SLICE_ALIASES[key]
    upper = str(name).strip().upper()
    upper = upper if upper.startswith("SLICE_") else f"SLICE_{upper}"
    if upper in SLICE_NAME_LIST:
        return SLICE_NAME_LIST.index(upper)
    raise SystemExit(f"未知切片 {name!r}；可选：{', '.join(SLICE_ALIASES)}")


def slice_offset_words(slice_index):
    return slice_index * SLICE_WORDS


def validate_frequency(cycles_per_us):
    if cycles_per_us is not None and (not math.isfinite(cycles_per_us) or cycles_per_us <= 0):
        raise ValueError("cycles_per_us must be a verified finite positive number")


def cycle_to_us(cycles, cycles_per_us):
    validate_frequency(cycles_per_us)
    if cycles_per_us is None:
        raise ValueError("clock frequency is not confirmed")
    return cycles / cycles_per_us


def is_accumulated(name):
    kind, _ = parse_dynamic_name(name)
    return name in ACCUMULATED_EVENTS or (kind is not None and DYNAMIC_TIMING_NAMES[kind] in ACCUMULATED_EVENTS)


def load_raw_data(csv_path):
    data = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if item:
                    data.append(int(item))
    return data


def split_data(raw_data, slice_index):
    """取指定切片的 SLICE_WORDS 个 int64，再切成计时区与迭代次数区。"""
    if len(raw_data) == TOTAL_BUFFER_WORDS:
        base = slice_offset_words(slice_index)
        words = raw_data[base:base + SLICE_WORDS]
    elif len(raw_data) == SLICE_WORDS:
        # 允许直接喂单切片 raw（例如只 dump 了该 launch 的切片）。
        words = list(raw_data)
    else:
        print(f"[ERROR] 数据量 {len(raw_data)} != {TOTAL_BUFFER_WORDS}（全切片）"
              f" 或 {SLICE_WORDS}（单切片）")
        print("[ERROR] 请确认 timer 张量长度来自 AscendTimer::TOTAL_BUFFER_WORDS，"
              "并且 kernel/host 已按当前头文件重新编译安装。")
        sys.exit(1)
    base_cycle_data = words[:N_TIMING_COUNTER]
    dynamic_iter_data = words[N_TIMING_COUNTER:N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE]
    return base_cycle_data, dynamic_iter_data


def compute_dynamic_max_iter(dynamic_iter_data):
    max_iters = [0] * DYNAMIC_TYPE_COUNT
    for core_id in range(N_CORE_COUNT):
        core_offset = core_id * DYNAMIC_ITER_PER_CORE_ALIGN
        for t in range(DYNAMIC_TYPE_COUNT):
            idx = core_offset + t
            if idx < len(dynamic_iter_data) and dynamic_iter_data[idx] > max_iters[t]:
                max_iters[t] = dynamic_iter_data[idx]
    return max_iters


def get_timing_names(dynamic_max_iters):
    names = [FIXED_TIMING_NAMES.get(i, f"UNKNOWN_FIXED_{i}") for i in range(FIXED_TIMING_COUNT)]
    for t in range(DYNAMIC_TYPE_COUNT):
        prefix = DYNAMIC_TIMING_NAMES.get(t, f"UNKNOWN_DYN_{t}")
        actual_max = dynamic_max_iters[t] if t < len(dynamic_max_iters) and dynamic_max_iters[t] > 0 else MAX_DYNAMIC_ITER
        actual_max = min(actual_max, MAX_DYNAMIC_ITER)  # 防止 iter 区脏数据导致槽位别名
        for iter_idx in range(actual_max):
            names.append(f"{prefix}_{iter_idx}")
    return names


def parse_dynamic_name(name):
    last_underscore = name.rfind("_")
    if last_underscore < 0:
        return None, None
    prefix = name[:last_underscore]
    iter_str = name[last_underscore + 1:]
    if not iter_str.isdigit():
        return None, None
    for type_id, type_name in DYNAMIC_TIMING_NAMES.items():
        if type_name == prefix:
            return type_id, int(iter_str)
    return None, None


def get_dynamic_timing_idx(type_id, iter_idx):
    return FIXED_TIMING_COUNT + type_id * MAX_DYNAMIC_ITER + iter_idx


def get_core_info(core_id):
    group_id = core_id // CORES_PER_GROUP
    sub_group_id = core_id % CORES_PER_GROUP
    if sub_group_id == 0:
        return group_id, "AIC", 0
    return group_id, "AIV", sub_group_id - 1


def get_base_index(core_id):
    group_id, core_type, sub_id = get_core_info(core_id)
    base = group_id * N_TIMING_COUNTER_PER_CORE_ALIGN * CORES_PER_GROUP
    if core_type == "AIV":
        base += N_TIMING_COUNTER_PER_CORE_ALIGN * (1 + sub_id)
    return base


def get_event_index(base_index, logical_idx):
    return base_index + logical_idx * 2


def timing_logical_idx(name_idx, name):
    if name_idx < FIXED_TIMING_COUNT:
        return name_idx
    type_id, iter_idx = parse_dynamic_name(name)
    if type_id is None:
        return -1
    return get_dynamic_timing_idx(type_id, iter_idx)


def parse_and_dump(raw_csv_path, output_csv_path, cycles_per_us=None, slice_index=1):
    validate_frequency(cycles_per_us)
    raw_data = load_raw_data(raw_csv_path)
    base_cycle_data, dynamic_iter_data = split_data(raw_data, slice_index)
    if not any(base_cycle_data):
        raise ValueError(
            f"{SLICE_NAME_LIST[slice_index]} 无计时记录；检查该 launch 是否走了这条路径、"
            "kernel 是否为最新版本、timer 是否确实传入"
        )
    dynamic_max_iters = compute_dynamic_max_iter(dynamic_iter_data)

    print(f"[INFO] timer slice: {SLICE_NAME_LIST[slice_index]}"
          f"（words {slice_offset_words(slice_index)}..{slice_offset_words(slice_index) + SLICE_WORDS}）")
    print("[INFO] timer layout: start_end")
    print("[INFO] 动态项全局最大迭代次数:")
    for t in range(DYNAMIC_TYPE_COUNT):
        print(f"  {DYNAMIC_TIMING_NAMES.get(t, f'DYN_{t}')}: {dynamic_max_iters[t]}")
        if dynamic_max_iters[t] > MAX_DYNAMIC_ITER:
            print(f"    [WARNING] {DYNAMIC_TIMING_NAMES.get(t, f'DYN_{t}')} 实际迭代次数 "
                  f"{dynamic_max_iters[t]} 超过上限 {MAX_DYNAMIC_ITER}，超出部分未记录")

    timing_names = get_timing_names(dynamic_max_iters)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["core_id", "group_id", "core_type", "sub_id"]
        for name in timing_names:
            header += [f"{name}_start_cycle", f"{name}_end_cycle", f"{name}_duration_cycles"]
            if cycles_per_us is not None:
                header.append(f"{name}_duration(us)")
        header += [f"{DYNAMIC_TIMING_NAMES.get(t, f'DYN_{t}')}_IterCount" for t in range(DYNAMIC_TYPE_COUNT)]
        writer.writerow(header)

        for core_id in range(N_CORE_COUNT):
            group_id, core_type, sub_id = get_core_info(core_id)
            base_index = get_base_index(core_id)
            kernel_start, kernel_end = base_cycle_data[base_index:base_index + 2]
            count_start = core_id * DYNAMIC_ITER_PER_CORE_ALIGN
            active = any(base_cycle_data[base_index:base_index + N_TIMING_COUNTER_PER_CORE_ALIGN]) or any(
                dynamic_iter_data[count_start:count_start + DYNAMIC_ITER_PER_CORE_ALIGN])
            if active and not 0 < kernel_start <= kernel_end:
                raise ValueError(f"incomplete {KERNEL_TIMING_NAME} at core {core_id}")
            row = [core_id, group_id, core_type, sub_id]

            for name_idx, name in enumerate(timing_names):
                logical_idx = timing_logical_idx(name_idx, name)
                event_idx = get_event_index(base_index, logical_idx) if logical_idx >= 0 else -1

                if 0 <= event_idx + 1 < len(base_cycle_data):
                    start_cycle = base_cycle_data[event_idx]
                    end_cycle = base_cycle_data[event_idx + 1]
                    if (start_cycle, end_cycle) != (0, 0) and not 0 < start_cycle <= end_cycle:
                        raise ValueError(f"invalid interval at core {core_id}: {name}")
                    if start_cycle > 0 and (not kernel_start <= start_cycle <= kernel_end or
                                            (not is_accumulated(name) and end_cycle > kernel_end)):
                        raise ValueError(f"event outside {KERNEL_TIMING_NAME} at core {core_id}: {name}")
                    duration_cycles = end_cycle - start_cycle
                    row += [str(start_cycle), str(end_cycle), str(duration_cycles)]
                    if cycles_per_us is not None:
                        row.append(f"{cycle_to_us(duration_cycles, cycles_per_us):.2f}")
                else:
                    row += ["0", "0", "0"]
                    if cycles_per_us is not None:
                        row.append("0.00")

            for t in range(DYNAMIC_TYPE_COUNT):
                idx = core_id * DYNAMIC_ITER_PER_CORE_ALIGN + t
                row.append(str(dynamic_iter_data[idx]) if idx < len(dynamic_iter_data) else "0")

            writer.writerow(row)

    print(f"[INFO] 解析完成, 结果写入: {output_csv_path}")


def print_summary(output_csv_path):
    with open(output_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        kernel_col = f"{KERNEL_TIMING_NAME}_duration(us)"
        if kernel_col not in header:
            kernel_col = f"{KERNEL_TIMING_NAME}_duration_cycles"
        unit = "us" if kernel_col.endswith("(us)") else "cycles"
        if kernel_col not in header:
            print(f"[WARNING] 未找到 {KERNEL_TIMING_NAME} duration")
            return
        kernel_idx = header.index(kernel_col)

        print(f"\n[INFO] 有计时数据的核心 ({KERNEL_TIMING_NAME} > 0):")
        print(f"{'core_id':>8} {'group':>6} {'type':>5} {'sub':>4} {(KERNEL_TIMING_NAME + '(' + unit + ')'):>22}")
        print("-" * 50)
        count = 0
        for row in reader:
            try:
                val = float(row[kernel_idx])
            except (ValueError, IndexError):
                continue
            if val > 0:
                print(f"{row[0]:>8} {row[1]:>6} {row[2]:>5} {row[3]:>4} {val:>18.2f}")
                count += 1
        if count == 0:
            print("  (无有效计时数据)")
        print(f"\n[INFO] 共 {count} 个核心有计时数据")


if __name__ == "__main__":
    default_timer_dir = str(Path.cwd() / "timer")

    parser = argparse.ArgumentParser(description="解析 AscendTimerV2 格式的 GDN timer CSV 数据")
    parser.add_argument("--dir", "-d", default=default_timer_dir, help="timer CSV 文件所在目录")
    parser.add_argument("--rank-start", type=int, default=0, help="起始 rank (包含)")
    parser.add_argument("--rank-end", type=int, default=7, help="结束 rank (包含)")
    parser.add_argument("--cycles-per-us", type=float, help="已核实的时钟频率；默认只导出 cycles")
    parser.add_argument("--slice", action="append", default=None,
                        choices=sorted(SLICE_ALIASES),
                        help="要解析的切片，可重复（默认 prepare/fwd_h/fwd_o）")
    args = parser.parse_args()

    slices = args.slice if args.slice else ["prepare", "fwd_h", "fwd_o"]
    timer_dir = Path(args.dir)
    for rank in range(args.rank_start, args.rank_end + 1):
        input_path = timer_dir / f"rank_{rank}_npu_time.csv"
        if not input_path.exists():
            print(f"[SKIP] rank_{rank}: 文件不存在 {input_path}")
            continue
        for slice_name in slices:
            output_path = timer_dir / f"rank_{rank}_{slice_name}_parsed.csv"
            print(f"\n{'=' * 60}")
            print(f"[INFO] 处理 rank_{rank} / {slice_name}")
            print(f"{'=' * 60}")
            try:
                parse_and_dump(str(input_path), str(output_path), args.cycles_per_us,
                               resolve_slice(slice_name))
            except ValueError as exc:
                print(f"[WARNING] {exc}")
                continue
            print_summary(str(output_path))
