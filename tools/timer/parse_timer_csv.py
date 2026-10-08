"""Parse chunk_gated_delta_rule_fwd AscendTimerV2 timer CSV data.

与 fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/timer/
AscendTimerV2.hpp 的布局保持一致：

  [0, N_TIMING_COUNTER)          : 每核计时 start/end 对（每项 2 个 int64）
  [N_TIMING_COUNTER, TOTAL_SIZE) : 每核动态项实际迭代次数

每个计时事件占 2 个 int64: [start_cycle, end_cycle]。
MIX_AIC_1_2 组内布局：AIC=group*3，AIV=group*3+1+subblock，每组占
3 * N_TIMING_COUNTER_PER_CORE_ALIGN 个计数。

用法:
  python parse_timer_csv.py --dir timer            # 解析 timer/rank_0_npu_time.csv
  python parse_timer_csv.py --file timer/rank_0_npu_time.csv --out parsed.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# 与 AscendTimerV2.hpp 的 FixedTiming 一致
FIXED_TIMING_NAMES = {
    0: "KERNEL_TIMING",
    1: "AIC_KKT_CUBE",
    2: "AIV_CUMSUM",
    3: "SYNC_SCORE_CUMSUM",
    4: "AIV_KKT_EPILOGUE",
    5: "SOLVE_TRI",
    6: "SOLVE_SYNC_WAIT",
    7: "RECOMPUTE_WU",
    8: "WRITE_GCUMSUM",
    9: "FWD_H",
    10: "HO_SYNC",
    11: "FWD_O",
    12: "FINAL_SYNC",
}

# 与 AscendTimerV2.hpp 的 DynamicTimingType 一致
DYNAMIC_TIMING_NAMES = {
    0: "FWH_AIC_C1",
    1: "FWH_AIC_C1_WAIT",
    2: "FWH_AIC_C2",
    3: "FWH_AIC_C2_WAIT",
    4: "FWH_AIV_VEC1",
    5: "FWH_AIV_VEC2",
    6: "FWO_AIC_QK",
    7: "FWO_AIC_QH",
    8: "FWO_AIC_ATTENV",
    9: "FWO_AIV_QKMASK",
    10: "FWO_AIV_OUTPUT",
}

FIXED_TIMING_COUNT = len(FIXED_TIMING_NAMES)
DYNAMIC_TYPE_COUNT_ENUM = len(DYNAMIC_TIMING_NAMES)
MAX_DYNAMIC_ITER = 128
N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER
N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2
N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) // 16 * 16
N_CORE_COUNT = 96  # 物理组数上限（每组 1 AIC + 2 AIV）
N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT

DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM
DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) // 16 * 16
DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT

TOTAL_BUFFER_SIZE = N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE

CORES_PER_GROUP = 3  # MIX_AIC_1_2：每组 1 AIC + 2 AIV
# A5/950 系统周期约 1 GHz，1 cycle ≈ 1 ns；此处按 1000 cycle = 1 us 换算。
CYCLE_TO_TIME_BASE = 1000.0


def cycle_to_us(cycles):
    return cycles / CYCLE_TO_TIME_BASE


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


def split_data(raw_data):
    if len(raw_data) != TOTAL_BUFFER_SIZE:
        print(f"[ERROR] 数据量 {len(raw_data)} != 预期 {TOTAL_BUFFER_SIZE}")
        print("[ERROR] 请确认 timer 张量长度与 AscendTimerV2.hpp 的 TOTAL_BUFFER_SIZE 一致"
              "（算子与 torch_custom 需同步重编译安装）。")
        sys.exit(1)
    base_cycle_data = raw_data[:N_TIMING_COUNTER]
    dynamic_iter_data = raw_data[N_TIMING_COUNTER:N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE]
    return base_cycle_data, dynamic_iter_data


def compute_dynamic_max_iter(dynamic_iter_data):
    # 动态迭代区按物理核槽位（core_id = group*3 + slot）索引，共 N_CORE_COUNT 个槽。
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
        actual_max = min(actual_max, MAX_DYNAMIC_ITER)   # 防止 iter 区脏数据导致槽位别名
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
    # 核内区间起点：core_id = group*3 + slot（AIC=group*3，AIV=group*3+1+subblock），
    # 与设备侧 calculate_index 的 group 展开公式等价于 core_id * 每核对齐槽位数。
    return core_id * N_TIMING_COUNTER_PER_CORE_ALIGN


def get_event_index(base_index, logical_idx):
    return base_index + logical_idx * 2


def timing_logical_idx(name_idx, name):
    if name_idx < FIXED_TIMING_COUNT:
        return name_idx
    type_id, iter_idx = parse_dynamic_name(name)
    if type_id is None:
        return -1
    return get_dynamic_timing_idx(type_id, iter_idx)


def parse_and_dump(raw_csv_path, output_csv_path):
    raw_data = load_raw_data(raw_csv_path)
    base_cycle_data, dynamic_iter_data = split_data(raw_data)
    dynamic_max_iters = compute_dynamic_max_iter(dynamic_iter_data)

    print("[INFO] timer layout: start_end")
    print("[INFO] 动态项全局最大迭代次数:")
    for t in range(DYNAMIC_TYPE_COUNT):
        print(f"  {DYNAMIC_TIMING_NAMES.get(t, f'DYN_{t}')}: {dynamic_max_iters[t]}")
        if dynamic_max_iters[t] > MAX_DYNAMIC_ITER:
            print(f"    [WARNING] {DYNAMIC_TIMING_NAMES.get(t, f'DYN_{t}')} 实际迭代次数 {dynamic_max_iters[t]} "
                  f"超过上限 {MAX_DYNAMIC_ITER}，超出部分迭代的计时已被截断/未记录")

    timing_names = get_timing_names(dynamic_max_iters)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["core_id", "group_id", "core_type", "sub_id"]
        for name in timing_names:
            header += [f"{name}_start_cycle", f"{name}_end_cycle", f"{name}_duration(us)"]
        header += [f"{DYNAMIC_TIMING_NAMES.get(t, f'DYN_{t}')}_IterCount" for t in range(DYNAMIC_TYPE_COUNT)]
        writer.writerow(header)

        for core_id in range(N_CORE_COUNT):
            group_id, core_type, sub_id = get_core_info(core_id)
            base_index = get_base_index(core_id)
            row = [core_id, group_id, core_type, sub_id]

            for name_idx, name in enumerate(timing_names):
                logical_idx = timing_logical_idx(name_idx, name)
                event_idx = get_event_index(base_index, logical_idx) if logical_idx >= 0 else -1

                if 0 <= event_idx + 1 < len(base_cycle_data):
                    start_cycle = base_cycle_data[event_idx]
                    end_cycle = base_cycle_data[event_idx + 1]
                    duration_cycles = end_cycle - start_cycle if end_cycle > start_cycle else 0
                    row += [str(start_cycle), str(end_cycle), f"{cycle_to_us(duration_cycles):.2f}"]
                else:
                    row += ["0", "0", "0.00"]

            for t in range(DYNAMIC_TYPE_COUNT):
                idx = core_id * DYNAMIC_ITER_PER_CORE_ALIGN + t
                row.append(str(dynamic_iter_data[idx]) if idx < len(dynamic_iter_data) else "0")

            writer.writerow(row)

    print(f"[INFO] 解析完成, 结果写入: {output_csv_path}")


def print_summary(output_csv_path):
    """按阶段聚合输出：每阶段取所有核的最大时长（关键路径视角）。"""
    with open(output_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    fixed_cols = {}
    for idx, name in enumerate(header):
        if name.endswith("_duration(us)") and name[: -len("_duration(us)")] in FIXED_TIMING_NAMES.values():
            fixed_cols[name[: -len("_duration(us)")]] = idx

    if "KERNEL_TIMING" not in fixed_cols:
        print("[WARNING] 未找到 KERNEL_TIMING_duration(us)")
        return

    print("\n[INFO] 阶段耗时汇总（取各核最大值，单位 us）:")
    kernel_idx = fixed_cols["KERNEL_TIMING"]
    kernel_max = 0.0
    for row in rows:
        try:
            kernel_max = max(kernel_max, float(row[kernel_idx]))
        except (ValueError, IndexError):
            continue
    print(f"  {'KERNEL_TIMING':<20} {kernel_max:>10.2f}  (最慢核)")
    for name, idx in fixed_cols.items():
        if name == "KERNEL_TIMING":
            continue
        stage_max = 0.0
        for row in rows:
            try:
                stage_max = max(stage_max, float(row[idx]))
            except (ValueError, IndexError):
                continue
        if stage_max > 0:
            pct = 100.0 * stage_max / kernel_max if kernel_max > 0 else 0.0
            print(f"  {name:<20} {stage_max:>10.2f}  ({pct:.1f}% of kernel)")

    print("\n[INFO] 有计时数据的核心 (KERNEL_TIMING > 0):")
    print(f"{'core_id':>8} {'group':>6} {'type':>5} {'sub':>4} {'KERNEL_TIMING(us)':>18}")
    print("-" * 50)
    count = 0
    for row in rows:
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
    parser = argparse.ArgumentParser(description="解析 chunk_gated_delta_rule_fwd 的 AscendTimerV2 timer CSV")
    parser.add_argument("--dir", "-d", default=str(Path.cwd() / "timer"), help="timer CSV 文件所在目录")
    parser.add_argument("--file", "-f", default=None, help="直接指定原始 CSV 文件路径")
    parser.add_argument("--out", "-o", default=None, help="解析结果输出路径（默认同目录 _parsed.csv）")
    args = parser.parse_args()

    if args.file is not None:
        input_path = Path(args.file)
        output_path = Path(args.out) if args.out is not None else input_path.with_name(
            input_path.stem + "_parsed.csv")
        parse_and_dump(str(input_path), str(output_path))
        print_summary(str(output_path))
    else:
        timer_dir = Path(args.dir)
        input_path = timer_dir / "rank_0_npu_time.csv"
        output_path = Path(args.out) if args.out is not None else timer_dir / "rank_0_npu_time_parsed.csv"
        if not input_path.exists():
            print(f"[ERROR] 文件不存在 {input_path}")
            sys.exit(1)
        parse_and_dump(str(input_path), str(output_path))
        print_summary(str(output_path))
