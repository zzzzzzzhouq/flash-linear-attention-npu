# chunk_gated_delta_rule_fwd 计时采集与解析（A5 三算子路径）

本目录只放**采集/解析脚本**，不做数值校验；算子测试见 `tests/atk/`。
打点设计见 `../../docs/timer_flow.md`，计时框架在
`fla/ops/ascendc/common/kernel_utils/gdn_timer/`。

| 文件 | 作用 |
| --- | --- |
| `profile_gdn.py` | 申请 timer 张量、跑一次 public op、导出 raw CSV + 逐核 parsed CSV +（可选）trace |
| `parse_timer_csv.py` | 解析 raw int64 缓冲；枚举与常量从 `AscendTimerV2.hpp` 解析，支持按切片解析 |
| `trace_parser.py` | parsed CSV → Chrome trace（Perfetto）JSON |

## 1. 计时缓冲与切片

`AscendTimer::TOTAL_BUFFER_WORDS` 个 int64，被 `KERNEL_SLICE_COUNT` 个 launch 切成
互不覆盖的切片：

| 切片 | 写入者 | CLI 名 |
| --- | --- | --- |
| `SLICE_PHASE6` | 融合 Phase6 路径（本轮不打点，占位） | `phase6` |
| `SLICE_PREPARE` | `ChunkGatedDeltaRuleFwdPrepare` | `prepare` |
| `SLICE_FWD_H` | `ChunkFwdH` | `fwd_h` |
| `SLICE_FWD_O` | `ChunkFwdO` | `fwd_o` |

每个切片内部：`[0, N_TIMING_COUNTER)` 是逐核 `[start_cycle, end_cycle]` 对，
`[N_TIMING_COUNTER, SLICE_WORDS)` 是逐核各动态项的实际迭代次数。
固定项是 `KERNEL_TIMING_IDX` / `MAIN_INIT` / `MAIN_BODY`；动态项见
`AscendTimer::DynamicTimingType`（Prepare 用 `S*` / FwdH 用 `H_*` / FwdO 用 `O_*`）。

## 2. 采集

timer 通过 public op 的可选输入传入，走 **ctypes 直调** 路线：

```bash
python profile_gdn.py --device 0 --out profiling_results/gdn_fwd_timer
```

之所以不用 stable ABI 入口：`torch_custom/fla_npu/csrc/src/stable_chunk_gated_delta_rule_fwd.cpp`
当前把 timer 固定传 `optional_tensor(std::nullopt)`（stable 侧的 schema 保持不变），
因此脚本调用
`fla_npu.ops.ascendc._aclnn_ctypes.npu_chunk_gated_delta_rule_fwd(..., timer=timer)`，
它内部对应 `aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize` 的 `timerOptional` 参数。

产物（默认 `profiling_results/gdn_fwd_timer/`）：

- `rank_<r>_npu_time.csv`：raw int64（一行，`parse_timer_csv.py` 的输入）
- `rank_<r>_prepare_parsed.csv` / `_fwd_h_parsed.csv` / `_fwd_o_parsed.csv`
- `rank_<r>_<slice>.json`：仅在同时给 `--cycles-per-us` 与 `--shared-clock` 时生成
- `metadata.json`：shape / dtype 开关 / timer 字数 / 频率 / 输出有限性

```bash
# 已核实时钟（AIC/AIV 共同时钟）才给 us 与 trace
python profile_gdn.py --device 0 --cycles-per-us 1800 --shared-clock

# 基线：不传 timer，验证"timer 为空时零行为差异"
python profile_gdn.py --device 0 --no-timer
```

默认输入满足 A5 Prepare 路径的 shape 契约：BF16、K=V=128、chunk=64、Hk=2/Hv=4（GVA=2）、
T=256（4 个 chunk）。`q/k/v` 按 `layout=BNSD` 给 `[B,H,T,D]`，`g/beta` 是 `[B,T,Hv]`
（与 `npu_chunk_gated_delta_rule_fwd` 的 python 契约一致）。

## 3. 离线重新解析

```bash
# 只解析指定切片（可重复）
python parse_timer_csv.py --dir profiling_results/gdn_fwd_timer --rank-start 0 --rank-end 0 --slice fwd_h

# 生成 trace（需要已核实频率 + 共同时钟）
python trace_parser.py --timer-dir profiling_results/gdn_fwd_timer \
    --cycles-per-us 1800 --shared-clock --rank-start 0 --rank-end 0
```

## 4. 读数约定

- 所有 `S*` / `H_*` / `O_*` 都是**下发/issue 窗口**，不是完成时延；
  异步 MMAD 的实际完成要由后续功能性等待或产物消费点体现。
- 窗口之间可以重叠，**不能相加**；总时长只看 `KERNEL_TIMING_IDX`。
- 每核每动态项最多记录 `MAX_DYNAMIC_ITER`（当前 1088）次，超出部分截断，
  实际次数写在 `<TYPE>_IterCount` 列；1088 槽覆盖官方 11264 tokens case 的 1056 次 FwdH 预取。
- 默认只导出 cycles；`cycles_per_us` 必须实测核实，跨核绝对时间要先核实
  AIC/AIV 共同时钟。
