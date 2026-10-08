# chunk_gated_delta_rule_fwd timer 打点工具

本目录是 `timer` 分支的分析工具，配套 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd`
的设备侧计时打点（移植自 vllm-ascend `fused_sparse_attention_overlap` 的 AscendTimerV2），
用于分析 Phase6 融合 kernel 内部各阶段耗时。

## 组成

| 文件 | 作用 |
| --- | --- |
| `run_chunk_gated_delta_rule_fwd_timer.py` | 采集脚本：构造输入 → 传 timer 张量调用算子 → 落盘 raw CSV → 解析汇总 → trace JSON |
| `parse_timer_csv.py` | raw CSV（int64 计数流）→ 每核每阶段 `*_parsed.csv`，含 `TOTAL_BUFFER_SIZE` 布局定义（与算子侧 `AscendTimerV2.hpp` 一致） |
| `trace_parser.py` | `*_parsed.csv` → Chrome trace JSON（Perfetto / chrome://tracing 可视化各核时间线） |

算子侧改动（同分支）：

- `fla/.../chunk_gated_delta_rule_fwd/op_kernel/timer/AscendTimerV2.hpp`：计时项枚举 + 缓冲布局常量 + `TIMER_BLOCK` 开关宏；
- `fla/.../op_kernel/timer/AscendTimerV2_device.hpp`：设备侧 `AscendTimerDevice`（`Tik/Tok`、NoBarrier 变体，基于 `GetSystemCycle()`）；
- kernel 入口与 A5 Phase6 各阶段、FwdH/FwdO 内部循环的打点；
- aclnn/ctypes ABI 增加 `timer` 可选输入（本分支允许改 ABI，不合入 main）。

## 计时项

固定项（每核一份）：`KERNEL_TIMING`、`AIC_KKT_CUBE`、`AIV_CUMSUM`、`SYNC_SCORE_CUMSUM`、
`AIV_KKT_EPILOGUE`、`SOLVE_TRI`、`SOLVE_SYNC_WAIT`、`RECOMPUTE_WU`、`WRITE_GCUMSUM`、
`FWD_H`、`HO_SYNC`、`FWD_O`、`FINAL_SYNC`。

动态项（每调度迭代一个 `[start, end]` 槽位）：FwdH 的 `FWH_AIC_C1/C2`（含 `_WAIT` 纯等待）、
`FWH_AIV_VEC1/VEC2`，FwdO 的 `FWO_AIC_QK/QH/ATTENV`、`FWO_AIV_QKMASK/OUTPUT`。
每核上限 128 个迭代槽（`MAX_DYNAMIC_ITER`），超出部分截断并在设备侧打印告警。

## 使用

前提：用 timer 分支构建并安装完整 wheel（算子侧与 `torch_custom` 的 ABI 需同步更新），
在 A5/950 环境运行。

```bash
# 默认 shape（GVA=8 强制走 Phase6 融合 kernel）
python tools/timer/run_chunk_gated_delta_rule_fwd_timer.py

# 指定 shape / 输出目录 / 采集次数
python tools/timer/run_chunk_gated_delta_rule_fwd_timer.py \
    --tokens 11274 --key-heads 16 --value-heads 128 --out-dir ./timer_out --iters 3
```

产物（默认写入仓库根 `timer/`）：

- `rank_0_npu_time.csv`：原始 int64 计数（每次采集一份，多 iter 时带 `_0/_1` 后缀）；
- `rank_0_npu_time_parsed.csv`：每核每阶段 start/end/duration(us)；
- `rank_0_npu_time.json`：Chrome trace JSON，用 <https://ui.perfetto.dev> 打开。

已有数据离线重解析：

```bash
python tools/timer/parse_timer_csv.py --file timer/rank_0_npu_time.csv
python tools/timer/trace_parser.py --timer-dir timer --output-dir timer_json
```

## 注意事项

1. **路径分派**：A5 上 4D BF16 + K=V=128 + chunk=64 + hv/hq≤4 的 shape 走
   `Prepare/FwdH/FwdO` 三算子路径，不进入 Phase6 融合 kernel，timer 无数据。
   需任选其一绕开：GVA>4、`--chunk-size 128`、`--value-dim 256`、`--dtype fp16`。
2. **计时口径**：阶段级统一使用 `TikNoBarrier/TokNoBarrier`（不排空流水，测下发时刻），
   `KERNEL_TIMING` 收尾用带 barrier 的 `Tok` 保证全流水排空。FwdH/FwdO 的 `_WAIT` 项
   单独度量 `CrossCoreWaitFlag` 纯等待，用于定位 AIC/AIV 相互等待。
3. **性能扰动**：打点本身有轻微开销（每阶段 2 次标量 GM 写）；`AscendTimerV2.hpp` 中
   `#define ENABLE_TIMER 1` 置 0 可编译无打点对照版本。
4. **时基**：A5 `GetSystemCycle()` 约 1 cycle = 1 ns，脚本按 1000 cycle = 1 us 换算。
5. **多 rank**：单卡脚本按 rank 0 落盘；多 rank 时把各自 raw CSV 命名为
   `rank_{i}_npu_time.csv` 后用 `trace_parser.py --rank-start 0 --rank-end N` 合并。
