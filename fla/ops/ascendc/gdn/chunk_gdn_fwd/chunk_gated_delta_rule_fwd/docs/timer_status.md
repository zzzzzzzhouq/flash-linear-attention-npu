# GDN fwd 计时分析结论与遗留问题（2026-10-26）

本文只记录**已用数据证实**的结论，以及计时器当前**未解决**的问题。后续重启这条线时，先读本文，
再看 `timer_flow.md`（打点设计）和 `tests/ut/README.md`（采集/解析用法）。

## 1. 结论：三段是串行，不是并行

同一份 trace（`457c2f01...json`，单 pid `rank0`，A5）把 prepare / fwd_h / fwd_o 放在**一条绝对时间轴**上：

| kernel | 区间 (us) | 时长 (us) | 占比 |
| --- | --- | --- | --- |
| prepare | 0 – 43.6 | 43.6 | 43% |
| fwd_h | 45.6 – 64.7 | 19.1 | 36% |
| fwd_o | 66.7 – 83.4 | 16.8 | 21% |

段间约 2us 的下发间隔，与 aclnn 在同一 executor 里顺序下发
`ChunkGatedDeltaRuleFwdPrepare → ChunkFwdH → ChunkFwdO` 一致（见
`op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp`）。因此端到端 ≈ 三段之和，
**不能用某一段的窗口代表全程**。

## 2. 结论："第一个块是 KKT_MM" 是读法问题，不是执行顺序

代码在 `ProcessAic` 里先 `WaitAivStage1Done(t)` 再 `Stage2_AicOne(t)`，
而 `NotifyAicStage1Done` 在 AIV 的 `Stage1_UploadK` 内部、`UploadBf16NdToL1` 之后发出。
所以同一 pack 内恒有：

```
PREP_K_UPLOAD.start < notify < KKT_MM.start
```

"KKT 第一"来自两个叠加：

1. AIC lane 上 KKT 之前没有任何打点（tiling 载入、flag 预热、stage1 等待都不在窗口内），
   所以它是 AIC lane 上第一个可见事件；
2. 把 AIC/AIV 两条 lane 合并、按最小 ts 排序，AIC 的 init 又比 AIV 短
   （prepare 实测 AIC 5.2us / AIV 6.8us），KKT 就排到了最前面。

**正确读法**：两条 lane 分开看，跨 lane 只保留代码保证的握手（notify → wait）。

## 3. 结论：跨核绝对时间戳目前不可比

同一份数据里有两处**硬违反**（依赖是代码保证的）：

| 依赖 | AIC 侧 | AIV 侧 | 偏差 |
| --- | --- | --- | --- |
| KKT 在 stage1 通知之后 | KKT **6.98** | K_UPLOAD **13.93–15.58** | 早 7–8.5us |
| `Stage4_AccIssue` 在 `WaitAivStage3Done` 之后 | ISSUE **11.21** | KKT_EPI **22.77** | 早 11.5us |

而两条 lane 的 INIT 起点（0.52 vs 0.68）和 body 时长（35.6 vs 34.7）又是一致的，
排除了整体时钟偏移/频率差异。结论：

- **核内**的 duration 有效（同一计数器做差）；
- **跨核**的绝对位置不可用，凡是"谁先谁后 / 重叠 / 谁等谁"的判断都作废。

## 4. 正确的两 lane 视图（prepare，单 pack）

用核内可信时长（us）：AIV `Init 6.8 → AUX 1.5 → S1 load/norm 3.2 → K_UPLOAD 1.6
→ GATE_BETA 1.8 → KKT_GATE 1.2 → KKT_EPI 6.4 → WU_V 2.1 → WU_K 1.8 → DRAIN 1.2`；
AIC `Init 5.2 → [WaitAivStage1Done ≈ 9.6] → KKT → PREFILL → ISSUE/DUMP → SOLVE/DUMP → W/U`。

要点：**AIC 的 body 起点到 KKT 之间是一段约 9.6us 的等待**，这才是 prepare 的主要空转；
优化方向是缩短 AIV 的 init + stage1，而不是继续压 vector 计算。

## 5. 遗留：新计时器在 prepare 上只落了部分事件

已加的诊断打点（`PREP_INIT_MAP/WS/GM`、`PREP_WAIT_S1/S3/S6/S4/CUBE`、`PREP_NOTIFY_S1`、
`ANCHOR`）在设备上表现异常，**尚未解决**：

- 固定项 `KERNEL_TIMING` / `MAIN_INIT` / `ANCHOR` 全 0；
- 动态项只有 `PREP_INIT_MAP`、`PREP_INIT_WS`、`PREP_NOTIFY_S1` 有值，各 8 个核；
  `PREP_K_UPLOAD`、`PREP_WAIT_*` 全 0；
- `*_IterCount` 全 0（`PublishCounts` 未落盘）；
- fwd_h / fwd_o 的 `KERNEL_TIMING` 正常，说明不是全局布局问题。

**已排除**：构建未更新（新枚举 type 27/28/35 已出现在数据里）、切片布局错位
（`slice_words=448512`，fwd_h/fwd_o 都已按新布局写入）、解析丢数据
（已加兜底，`rank_0_prepare_parsed.csv` 有 96 行）。

**下一步**：在 `write_time()` 前后加设备 printf（只加日志），确认是"没写"还是"写了没落盘"；
或写一个只记 4 个事件的极小 kernel 验证 publish 路径。定位前不要再用 prepare 的新窗口做结论。

## 6. 复现

```bash
FLA_NPU_SOC=ascend950 FLA_NPU_OPS= python scripts/build_wheel.py
python -m pip install --force-reinstall --no-cache-dir --no-deps dist/<wheel>
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/profile_gdn.py --device 0
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut/parse_timer_csv.py \
    --dir profiling_results/gdn_fwd_timer --rank-start 0 --rank-end 0 --slice prepare
```
