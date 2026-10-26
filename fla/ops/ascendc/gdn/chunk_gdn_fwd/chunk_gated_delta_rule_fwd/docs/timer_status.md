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

## 3. 结论：图上的"KKT 在 K_UPLOAD 前面"是 issue 时刻，不是执行时刻

新数据（含 `PREP_WAIT_*` / `PREP_NOTIFY_S1` / `ANCHOR`）可以定量说明这一点。group 0 的三条 lane：

```
AIC : body 3.914  WAIT_S1_0 [4.558→5.033]  KKT_MM_0 [5.534→6.123]
AIV0: body 4.655  AUX [4.669→5.476] ...    K_UPLOAD [8.191→9.353]  NOTIFY [8.721→8.973]
```

数值 vs 执行：

- 两条 lane 的 `MAIN_INIT` 起点（1.14 vs 1.49）和结束时间（446.5 vs 447.5）都对齐，
  **时钟是共用的**，不存在整体偏移；
- `WaitAivStage1Done` 用的是 `CrossCoreWaitFlag<0x4, PIPE_MTE1>`：**只挡 MTE1，标量不挡**；
- 计时 `Tik/Tok` 是标量读，所以 AIC 的标量在 4.6us 就越过了 wait，并继续下发 KKT 的 Tik（5.5us），
  而真正读 k' 的 MTE1 装载被挡到 AIV 写完 L1（≈9.3us）之后；
- 之后 AIC 的每个 stage 都提前同样的量（PREFILL 7.4 vs 10.2、WAIT_S3 8.7 vs KKT_EPI 13.3），
  到 kernel 末尾两条 lane 又追平。

因此：

- **执行顺序是对的**（k' 一定在 KKT 的 MTE1 装载之前写好），不是同步 bug，也不是时钟错位；
- **图上画的是标量 issue 时刻**，标量跑在 pipe 前面，所以跨 lane 的先后没有意义；
- `PREP_WAIT_*` 只有 0.02–0.08us 也是同一原因：它测的是"标量越过 wait 指令"的时间，
  不是 pipe 真正阻塞的时间，**不能用来量化等待**。

**正确读法**：每条 lane 内看顺序和时长；跨 lane 只看总时长，不看相对位置。

## 4. 正确的两 lane 视图（prepare，单 pack）

用核内可信时长（大 case，us）：AIV `Init ~1.2 → AUX 0.68 → S1 load/norm（窗口外，约 1.8）
→ K_UPLOAD 0.53 → GATE_BETA 0.22 → KKT_GATE 0.14 → KKT_EPI 1.16 → WU_V 0.48 → WU_K 0.55 → DRAIN 0.33`；
AIC `Init ~1.1 → KKT 0.12 → PREFILL 0.04 → ISSUE 0.09 → SOLVE 0.21 → W/U 0.21`。

要点：**AIC 的 issue 会比 AIV 提前 3–4us，但那是标量提前，不是真实等待**；
prepare 的真实构成是 AIV 的搬运 + vector 运算（打点覆盖 77%），等待只占很小一部分。

## 5. 已解决：prepare 计时数据只落一部分（`83ce802d`）

现象（修复前）：固定项 `KERNEL_TIMING` / `MAIN_INIT` / `ANCHOR` 全 0，
动态项只有 `PREP_INIT_MAP`、`PREP_INIT_WS`、`PREP_NOTIFY_S1` 少量有值，
`*_IterCount` 全 0；而 fwd_h / fwd_o 正常。排除了构建、切片布局和解析。

根因：计时区的 GM 写回在写点多的大 kernel（prepare）上没有全部落盘。
修复：启用 `ASCEND_TIMER_EXPLICIT_GM_PUBLISH=1`，并在 prepare / fwd_h / fwd_o 的
`Process()` 末尾（`KERNEL_TIMING` 之后）调用 `timer_.Publish()`，逐 cache line 刷本核区域。
`Publish()` 在被测窗口之外，不影响测量值。

修复后校验：`cores KERNEL>0 = 84`、`cores ANCHOR>0 = 84`，prepare 全部阶段可见。

**注意**：解析器保留了一条兜底（KERNEL 缺失时用动态事件范围），以便旧数据还能读。

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
