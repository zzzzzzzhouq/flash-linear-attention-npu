# chunk_gated_delta_rule_fwd 计时打点设计（A5 三算子路径）

> 本文是 `op-timer-ascendc` skill 的 Step 3 产物：原始流程图 + 打点清单 + 锚点/模式说明。
> 本轮范围：**A5 三算子路径**（`UsePreparePath = true`：BF16 + K=128 + V=128 + chunk=64 + GVA≤4），
> 即 prefill 实际走的路径。融合 Phase6 路径本轮不打点（切片保留占位）。
> 硬约束：**不给等待打点**（不包 `WaitFlag` / `CrossCoreWaitFlag` / `SyncAll`）；不新增任何同步。

## 1. 原始流程图（改造前）

```text
aclnn ChunkGatedDeltaRuleFwd
  ├─ UsePreparePath == true
  │    ├─ ChunkGatedDeltaRuleFwdPrepare（MIX 1:2，一包 4 个 task；G=2 时 owner=2）
  │    │    AIV: S0 常驻辅助
  │    │         pack 循环:
  │    │           S1  Stage1_OneTask: pad tail -> k̂ 归一化/上传(NotifyAicStage1Done) -> gate cumsum/β/导出
  │    │           S3  Stage3_PrepareGate(GateLowerLVF + I_vcs)  [不等 KKT]
  │    │               Stage3_AivOne(WaitCubeKktDone -> L=KKT⊙G -> 打包 -> -L/leaf 上传)
  │    │           drain 上一包 w/u（UB→GM）
  │    │           S6  Stage6_AivOne: vb=v·β(GM→UB→L1) -> kbg=k̂·β·e^g(GM→UB→L1) -> NotifyAicStage6Done
  │    │    AIC: pack 循环:
  │    │           S2  每 (chunk,HK): WaitAivStage1Done -> KKT MMAD -> 通知 sibling
  │    │           S4  Stage4_PrefillI(I·I) -> 每 task: WaitAivStage3Done -> AccIssue -> Dump(t-1)
  │    │           S5  Stage5_Compute -> Stage5_Dump
  │    │           S7  每 task: WaitAivStage6Done -> W=A·kbg -> U=A·vb -> Fixpipe 到 AIV UB
  │    ├─ ChunkFwdH（独立 kernel）
  │    │    AIC: 每 head/chunk: C1（w·S）-> C2（kᵀ·v_new），W/K lookahead 预取
  │    │    AIV: 每 head/chunk: V1（v_new）-> V2（状态更新）
  │    └─ ChunkFwdO（独立 kernel）
  │         AIV: V1（exp*(g) 因子）; 每 head: V2（tril/衰减 -> A′）-> V3（融合输出）
  │         AIC: 每 head: C1（QK + QH）; C3（A′·v_new）
  └─ UsePreparePath == false -> 融合 Phase6（本轮不打点）
```

三个 launch 各自写 timer 张量的一个切片：`SLICE_PREPARE` / `SLICE_FWD_H` / `SLICE_FWD_O`。

## 2. 打点清单

约定：`Tik` 一律 `TikNoBarrier`；`Tok` 一律 `TokNoBarrier<Overwrite>`（窗口是下发/issue window）。
所有窗口**不含等待**：若计算前有功能性等待，Tik 放在等待之后；等待本身不计时。

### Prepare

| 原理步骤 | 枚举 | 位置（`chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h`） | 核 | 固定/动态 | iter 轴 | 说明 |
|---|---|---|---|---|---|---|
| S0 常驻辅助 | `PREP_RESIDENT_AUX` | `Stage0_GenerateResidentAux`（`auxReady` 判定之后） | AIV | 固定 | — | 整个 kernel 一次 |
| S1-pre 尾块补齐 | `PREP_PAD_TAIL` | `Stage1_PadTailIfNeeded`（`nValid >= nPad` 判定之后） | AIV | 动态 | task | 仅 tail 命中 |
| S1a k̂ 上传 | `PREP_K_UPLOAD` | `Stage1_UploadK` 中 **`WaitAicStage4Done(taskIdx)` 之后**到函数结束（Upload + `NotifyAicStage1Done` + q̂ 导出） | AIV | 动态 | task | 放行点（Notify）在窗口内；S4 槽位等待及其之前的 DataCopy/L2Norm 不计入本窗口 |
| S1b gate/β | `PREP_GATE_BETA` | `Stage1_GateAndBeta` | AIV | 动态 | task | 与 Cube KKT 并行 |
| S2 KKT | `PREP_KKT_MM` | `Stage2_AicOne` 的 `MatmulToL0C`（**不含前面的 `WaitFlag<M_MTE1>`**，也不含后面的 dump） | AIC | 动态 | (chunk,HK) | 每 HK 一次，G=2 每包 2 次 |
| S3a gate 预计算 | `PREP_KKT_GATE` | `Stage3_PrepareGate` | AIV | 动态 | (chunk,HV) | 可在等 KKT 之前完成 |
| S3b L 计算 | `PREP_KKT_EPI` | `Stage3_AivOne`，**Tik 放在 `WaitCubeKktDone` 之后**，Tok 在 `NotifyAicStage3Done` 之前 | AIV | 动态 | (chunk,HV) | 纯计算窗口 |
| S4-pre 预填充 | `PREP_SOLVE_PREFILL` | `Stage4_PrefillI` 的 `MmadL0ABToL0C`（循环内逐 task） | AIC | 动态 | task | 与 AIV 的 S3b 重叠 |
| S4-issue | `PREP_SOLVE_ISSUE` | `Stage4_AccIssue` 的 MMAD | AIC | 动态 | task | 等 S3(t) 之后 |
| S4-dump | `PREP_SOLVE_DUMP` | `Stage4_Dump` / `Stage5_Dump` 的 Fixpipe+dump 段 | AIC | 动态 | task | issue(t) 与 dump(t-1) 流水 |
| S5 求逆 | `PREP_SOLVE` | `Stage5_Compute`（`WaitFlag<M_MTE1>` 之后包住两个 `MatmulToL0C`） | AIC | 动态 | task | |
| S6a vb | `PREP_WU_V` | `Stage6_AivOne` 的 v 段（`DataCopy(ubVnd...)` 到 `UploadBf16NdToL1(l1Vb)`） | AIV | 动态 | task | |
| S6b kbg | `PREP_WU_K` | `Stage6_AivOne` 的 k 段（`DataCopy(ubKnd)` 到 `UploadBf16NdToL1(l1Kbg)`） | AIV | 动态 | task | |
| S7a W | `PREP_W_ISSUE` | `Stage7_AicOne` 的 `WuMatmulToL0C(l1A,l1Kbg,...)` | AIC | 动态 | task | 只包 MMAD，不含 helper 内部等待 |
| S7b U | `PREP_U_ISSUE` | `Stage7_AicOne` 的 `WuMatmulToL0C(l1A,l1Vb,...)`（V=256 两次） | AIC | 动态 | task | |
| w/u 回写 | `PREP_WU_DRAIN` | `DrainPackWuToGm` | AIV | 动态 | task | 下一包窗口内 |

### FwdH

| 原理步骤 | 枚举 | 位置（`chunk_fwd_h/op_kernel/arch35/`） | 核 | iter 轴 | 说明 |
|---|---|---|---|---|---|
| C1 P=w·S | `FWH_WH_MM` | `chunk_fwd_h_cube.h` 的 `ComputeStage0Head`：在 W/H/L0/`MTE1_M`/`FIX_M` 全部等待之后，只包 `TileMmadS0` 调用 | AIC | (chunk,head) | 不含等待 |
| V1 v_new | `FWH_VNEW` | `chunk_fwd_h_vec.h` 的 `ConsumeStage1Head` 里 `VF_CALL<FwdHStage1Arch35Vf<...>>`（arch35 无 `epilogueGDNFwdHVnew` 命名，见 §5） | AIV | (chunk,head) | |
| C2 kᵀ·v_new | `FWH_KV_MM` | `chunk_fwd_h_cube.h` 的 `ComputeStage2Head`：在 L0/MTE2/`MTE1_M`/`FIX_M` 等待之后，只包 `TileMmadS2` 调用 | AIC | (chunk,head) | |
| V2 状态更新 | `FWH_STATE` | `chunk_fwd_h_vec.h` 的 `ConsumeStage3Head` 里 `VF_CALL<FwdHStage3Arch35Vf<...>>`（arch35 无 `epilogueGDNFwdHUpdate` 命名，见 §5） | AIV | (chunk,head) | |
| W/K 预取 | `FWH_PREFETCH` | `chunk_fwd_h_cube.h` 的 `LoadStage0W`（W 的 `copyW`）与 `LoadKg`（K/kg 的 `copy`，含 `STATE_V_FIRST` 两支） | AIC | (chunk,head) | MTE2 下发窗口 |

### FwdO

| 原理步骤 | 枚举 | 位置（`chunk_fwd_o/op_kernel/arch35/`） | 核 | iter 轴 | 说明 |
|---|---|---|---|---|---|
| V1 gate | `FWO_GATE` | `chunk_fwd_o_vector.h` 的 `ComputeStage1Gate`（函数体） | AIV | (chunk,head) | |
| C1a q·kᵀ | `FWO_QK` | `chunk_fwd_o_cube.h` 的 `QkTileMmad` 调用前后 | AIC | (chunk,head) | |
| C1b q·h | `FWO_QH` | `chunk_fwd_o_cube.h` 的 `QhTileMmad` 调用前后 | AIC | (chunk,head) | 含 ToUB 变体 |
| V2 mask | `FWO_MASK` | `chunk_fwd_o_vector.h` 的 `VF_CALL<Stage3Gate64VF<true/false>>` 两处 | AIV | (chunk,head) | |
| C3 Attn·v_new | `FWO_ATTN_V` | `chunk_fwd_o_cube.h` 的 `ProcessStage4Head` 里两处 `TileMmad{}` | AIC | (chunk,head) | |
| V3 输出融合 | `FWO_OUTPUT` | `chunk_fwd_o_vector.h` 的 `VF_CALL<Stage5Fuse64VF>` | AIV | (chunk,head) | |

### 固定项（三个 launch 各写一份）

| 枚举 | 落点 | 说明 |
|---|---|---|
| `MAIN_INIT` | 各自 `Init()` 内：`timer_.Init(...)` + `TikNoBarrier()` 之后，`Init()` 返回前 `TokNoBarrier<Overwrite>` | Init → InitDone |
| `MAIN_BODY` | 各自 `Process()` 首行 `TikNoBarrier()`，最后一次 `TokNoBarrier<Overwrite>` | InitDone → kernel 结束 |
| `KERNEL_TIMING_IDX` | 收尾：`PublishCounts()` 之后 `Tik(kernelStart)` + `Tok<Overwrite>`（**唯一带 barrier 的 Tok**） | 整体墙钟 |

`Init()` 内保存 `timerKernelStartCycle_ = timer_.GetStartCycle()`，收尾时用
`timer_.Tik(timerKernelStartCycle_)` 还原总窗口起点，因此 `KERNEL_TIMING_IDX`
覆盖 Init 起点到 kernel 结束，而 `MAIN_BODY` 只覆盖 body。


## 3. 与上一版打点的差异（针对性修正）

| 上一版问题 | 本版处理 |
|---|---|
| `PREPARE_NORM_GATE` 把 k̂ 上传和 gate/β 混在一个窗口，S2 放行点不可见 | 拆 `PREP_K_UPLOAD` / `PREP_GATE_BETA` |
| `KKT_EPILOGUE` 从 `WaitCubeKktDone` 开始，时长含等待 | Tik 移到等待之后（`PREP_KKT_EPI`），等待不计 |
| `H_WH` / `H_KV` 窗口从函数入口开始，含 L0/L1 等待 | Tik 放在等待之后，只包 MMAD |
| `W_MM` / `U_MM` 一个含 helper 内部等待、一个不含 | 只包 `WuMatmulToL0C` 内的 MMAD；不使用旧命名 |
| 回写 w/u 无打点 | 新增 `PREP_WU_DRAIN` |
| S0 无打点 | 新增 `PREP_RESIDENT_AUX` |
| S6 的 vb/kbg 混在一个窗口 | 拆 `PREP_WU_V` / `PREP_WU_K` |
| FwdH 预取无打点 | 新增 `FWH_PREFETCH` |
| 所有 WAIT 项（`FRONT_WAIT` / `H_WAIT` / `O_WAIT`） | **全部不打**（等待由原同步保证，计时不含等待） |

## 4. 读数约定

- 所有动态项是 issue window，**不是完成时延**；异步 MMAD 的完成要看后续功能性等待/产物消费点。
- 窗口可重叠，**不可相加**；总时间只看 `KERNEL_TIMING_IDX`。
- 每核每阶段最多 `MAX_DYNAMIC_ITER = 64` 次，超出截断（`PublishCounts()` → `setDynamicActualIter` 记录实际次数）。
- 跨核绝对时间需先核实 AIC/AIV 共同时钟；未核实时只比较本核内的相对顺序与时长。

## 5. 实现落点与设计的偏差（以代码为准）

1. **FwdH 的 V1/V2 命名**：设计里的 `epilogueGDNFwdHVnew` / `epilogueGDNFwdHUpdate` 是
   arch22（`internal/operators/.../epilogue/block/block_epilogue_gdn_fwdh_*.hpp`）里的命名，
   arch35 走 `chunk_fwd_h_vec.h` 的单 VF 实现。落点改为：
   `ConsumeStage1Head` 的 `VF_CALL<FwdHStage1Arch35Vf<...>>`（`FWH_VNEW`）与
   `ConsumeStage3Head` 的 `VF_CALL<FwdHStage3Arch35Vf<...>>`（`FWH_STATE`）；
   两者都有 `SCALAR_G` / 非 `SCALAR_G` 两支，两支各放一个同类型 scope。
2. **FWH_WH_MM / FWH_KV_MM 的 Tik 位置**：设计写"在 W/H/L0 等待之后"，实现放在
   `TileMmadS0` / `TileMmadS2` 调用前一刻（即 L1→L0 拷贝和 `MTE1_M`/`FIX_M` 等待之后），
   这样窗口内**不含任何等待**，只测 MMAD 下发。`LoadStage0W` / `LoadKg` 的 GM→L1
   搬入确实存在独立调用点，已打 `FWH_PREFETCH`（W 与 K/kg 各一处，K 的
   `STATE_V_FIRST` 两支同类型）。
3. **PREP_K_UPLOAD 不含跨核等待**：窗口从 `WaitAicStage4Done(taskIdx)` **之后**开始，
   覆盖 UploadBf16NdToL1 + `NotifyAicStage1Done` + q̂ 导出；S4 槽位等待本身
   以及等待之前的 DataCopy/L2Norm 不计入本窗口（未新增任何同步，等待仍按原位置执行）。
4. **PREP_SOLVE_PREFILL 逐 task**：`Stage4_PrefillI` 的 `I@I` 在 `nThis` 循环里，
   每个 task 一个 scope、一个 iter，因此 `PREP_SOLVE_PREFILL_N` 对应第 N 个 task。
5. **PREP_SOLVE_DUMP 两处同类型**：`Stage4_Dump`（L0C→GM workspace→L1）与
   `Stage5_Dump`（L0C→GM A / L0C→L1 A）共用 `PREP_SOLVE_DUMP`，按 iter 递增区分。
6. **FwdO 的 C3 与 V2 各有两支**：`ProcessStage4Head` 的满块/尾块两支各一个
   `FWO_ATTN_V`；Stage3 的 `Stage3Gate64VF<true/false>` 两支各一个 `FWO_MASK`，
   两者都是同类型、不同 iter。
7. **FwdO 非 A5 路径不打点**：`chunk_fwd_o.cpp` 的 `ChunkFwdODispatch` 只把 timer
   传到 `ChunkFwdOA5Dispatch`；A2/A3 的 CATLASS 融合实现（`ChunkFwdOKernelImpl`）
   本轮不打点。
8. **timer 为空时零行为差异**：`Init` 里只在 `timerAddr != nullptr` 时 `timer_.Init(...)`，
   之后所有 `TikNoBarrier` / `TokNoBarrier` / scope 构造都走 `t_addr == nullptr` 的
   早退分支（只剩 `AscendTimerDevice` 成员构造本身的开销）；FwdH AIC 侧在
   "本核无 Cube 工作"时完全不写计时缓冲（该核视为未启用）。数值路径不改一行。
