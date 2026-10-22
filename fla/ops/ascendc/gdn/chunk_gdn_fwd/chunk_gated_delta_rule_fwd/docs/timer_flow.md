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
| S0 常驻辅助 | `PREP_RESIDENT_AUX` | `Stage0_GenerateResidentAux` | AIV | 固定 | — | 整个 kernel 一次 |
| S1-pre 尾块补齐 | `PREP_PAD_TAIL` | `Stage1_PadTailIfNeeded` | AIV | 动态 | task | 仅 tail 命中 |
| S1a k̂ 上传 | `PREP_K_UPLOAD` | `Stage1_UploadK`（**到 NotifyAicStage1Done 为止**） | AIV | 动态 | task | 放行点在这里；q̂ 导出也在此窗口内，需在报告中注明 |
| S1b gate/β | `PREP_GATE_BETA` | `Stage1_GateAndBeta` | AIV | 动态 | task | 与 Cube KKT 并行 |
| S2 KKT | `PREP_KKT_MM` | `Stage2_AicOne` 的 `MatmulToL0C`（**不含前面的 `WaitFlag<M_MTE1>`**） | AIC | 动态 | (chunk,HK) | 每 HK 一次，G=2 每包 2 次 |
| S3a gate 预计算 | `PREP_KKT_GATE` | `Stage3_PrepareGate` | AIV | 动态 | (chunk,HV) | 可在等 KKT 之前完成 |
| S3b L 计算 | `PREP_KKT_EPI` | `Stage3_AivOne`，**Tik 放在 `WaitCubeKktDone` 之后** | AIV | 动态 | (chunk,HV) | 纯计算窗口 |
| S4-pre 预填充 | `PREP_SOLVE_PREFILL` | `Stage4_PrefillI` 的 `MmadL0ABToL0C` | AIC | 动态 | task | 与 AIV 的 S3b 重叠 |
| S4-issue | `PREP_SOLVE_ISSUE` | `Stage4_AccIssue` 的 MMAD | AIC | 动态 | task | 等 S3(t) 之后 |
| S4-dump | `PREP_SOLVE_DUMP` | `Stage4_Dump` / `Stage5_Dump` 的 Fixpipe | AIC | 动态 | task | issue(t) 与 dump(t-1) 流水 |
| S5 求逆 | `PREP_SOLVE` | `Stage5_Compute` | AIC | 动态 | task | |
| S6a vb | `PREP_WU_V` | `Stage6_AivOne` 的 v 段（到 `UploadBf16NdToL1(l1Vb)`） | AIV | 动态 | task | |
| S6b kbg | `PREP_WU_K` | `Stage6_AivOne` 的 k 段（`DataCopy(ubKnd)` 起） | AIV | 动态 | task | |
| S7a W | `PREP_W_ISSUE` | `Stage7_AicOne` 的 `WuMatmulToL0C(l1A,l1Kbg,...)` | AIC | 动态 | task | 只包 MMAD，不含 helper 内部等待 |
| S7b U | `PREP_U_ISSUE` | `Stage7_AicOne` 的 `WuMatmulToL0C(l1A,l1Vb,...)`（V=256 两次） | AIC | 动态 | task | |
| w/u 回写 | `PREP_WU_DRAIN` | `DrainPackWuToGm` | AIV | 动态 | task | 下一包窗口内 |

### FwdH

| 原理步骤 | 枚举 | 位置（`chunk_fwd_h/op_kernel/arch35/`） | 核 | iter 轴 | 说明 |
|---|---|---|---|---|---|
| C1 P=w·S | `FWH_WH_MM` | `chunk_fwd_h_cube.h` 的 MMAD（**Tik 在 W/H/L0 等待之后**） | AIC | (chunk,head) | 不含等待 |
| V1 v_new | `FWH_VNEW` | `chunk_fwd_h_vec.h` 的 `epilogueGDNFwdHVnew` | AIV | (chunk,head) | |
| C2 kᵀ·v_new | `FWH_KV_MM` | `chunk_fwd_h_cube.h` 的 MMAD | AIC | (chunk,head) | |
| V2 状态更新 | `FWH_STATE` | `chunk_fwd_h_vec.h` 的 `epilogueGDNFwdHUpdate` | AIV | (chunk,head) | |
| W/K 预取 | `FWH_PREFETCH` | W/K 的 GM→L1 搬入点 | AIC | (chunk,head) | MTE2 下发窗口 |

### FwdO

| 原理步骤 | 枚举 | 位置（`chunk_fwd_o/op_kernel/arch35/`） | 核 | iter 轴 | 说明 |
|---|---|---|---|---|---|
| V1 gate | `FWO_GATE` | `chunk_fwd_o_vector.h` 的 `ComputeStage1Gate` | AIV | (chunk,head) | |
| C1a q·kᵀ | `FWO_QK` | `chunk_fwd_o_cube.h` 的 `blockMmadQK` | AIC | (chunk,head) | |
| C1b q·h | `FWO_QH` | `chunk_fwd_o_cube.h` 的 `blockMmadQH*` | AIC | (chunk,head) | 含 ToUB 变体 |
| V2 mask | `FWO_MASK` | `chunk_fwd_o_vector.h` 的 `Stage3Gate64VF` | AIV | (chunk,head) | |
| C3 Attn·v_new | `FWO_ATTN_V` | `chunk_fwd_o_cube.h` 的 `TileMmad` | AIC | (chunk,head) | |
| V3 输出融合 | `FWO_OUTPUT` | `chunk_fwd_o_vector.h` 的 `Stage5Fuse64VF` | AIV | (chunk,head) | |

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
- 窗口可重叠，**不可相加**；总时间只看 `KERNEL_TIMING`。
- 每核每阶段最多 `MAX_DYNAMIC_ITER = 64` 次，超出截断（`setDynamicActualIter` 记录实际次数）。
- 跨核绝对时间需先核实 AIC/AIV 共同时钟；未核实时只比较本核内的相对顺序与时长。
