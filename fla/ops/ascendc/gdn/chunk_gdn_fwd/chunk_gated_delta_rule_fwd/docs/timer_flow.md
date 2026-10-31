# GDN A5 前向阶段计时流程

本文只描述 `main` 的 Prepare／FwdH／FwdO 三个独立 A5 kernel。计时缓冲按 kernel 分 slice；三次 launch 的 cycle 不直接拼成一条绝对时间轴。阶段 Tik 全部使用 `TikNoBarrier`，Tok 复用原实现已有的同步点或异步下发边界，不为阶段新增 `SyncAll` 或 `PipeBarrier`。不设置独立 wait／notify 计时项。

## 调用关系

```text
aclnnChunkGatedDeltaRuleFwdProfileGetWorkspaceSize
  └─ ChunkGatedDeltaRuleFwdGetWorkspaceSizeImpl(timer)
     ├─ l0op::ChunkGatedDeltaRuleFwdPrepare(..., timer)
     │  └─ chunk_gated_delta_rule_fwd_prepare + timer slice 0
     ├─ l0op::ChunkFwdH(..., timer)
     │  └─ chunk_fwd_h + timer slice 1
     └─ l0op::ChunkFwdO(..., timer)
        └─ chunk_fwd_o + timer slice 2
```

原 `aclnnChunkGatedDeltaRuleFwd*` ABI 保持不变，三个子算子的原入口均传空 timer。arch22 忽略 timer。

## 改造前原始流程

```text
aclnn host：准备 g/w/u/a 等中间张量，顺序下发三个独立 kernel

Prepare：AIV gate/beta → AIC KKT → AIV mask/leaf
         → AIC AccIssue(t) 与 Dump(t-1) 交错 → AIC A
         → AIV vb/kbg → AIC W/U
FwdH：  可选 AIV initial-state cast → AIC W@H → AIV value/right
         → AIC kg/right → AIV next-state（按 chunk/head 循环）
FwdO：  AIV gate → AIC QK/QH → AIV mask/A′
         → AIC A′@V → AIV output fusion（按 chunk/head 循环）
```

这些节点是原实现的真实工作，不代表额外增加了全局同步。原流水中的 MTE／MMAD／Fixpipe 异步关系及 AIC/AIV ready 链保持不变。

## Prepare

| 名称 | 核型 | 实际工作 | Tik | Tok／完成语义 |
| --- | --- | --- | --- | --- |
| S1_BATE | AIV | gate 累加／缩放及 beta 处理 | `Stage1_GateAndBeta` 入口 | 函数末尾；内部已有向量完成等待，GM 写回仍属异步下发边界 |
| S2_KKT | AIC | `K' @ K'^T` MMAD | `Stage2_AicOne` 输入 ready 后、L0 搬运前 | 原 `M_FIX` 后、`DumpKktToUb` 发布前；不含 UB 发布 |
| S3_MASK | AIV | KKT gate、causal leaf、`-L` 与 leaf 上传 | `Stage3_AivOne` 的 KKT ready 之后 | 末尾 leaf 上传之后；不把 KKT ready 前的跨核等待计入窗口 |
| S4_Y0／S4_Y1 | AIC | `Y=I+LeafLeft@(-L)`，两个 MMAD bank | `Stage4_AccIssue` 输入 ready 后、MMAD 下发前 | 对应 `Stage4_Dump` 的原 `FIX_MTE2` 后、Y 回 L1 **下发后**；不代表 L1 搬运已完成。两个动态类型对应两个 bank，防止交错覆盖起点 |
| S5_A | AIC | `A=LeafLeft+Y@LeafRight` | `Stage5_Compute` 入口 | `Stage5_Dump` 的 MMAD／GM Fixpipe 有原完成锚点；末段 L1 搬运可能只记录到下发边界。同 task 先 Compute 后 Dump |
| S6_VB_KBG | AIV | `vb`、`kbg` 及 ND→L1 | `Stage6_AivOne` 入口 | 两次向量完成及两次上传下发之后；终点是阶段 issue window，不冒充 GM 完成 |
| S7_WU | AIC | `W=A@kbg`、`U=A@vb` | `Stage7_AicOne` 入口 | 函数末尾；W/U MMAD 有原完成锚点，最后 UB／GM Fixpipe 可能只记录到下发边界 |

S4 的 `0/1` 是实现 bank，不是 wait 或 notify。动态 iter 在每个物理核、每个 kernel 内独立递增，跨 pack 不复用槽位。

## FwdH

| 名称 | 核型 | 实际工作 | Tik | Tok／完成语义 |
| --- | --- | --- | --- | --- |
| H_V0_CAST | AIV | 可选 FP32 initial state → BF16 H0 | `ConsumeSMinusOneHead` 输入 ready 后 | 原 `V_MTE3` 后；不含后续 H GM 写回 |
| H_C1_WH | AIC | `P=W@H` | `ComputeStage0Head` 的输入 ready 后 | 原 `M_FIX` 后、P 发布前；窗口可能含实现中原有 UB slot 可用等待 |
| H_V1_VALUE | AIV | gate decay、V_new／right/state | `ConsumeStage1Head` 输入 ready 后 | 原 `V_MTE3` 后；不含后续 GM 写回 |
| H_C2_KG_RIGHT | AIC | `D=kg^T@right` 或 state-v-first 等价形式 | `ComputeStage2Head` 的输入 ready 后 | 原 `M_FIX` 后、D 发布前；窗口可能含实现中原有 UB slot 可用等待 |
| H_V2_STATE | AIV | `R_next=decay*R+D` 并生成下一 H | `ConsumeStage3Head` 的 D／输入 ready 后 | 原 `V_MTE3` 后；不含后续 state／H GM 写回 |

## FwdO

| 名称 | 核型 | 实际工作 | Tik | Tok／完成语义 |
| --- | --- | --- | --- | --- |
| O_V1_GATE | AIV | gate_o、gate_A | `ComputeStage1Gate` 入口 | 末尾原 `PipeBarrier<PIPE_V>` 后 |
| O_C1_QK | AIC | `Q@K^T` | Q/K 输入 ready 后、QK L0 搬运前 | QK 原 `M_FIX` 后、结果发布前 |
| O_C2_QH | AIC | `Q@H` | QK 完成后、QH L0 搬运前 | QH 原 `M_FIX` 后、结果发布前 |
| O_V2_MASK | AIV | causal mask、O_s gate、A′ | Stage2 ready 与本地输入 ready 后 | 原 `V_MTE3` 后；不含 A′ 到 L1 的后续搬运 |
| O_C3_ATTNV | AIC | `A′@V` | `ProcessStage4Head` 的 V ready 后 | 原 `M_FIX` 后、O_l 发布前 |
| O_V3_OUTPUT | AIV | O_s/O_l 融合与缩放 | Stage4 ready 且 owner AIV 确认后 | 原 `V_MTE3` 后；不含最终 O GM 写回 |

所有窗口是“核上代码从 Tik 到原同步点或异步下发边界”的执行窗口，不等同于纯指令独占时长；窗口中原有的数据就绪和资源复用等待保持可见。最终 kernel 总计时允许使用一次末尾 `Tok`，但不据此推断其他 kernel 的绝对先后。
