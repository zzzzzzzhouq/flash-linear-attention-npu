# GDN 三阶段 Profiling、瓶颈与优化分析

> 分析对象：`feat/gdn-timer-align` 的 A5 Prepare／FwdH／FwdO 三算子路径。
> 数据：修改后采集的 512 tokens 与 8960 tokens 三切片 JSON，结合当前实现、设计文档和逐事件统计。
> 证据边界：本文解释这两个 case 的代码结构与带 timer 采集；**8960 FwdH 的整体构成已剔除已识别的 159.800 µs 槽 63 尾跳变，但校正值仍不是 Guard 修复后的重采或无 timer 实测。**

## 1. 执行摘要

1. **512 tokens 的采集表观瓶颈是 Prepare。** Prepare／FwdH／FwdO 跨度分别为 35.700／26.125／11.359 µs，表观占比为 48.8%／35.7%／15.5%。这个 shape 的 FwdH 没有动态槽溢出，但数值仍是带 timer 的采集结果，最终性能需用无 timer A/B 验收。
2. **8960 tokens 的 FwdH 校正跨度为 144.133 µs。** 原始跨度 303.933 µs 中，H_C1／H_C2 等事件在槽 62→63 处包含约 159.800 µs 的尾跳变；按 `303.933 − 159.800` 剔除后，FwdH 为 144.133 µs，三阶段合计 247.673 µs，校正占比为 58.2%。原始 74.6% 不再用于整体构成分析。
3. **FwdH 仍是代码可确认的结构瓶颈候选：4 个活跃核组 + 长串行递推。** 当前 `B=1、HV=4`，每个 value head 的全部 chunks 固定在同一核组顺序执行；8960 下每 head 有 139 次 state 更新。token 增多会延长递推链，却不会增加活跃核组数。这个结构判断不依赖原始长 trace，但优化收益必须重新测量。
4. **Prepare 的主要问题是 Stage3 数据链和首次跨 pack 交接，不是已证实的 KKT MMAD throughput bound。** 8960 下 pack 1 的 notify→KKT 中位为 6.460 µs、S3 等待合计为 10.267 µs；稳态 S3 EPI 约 3.8 µs／pack。每核组五包 `S2_KKT` 合计仅约 1.6 µs，而 `S3B_EPI` 约 22.5 µs。
5. **FwdO 局内最大命名 AIV 阶段是 Mask，但整体优先级靠后。** 8960 下 28 个核组活跃，FwdO 表观跨度 28.544 µs，占三阶段校正总和的 11.5%；先做大范围 FwdO 重构，整体收益空间有限。
6. **下一步用修复后的 Guard 重新采集，再做优化性能验收。** 当前对齐源码已在溢出后通过现有 Tik/Tok Guard 丢弃后续采样，保留每 Stage 首次记录的 0～63 槽；4 处 `PIPE_S` 不需要再改。重新采集后，用 `msprof op_summary Task Duration` 与 `--no-timer` A/B 确认真实性能。

![GDN 三阶段采集跨度构成](assets/gdn_profiling/gdn_overview.svg)

## 2. Case、数据和分析口径

### 2.1 已测 case 与工作分布

两个 case 均为：`B=1、HK=2、HV=4、K=V=128、chunk=64`，q/k/v 为 BF16，g/beta 为 FP32；调用不请求 final state，不走 kernel 内 QK L2Norm。

| tokens | chunk 数 | Prepare 工作分布 | FwdH 工作分布 | FwdO 工作分布 |
|---:|---:|---|---|---|
| 512 | 8 | 8／28 核组，各 1 个 pack | 4 核组，每 head 8 chunks | 8／28 核组，各 1 个 group task |
| 8960 | 140 | 28／28 核组，各 5 个 packs | 4 核组，每 head 140 chunks | 28／28 核组，各 5 个 group tasks |

Prepare 的一个完整 pack 在当前 `G=HV/HK=2` 下对应一个 chunk 的 4 个 value-head task、2 份 KKT。FwdO 的一个 group task 同样覆盖一个 chunk 的 4 个 value heads。FwdH 按 `(sequence, value_head)` 分派，每个 head 的全部 chunks 固定在同一核组顺序执行。[design](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/docs/design.md&line=176&text=Host "citation")

#### 2.1.1 当前 HV=4 与目标 HV=32 的调度对比

目标采集 shape 为 `B=1、HK=16、HV=32、K=V=128、chunk=64`，因此 `G=2` 不变。Prepare 的 pack 和 FwdO 的 group task 仍然都是 4 个 value heads，但每 chunk 的组数从 `ceil(4/4)=1` 增至 `ceil(32/4)=8`。Prepare 按 `pack += numCore` 跨步领取，FwdO 也按 `groupTaskIdx += coreNum` 跨步领取。[prepare_kernel](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h&line=779&text=const%20int64_t%20packSize "citation") [chunk_fwd_o_cube](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h&line=107&text=groupTaskNum "citation")

| 阶段 | tokens | 当前 HK=2、HV=4 | 目标 HK=16、HV=32 |
|---|---:|---|---|
| Prepare | 512 | 8 packs；8／28 核组，各 1 包 | 64 packs；28／28 核组，20 组各 2 包、8 组各 3 包 |
| Prepare | 8960 | 140 packs；28／28 核组，各 5 包 | 1120 packs；28／28 核组，各 40 包 |
| FwdH | 512 | 4／28 核组，各 1 head；每链 8 chunks | 16／28 核组，各 2 heads；每链仍为 8 chunks |
| FwdH | 8960 | 4／28 核组，各 1 head；每链 140 chunks | 16／28 核组，各 2 heads；每链仍为 140 chunks |
| FwdO | 512 | 8 group tasks；8／28 核组，各 1 组 | 64 group tasks；28／28 核组，20 组各 2 个、8 组各 3 个 |
| FwdO | 8960 | 140 group tasks；28／28 核组，各 5 组 | 1120 group tasks；28／28 核组，各 40 组 |

FwdH 的 16 核组来自 host 公式，而不是把 32 个 heads 当成 32 个核：`headsPerCore=ceil(32/28)=2`，`blockDim=ceil(32/2)=16`；每核组处理连续的两个 value heads。[chunk_fwd_h_tiling_processor](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_host/chunk_fwd_h_tiling_processor.h&line=84&text=const%20uint64_t%20totalHeadTasks "citation")

本次目标采集用 `tokens=11264`，即 176 个完整 chunks：Prepare／FwdO 各有 `176×8=1408` 个 packs／group tasks，28 核组中 20 组各 50 个、8 组各 51 个；FwdH 为 16 核组各 2 heads，每 head 仍有 176 个顺序 chunks。

**调度变化的正确解读：** Prepare／FwdO 从部分或小规模填充变为 28 核组满载，但每核组工作量也增至同 token 下的 8 倍；FwdH 活跃核组增至 4 倍，每核组承担 2 个 heads，而单 head 递推链长不变。因此 HV=32 能改善并行度，却不能仅由核数判断总耗时升降，必须重新采集三阶段 Task Duration 和内部 trace。

### 2.2 时间轴与跨度

- 分析使用 JSON 原始 `start_cycle/end_cycle`，按 `1800 cycles/µs` 换算。
- 不做逐核 ANCHOR 对齐。脚本核验 `ts == (start_cycle - 全 slice 最小 start_cycle) / 1800`，最大误差为 0，保留核启动散布和跨核相对位置。[gdn-modified-steady-analysis](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-steady-analysis.txt&line=34&text=raw%20min-start%20check "citation")
- Prepare、FwdH、FwdO 三个切片各自独立取 min-start。三阶段跨度之和只用于描述采集构成，**不能**用于推算真实 launch gap，也不等于端到端墙钟。
- 512 使用原始采集跨度。8960 的整体构成使用校正跨度：仅从 FwdH 原始跨度中剔除已识别的 159.800 µs 槽 63 尾跳变，Prepare／FwdO 不变；“校正占比”指各段占校正后三阶段之和的比例。其他逐事件分析仍保留原始 timer 口径。

### 2.3 Timer 窗口不能等同于 pipe busy time

部分窗口只包指令下发，硬件完成由后续同步保证；MTE2／MTE3 窗口可能包含本地等待；不同核和不同 pipe 的窗口还会重叠。因此本文使用“命名窗口”“可见间隔”“起点节奏”，不把窗口总和当成硬件利用率，也不据此宣称 DDR、Cube 或 Vector 已打满。[timer_flow](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/docs/timer_flow.md&line=112&text=%E8%AF%BB%E6%95%B0%E7%BA%A6%E5%AE%9A "citation")

## 3. 三个 kernel 在算什么

### 3.1 Prepare：构造递推和输出所需矩阵

```text
S0  生成常驻单位阵、scatter index 和零块
S1  k 上传 L1 并 notify；并行计算 q/g 累加和、beta_eff
S2  KKT = k_hat @ k_hat^T                    AIC
S3  gate lower 预计算；L = KKT ⊙ G，打包叶块并上传 L1
S4  Y = I + LeafLeft @ (-L)                 AIC
S5  A = LeafLeft + Y @ LeafRight             AIC
S6  vb = v·beta；kbg = k_hat·beta·exp(g)     AIV
S7  W = A @ kbg；U = A @ vb                  AIC
```

#### 3.1.1 pack 是什么：task 分组与资源生命周期单位

Prepare 里的 **pack 不是矩阵 packing／拼接，也不是一个 token chunk 的别名**。最小的 work item 是一个 `(chunk, value_head)` task；pack 则是 kernel 一次领取的一组 task，用来确定 AIV 分工、同一 HK 下的 KKT 复用，以及 k_hat／NegL、W／U 等片上缓冲的释放和交接边界。[prepare_common](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare_common.h&line=317&text=G%3D3%20cannot%20fill%204%20consecutive%20HV "citation")

当前 `G=HV/HK=2`，一个完整 pack 有 4 个 task，覆盖同一 chunk 的 4 个 value heads 和 2 个 HK。task 顺序不是简单的 hv0／1／2／3，而是把两个 HK 的 owner 放在前面：

| pack 内 task | value head | HK | 在 HK 组内的角色 | 执行 AIV |
|---:|---:|---:|---|---|
| 0 | hv0 | HK0 | owner，产生该 HK 的 KKT | AIV0 |
| 1 | hv2 | HK1 | owner，产生该 HK 的 KKT | AIV1 |
| 2 | hv1 | HK0 | sibling，复用 HK0 的 KKT | AIV0 |
| 3 | hv3 | HK1 | sibling，复用 HK1 的 KKT | AIV1 |

因此一个完整 pack 只算 **2 份 KKT**，不是每个 value head 都算一份。更一般地，`G=3` 时一个 pack 放 3 个 task、覆盖 1 个 HK；其余情况一个完整 pack 放 4 个 task，`G=1／2／4` 分别覆盖 4／2／1 个 HK。[prepare_kernel](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h&line=8&text=Pack%3A%20TG%20%3D%203%20if%20G%3D3%20else%204 "citation")

当前 case 的 work item 总数为 `chunk 数 × HV=4`，所以 pack 数为：

- **512：**`8 chunks × 4 ÷ 4 tasks/pack = 8 packs`。28 个核组按 `pack = coreIdx; pack += numCore` 跨步领取，只有 8 个核组各执行 1 包。
- **8960：**`140 × 4 ÷ 4 = 140 packs`，28 个核组全部活跃，每个核组依次执行 5 包。[prepare_kernel](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h&line=779&text=const%20int64_t%20packSize "citation")

pack 边界也是跨包流水的依据：下一 pack 的 S1 只等上一 pack S4 释放 k_hat／NegL，不等 S7；上一 pack 的 W／U 先留在对应 AIV 的 UB，下一 pack 再把它们 drain 到 GM。于是一个核组内可以出现“上一包 S5／S7／回写”与“下一包 S1”重叠。profiling 按 pack 分组，就是为了把首包、首次交接和稳态包分开，而不是把不同 chunk／head 的事件混成一条平均时间线。

主依赖是 `S1 → S2 → S3 → S4 → S5 → S7`，同时存在两条重要重叠：

- S6 只依赖 S3 和输入，不必等 S4／S5，因此可与 Cube 的 S4／S5 重叠。
- 下一 pack 的 S1 只等上一 pack S4 释放 k_hat／NegL 槽，不等上一 pack S7，因此可覆盖上一 pack S5／S7。

代码已实现 KKT 与 gate/beta 重叠、S4 prefill 藏在 S3 下、S4 issue(t) 与 dump(t−1) 重叠，这些不是待补的新优化。[chunk_gated_delta_rule_fwd_prepare](kimi-cite://text?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h&text=Do%20not%20gate%20Stage1%20on%20Stage7 "citation")

![Prepare CV 依赖与跨 pack 重叠](assets/gdn_profiling/gdn_prepare_pipeline.svg)

### 3.2 FwdH：沿 chunk 严格递推 state

#### 3.2.1 先分清 R、H、V_new 和 right

- `R_c`：chunk 入口处的**真实 rolling state**。当前 FP32 路径把它保存在 AIV UB，Stage3 用 FP32 算术更新。
- `H_c = cast_BF16(R_c)`：入口 state 的 **BF16 快照**。AIC 用它算 `P=W_c@H_c`，FwdO 也用它算当前 chunk 的历史输出项。
- `V_new,c`：当前 chunk 扣掉旧 state 已能解释的部分后，真正要写入的新内容，FwdO 的 `A′@V_new` 也需要它。
- `right_c`：带相对衰减的 `V_new`，只供本 chunk 的 `D=K_c^T@right_c` 使用。

普通 chunk 的完整数据流是：

```text
UB 中的 FP32 R_c
    ├─生成 BF16 H_c──────────────→ FwdO 使用 H_c
    ↓
AIC Stage0：P = W_c @ H_c
    ↓ P
AIV Stage1：V_new = cast_BF16(fp32(U_c) - fp32(P))
    ├─写 V_new,c─────────────────→ FwdO 使用 V_new,c
    ↓ 再乘 exp(g_last-g_i)
AIV 生成 right_c
    ↓ right_c
AIC Stage2：D_c = K_c^T @ right_c
    ↓ D_c
AIV Stage3：R_{c+1} = exp(g_last) * R_c + D_c
    ├─更新 UB 中的 FP32 rolling state
    └─生成并写出下一块 BF16 H_{c+1}
```

所以 Stage1 和 Stage2／3 服务的是两个不同消费者：**FwdO 要当前块的 H_c 和 V_new,c；下一块 FwdH 才需要 right_c、D_c 和更新后的 R_{c+1}。**[design](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/docs/design.md&line=13&text=%E8%AE%A1%E7%AE%97%E8%AF%AD%E4%B9%89 "citation")

#### 3.2.2 首个无初态 chunk：为什么跳过 P=W@H

无 `initial_state` 时，数学上入口状态是零矩阵 `R_0=0`，因此：

```text
P_0 = W_0 @ H_0 = W_0 @ 0 = 0
V_new,0 = U_0 - P_0 = U_0
```

代码在 `chunk.first && useInitialState==0` 时直接从 Stage0 返回；AIV 侧同一个条件令 `hasP=false`，不再等待或读取 P，并按零状态路径计算 `V_new=U`。这不是跳过整个首块：首块仍然生成 FwdO 需要的零 H_0 和 V_new,0；只要后面还有 chunk，仍然要生成 right_0、计算 D_0 并更新 R_1。[chunk_fwd_h_cube](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h&line=470&text=chunk.first "citation") [chunk_fwd_h_vec](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h&line=797&text=hasP "citation")

#### 3.2.3 最后 chunk 不请求 final_state：为什么跳过 Stage2／Stage3

设最后 chunk 为 `c=139`。FwdO 需要的是它**入口处**的 `H_139` 和本块 `V_new,139`，并不需要 `R_140`；调用方又不要求输出 final state，所以以下产物没有消费者：

```text
right_139 → D_139 → R_140 / final_state
```

kernel 因而仍执行 Stage0 和 Stage1，生成 `P_139` 与 `V_new,139`，但不生成 right_139，也不执行 K 装载、`D=K^T@right` 和 state 更新。`NeedsStage2/NeedsStage3` 的条件都是 `storeFinalState || !chunk.last`。[chunk_fwd_h_cube](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h&line=702&text=NeedsStage2 "citation") [chunk_fwd_h_vec](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h&line=798&text=writeRight "citation")

因此 8960 case 每 head 的计数是：

| 项目 | 数量 | 原因 |
|---|---:|---|
| Stage1／V_new | 140 | 每个 chunk 都要给 FwdO 提供 V_new，包括首块和尾块 |
| 完整 right→D→R_next 更新 | 139 | 只有 chunk 0～138 需要为下一块产生状态 |
| 尾块 Stage2／Stage3 | 0 | 不请求 final state，R_140 没有消费者 |

若请求 final state，尾块也会执行完整 Stage2／Stage3；若序列只有一个 chunk 且无初态、无 final state，则 Stage0 和 Stage2／3 都可省略，只保留必要的 H／V_new 生成。

#### 3.2.4 W/K 双 bank：具体预取什么、在哪里重叠

当前 FP32 scalar-g **单 head work unit** 在 AIC L1 中给 W 准备 slot 0／1、给 K 准备 slot 2／3，按 chunk 奇偶使用：

```text
chunk c 使用：       W[c%2]、K[2+c%2]
chunk c+1 预取到：   W[(c+1)%2]、K[2+(c+1)%2]
```

进入循环前先装载首块 W／K。处理 chunk c 时，AIC 在当前 right 的 GM→L1 已经下发后，立即向另一组 bank 下发 chunk c+1 的 W／K；随后继续使用当前 bank 做 Stage2。这样下一块的 MTE2 输入搬运可以覆盖当前块的 MTE1／MMAD／Fixpipe，而不会覆盖正在使用的 W／K。实现见单 head 流水。[chunk_fwd_h_cube](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h&line=715&text=%E5%8D%95%20head%20%E7%A8%B3%E6%80%81 "citation")

它隐藏的是**输入搬运延迟**，不是递推等待：下一块 H 仍必须等上一块 Stage3 发布 `H_READY`，所以普通 W/K lookahead 不能把同一 head 的 chunks 并行化。

#### 3.2.5 U/g 双 bank：AIV 侧如何配合

AIV 也用 chunk 奇偶选择两个 U/g input bank。处理 chunk c 前，先向另一个 bank 下发 chunk c+1 的 U/g，再消费当前 bank 做 Stage1；当前计算与下一块 MTE2 因此重叠。自由信用防止预取覆盖尚未消费的数据：U/V_new work bank 由 MTE3→MTE2 free credit 保护，gate/alpha bank 等 Stage3 的最后向量消费者完成后才释放。[design](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/docs/design.md&line=160&text=lookahead "citation")

也就是说，W/K 双 bank 在 AIC L1，U/g 双 bank 在 AIV UB，两边分别隐藏输入流；P、D、right、H 的生产消费顺序仍由 ready/free 协议维持。

#### 3.2.6 FP32 rolling state 常驻：哪些 GM 搬运已经省掉

A5 FP32 路径在每个 AIV 的 UB 中预留两份 64 KiB state bank，分别绑定 local head slot 0／1；当前单 head work unit 使用 local slot 0。首块有 initial state 时只从 GM 读入一次，无初态时直接在 VF 中构造零状态。之后每个 Stage3 都在 UB 内原地执行 `R_next=decay*R+D`，后续 chunk 不再从 GM workspace 恢复 R，也不逐块把 R 写回 workspace。[design](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/docs/design.md&line=124&text=state "citation")

但“state 常驻”不等于 FwdH 不再写 GM：每块仍要写出 BF16 H 快照供下一块 AIC 和 FwdO 使用，V_new 仍是 FwdO 输入，right 在当前实现中仍需经 GM workspace 交给 AIC。只有请求 final state 时，末尾才把完整 FP32 R 写到 final_state。

因此，“再增加普通 W/K 预取”和“把 FP32 state 改为常驻”都已经是现状，不是新的优化方案。后续优化应针对尚未消除的 right／V_new／H 数据路径、4 核组并行度和 CV 交接。

### 3.3 FwdO：chunk 内输出计算

```text
V1     计算 gate_o、gate_A
C1a    A_raw = Q @ K^T
C1b    O_s   = Q @ H
V2     tril + 衰减生成 A′，同时生成缩放后的 O_s′，A′写入 L1
C3     O_l   = A′ @ V_new
V3     O     = scale · (O_s′ + O_l)，写回 GM
```

Q/K 的 L1 输入在共享同一 HK 的 value heads 间复用，QK 也只在每个 HK 的首个 value-head 路径上装载并计算，其余 value heads 复用结果。[chunk_fwd_o_cube](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h&line=330&text=Load%20Q%20and%20K%20once "citation") V_new 的 GM→L1 预取领先一个 head，L0 使用 ping/pong。依赖链是 `C1a/C1b → V2 → C3 → V3`，不同 head 和不同 group task 在数学上独立，但缓冲复用仍需组边界同步。

## 4. 剔除 FwdH 尾跳变后的整体构成

| tokens | Prepare | FwdH | FwdO | 三阶段跨度之和 |
|---|---:|---:|---:|---:|
| 512 | 35.700 µs（48.8%） | 26.125 µs（35.7%） | 11.359 µs（15.5%） | 73.184 µs |
| 8960（校正值） | 74.996 µs（30.3%） | 144.133 µs（58.2%） | 28.544 µs（11.5%） | 247.673 µs |

512 的三段跨度来自新采集统计。[gdn-modified-comparison](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-comparison.txt&line=21&text=gdn_modified_download.json "citation") 8960 的 Prepare／FwdH／FwdO 原始跨度分别为 74.996／303.933／28.544 µs。[gdn-modified-comparison](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-comparison.txt&line=40&text=gdn_modified_8960.json "citation")

8960 的校正计算为：`FwdH = 303.933 − 159.800 = 144.133 µs`，`三阶段合计 = 74.996 + 144.133 + 28.544 = 247.673 µs`。这里只剔除槽 62→63 的已识别尾跳变，不再使用原始 303.933 µs／74.6% 描述整体构成。

**校正值仍不是无 timer 实测，也不等价于 Guard 修复后的完整重采。** 它用于消除已定位的尾部采样伪影；FwdH 的完整跨 Stage 形状和最终性能仍需重新采集确认。

更新后的 shape 结论是：

- **512：** Prepare 是本次采集的最大段，首包、初始化和单链填充值得优先检查。
- **8960：** 校正后 FwdH 仍占 58.2%，高于 Prepare 的 30.3% 和 FwdO 的 11.5%；同时代码上只有 4 核组并承担 140 chunks／head 的串行链，因此它仍是首要结构优化候选。
- **仍需重测：** 不用校正值外推每轮全程均值，也不把它当成最终 Task Duration；完整跨 Stage 长链形状要等 Guard 修复后确认。

## 5. Prepare 瓶颈分析

### 5.1 512：首包和固定成本主导

512 只有 8 个 chunk，仅 8／28 核组执行 S1–S7。活跃核组首 notify 的中位时刻约 11.453 µs，整个 kernel 采集跨度 35.700 µs；单 pack 没有后续 pack 摊薄初始化和流水填充成本。[gdn-modified-steady-analysis](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-steady-analysis.txt&line=8&text=11.453 "citation")

首包 notify→KKT 中位仅 0.881 µs，而 KKT end→S3 EPI start 为 5.342 µs。这说明小 shape 首包的主要现象不是“KKT 发布很晚”，而是 KKT 之后的 dump／通知、AIV 可用性和后续单条数据链尚未完全展开。

### 5.2 8960：首包、首次交接和稳态必须分开

下表先取 28 核组的中位数。`S3 wait/core`、`EPI/core`、`S4 wait/core` 是一个核组内相关 task／AIV 窗口之和，**不是 pack 墙钟，也不能相加**。

| pack | notify→KKT | S3 wait/core | S3 EPI/core | S4 wait/core |
|---:|---:|---:|---:|---:|
| 0 首包 | 0.883 µs | 0.787 µs | 7.284 µs | 10.967 µs |
| 1 首次交接 | 6.460 µs | 10.267 µs | 3.856 µs | 4.088 µs |
| 2 | 1.697 µs | 0.711 µs | 3.851 µs | 4.396 µs |
| 3 稳态 | 2.294 µs | 1.728 µs | 3.808 µs | 4.133 µs |
| 4 稳态 | 2.176 µs | 1.698 µs | 3.784 µs | 4.155 µs |

逐 pack 原始统计见分析输出。[gdn-modified-steady-analysis](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-steady-analysis.txt&line=35&text=pack%20%7C%20notify_t "citation")

由此得到三个结论：

1. **首次跨 pack 交接有明显供给空泡。** pack 1 的 notify→KKT 为 6.460 µs，S3 等待为 10.267 µs；下一包 AIV 已接近 S3，但对应 KKT 尚未完成供给。
2. **不能把 pack 1 外推为所有稳态包。** pack 2–4 的 S3 等待回落至 0.711～1.728 µs，说明首次交接与稳态要分开优化和验收。
3. **稳态 KKT 产物路径仍有成本。** pack 2–4 的 KKT end→S3 EPI start 中位约 1.136～1.248 µs，其中包含 KKT dump、通知和 AIV 可用性；优化对象应是供给顺序和产物路径，而不是盲目提升 KKT MMAD。

### 5.3 稳态主要命名工作项

8960 每核组五包命名窗口合计的主要项为：

| 窗口 | 每核组合计 | 正确解读 |
|---|---:|---|
| S3B_EPI | 22.535 µs（均值 22.560） | 最大 Prepare 命名工作项；含 L 计算、叶块打包和 UB/L1 数据路径 |
| S3B_WAIT_KKT | 15.212 µs（均值 15.420） | 大部分集中于首次交接，稳态明显回落 |
| S4_WAIT_S3 | 13.898 µs（均值 13.921） | AIC 等 AIV S3 产物，反映 Stage3 位于关键路径 |
| S6_VB / S6_KBG | 9.869 / 10.124 µs | AIV 工作，但设计上与 S4／S5 重叠，不能全额计入关键路径 |
| S1B_GATE_BETA | 7.463 µs | 与 KKT 等阶段存在重叠 |
| S5_SOLVE / S5_DUMP | 3.504 / 2.992 µs | AIC 求逆及 A 的产物路径 |
| S7_W / S7_U / S7_DRAIN | 2.096 / 1.573 / 3.752 µs | W/U 计算和回写；与下一包 S1 有重叠 |
| S2_KKT | 1.631 µs（均值 1.622） | 纯 KKT 窗口很小，不能据此宣称 Cube bound |

这些窗口分布在 AIC、两个 AIV 和不同 pipe 上，彼此重叠；表格用于找大项，不用于计算利用率。[gdn-modified-inspect](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-inspect.txt&line=198&text=PER-GROUP%20WORK%20SUM "citation")

下一包首个已计时 S1 起点相对上一包 S7 W/U 终点的中位差为 **−1.966 µs**，证明“下一包 S1 覆盖上一包后段”已在采集里发生，不只是设计意图。[gdn-modified-steady-analysis](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-steady-analysis.txt&line=43&text=next%20pack%20first%20timed%20S1%20start%20-%20previous%20S7 "citation")

### 5.4 Prepare 的 bound 结论

- **首包／小 shape：启动、常驻初始化和单条 CV 数据链填充 bound。**
- **首次跨 pack：AIC 后段与下一包 KKT 启动之间的调度／供给 bound。**
- **稳态：AIV Stage3 计算与 UB↔L1 数据重排、S3→S4 交接共同 bound。** S6 和 S7 会影响交接，但已有重叠，不能凭窗口总和把它们各自独立算成全程瓶颈。
- **不是已证实的 KKT MMAD throughput bound，也没有证据宣称 DDR bound。**

## 6. FwdH：结构瓶颈与采样伪影必须分开

### 6.1 代码可确认的结构瓶颈

同一 head 的 chunk `c+1` 必须等 chunk `c` 的 H/state 就绪，不能简单把 140 个 chunks 分到 140 个核。当前只有 4 个 value heads，host 按 `N×HV` 分派，所以 `B=1` 时只有 4 个活跃核组；token 增多只延长每核组的串行链，不增加核组数。新统计同样识别出 `chunks/head=140、recurrence updates/head=139、active groups=4`。[gdn-modified-steady-analysis](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-steady-analysis.txt&line=48&text=chunks%2Fhead%3D140 "citation")

单 chunk 内的真实依赖为：

```text
上一 chunk H_READY
    ↓
C0 读取 W/H；C1 计算 P=W@H
    ↓ P_READY
V1 计算 V_new/right，写 V_new 和 right
    ↓ RIGHT_READY
C0 读取 right/K；C2 计算 D=K^T@right
    ↓ D_READY
V2 更新 FP32 state，写下一 chunk H
    ↓ H_READY / D_FREE
下一 chunk
```

W/K/U/g 的下一 chunk 预取可以藏到当前计算下，但 P、D、right、H 的生产消费依赖不能被普通预取消除。因此 FwdH 是 **parallelism + recurrence critical-path bound 的强候选**；至于单 head 内部 Cube、Vector、MTE 和同步各占多少，当前 timer 不能给出完整 pipe 完成时间，不能硬拆比例。

### 6.2 为什么原始 8960 跨度要剔除尾跳变

旧动态槽行为是：

```text
MAX_DYNAMIC_ITER = 64
前 63 次记录：写入槽 0～62
达到上限后：NextIter() 持续返回 63
后续事件：不断覆盖槽 63，而不是各自保留独立轮次
```

当前对齐源码已改为溢出时返回 `MAX_DYNAMIC_ITER`，让 Tik/Tok 的现有边界检查丢弃后续采样，不再复用最后一个槽。[AscendTimerV2_device](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/common/kernel_utils/gdn_timer/AscendTimerV2_device.hpp&line=432&text=%E6%BA%A2%E5%87%BA%E6%97%B6%E8%BF%94%E5%9B%9E%20MAX_DYNAMIC_ITER "citation")

新 8960 trace 的 H_C1／H_C2 等起点节奏中位数约 2.09 µs，但 62→63 的最大间隔约 159.8 µs。这个巨大跳变与“槽 63 留到最后一次调用”完全一致，说明跨 Stage 末尾事件并不属于同一轮，会拼出假倒挂、假长链和失真的 MAIN_BODY 关系。[gdn-modified-comparison](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-comparison.txt&line=52&text=H_C1_WH%20start%20cadence "citation")

因此整体构成从原始 FwdH 跨度中减去 159.800 µs，得到 144.133 µs；这是对本次已采集 JSON 的尾部伪影校正，不是把后续 139 轮递推直接删除。

前 63 槽在当前带 timer 条件下得到的单 chunk 起点中位节奏约 2.09 µs，只能作为**前缀样本**，不能外推为 139 次递推的全程均值，也不能证明旧报告中的“全程稳定递推”。

![FwdH 递推约束与 Timer Guard 修法](assets/gdn_profiling/gdn_fwdh_timer_guard.svg)

### 6.3 正确修法与验收

- 溢出后让现有 Tik/Tok Guard 丢弃后续采样，保留每个 Stage 首次记录的 0～63 槽。
- 槽 63 一旦记录，不再被后续调用覆盖；不同 Stage 的前缀轮次保持一致。
- **4 处 `PIPE_S` 不需要再改。** 本问题是动态采样槽覆盖，不是继续调整 PIPE_S 打点位置。
- 重新采集后先检查：不再出现槽 62→63 跳到 kernel 末尾的现象；再分析跨 Stage 依赖。
- 最终真实耗时以 `msprof op_summary Task Duration` 和优化前后 `--no-timer` A/B 为准，不以 timer 跨度替代。

### 6.4 512 FwdH 能说明什么

512 每 head 只有 8 chunks，新统计确认所有动态迭代均在容量内，没有槽溢出；采集跨度为 26.125 µs，活跃核组 MAIN_BODY 中位约 24.134 µs。前 63 槽内的 C1 起点中位节奏约 2.108 µs。[gdn-modified-steady-analysis](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-steady-analysis.txt&line=20&text=All%20dynamic%20iterations "citation")

这可用来分析小 shape 的短递推和流水填充，但它仍是带 timer 数据；与旧打点或其他分支的差值不能直接解释为真实性能回退。

## 7. FwdO 瓶颈分析

8960 下 28 个核组全部活跃，每核组 5 个 group tasks；主体中位约 26.811 µs，整体采集跨度 28.544 µs。首个 QK 的相邻 group 起点中位间隔约 4.228 µs，单 group 可见区间中位约 3.914 µs。[gdn-modified-steady-analysis](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-steady-analysis.txt&line=57&text=rank0_fwd_o "citation")

每核组五组任务的命名窗口合计：

| 窗口 | 每核组合计 | 含义 |
|---|---:|---|
| O_V2_MASK | 6.730 µs（均值 6.796） | 最大命名 AIV 阶段；tril、衰减、A′及 L1 写入 |
| O_V3_OUTPUT | 3.779 µs（均值 3.777） | 输出融合和 GM 写回路径 |
| O_V1_GATE | 2.757 µs（均值 2.791） | gate_o/gate_A |
| O_C3_ATTNV | 2.326 µs（均值 2.312） | A′@V_new 及完成锚点 |
| O_C1A_QK / O_C1B_QH | 1.431 / 1.258 µs | QK/QH 矩阵路径 |
| MTE3 | 6.096 µs（均值 6.331） | 多个本地等待窗口之和，不是 MTE3 硬件忙碌率 |

关键交接的中位间隔为：QK end→Mask start 1.268 µs（中间还含 QH 路径），Mask end→AttnV start 0.463 µs，AttnV end→Output start 0.705 µs。[gdn-modified-inspect](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/timer-analysis/gdn-modified-inspect.txt&line=154&text=PER-GROUP%20WORK%20SUM "citation")

局部优化应优先看 Mask 和 CV 产物交接；但 FwdO 在大 shape 校正总构成中仅约 11.5%，所以它是后续局部优化，不是当前第一优先级。512 下只有 8／28 核组活跃、每核组仅一个 group task，11.359 µs 中包含启动和单链填充成本。

## 8. 优化方案与优先级

### 8.1 预期收益区间：先统一口径

下表是**方案评审用的工程估算，不是实测收益**。正数表示**耗时下降比例**，不是吞吐提升比例；“整体”指当前 8960 case 的三阶段校正跨度之和，按 `整体降幅≈阶段占比×该阶段耗时降幅` 折算，不代表已经测得真实端到端收益。P5 单独按 512 case 折算。

估算只适用于本文 `B=1、HK=2、HV=4` 的采集 shape；**不能直接外推到 HV=32**。各项之间存在依赖和重叠，区间不能相加；组合方案必须重新做无 timer A/B。P1 的 2／4 列块理想并行上界分别为 FwdH 耗时下降 50%／75%，工程区间已为小 MMAD、复制读取和同步开销留出折损，但仍有较大不确定性。

| 优化点 | 局部 Task Duration 降幅预估 | 三阶段合计耗时降幅预估 | 估算边界 |
|---|---:|---:|---|
| P0 Timer Guard 重采 | 0% | 0% | 测量修正，不是算子加速 |
| P1 FwdH V 切 2 列块 | 25%～40%（FwdH） | 14.6%～23.3% | 理想上界 50%；依赖跨核扩展净收益 |
| P1 FwdH V 切 4 列块 | 40%～60%（FwdH） | 23.3%～34.9% | 理想上界 75%；小 MMAD 和 W/K 复制风险更高 |
| P1 消除 right GM 往返 | 5%～12%（FwdH） | 2.9%～7.0% | 新产物路径尚未验证，不能按字节数直接换算 |
| P1 增广矩阵直接产 V_new | 暂不估正收益 | 暂不估正收益 | 研究项，额外 MMAD 可能抵消收益 |
| P2 Stage3 严格下三角 | 5%～12%（Prepare） | 1.5%～3.6% | 只裁剪有效消费者所需区域，不是全 Stage3 减半 |
| P2 减少 Stage3 遍历／重排 | 5%～15%（Prepare） | 1.5%～4.5% | 与上一项重叠，不能累加 |
| P2 合并 vb／kbg 公共流程 | 0%～4%（Prepare） | 0%～1.2% | S6 已有重叠，命名窗口下降未必等于墙钟下降 |
| P2 修复首次跨 pack 交接 | 2%～6%（Prepare） | 0.6%～1.8% | 主要是一次性供给空泡，不能按五包重复计算 |
| P2 深化 KKT 与 S5／S7 重叠 | 2%～8%（Prepare） | 0.6%～2.4% | 以资源生命期可闭合为前提，与交接优化重叠 |
| P2 缩短 KKT→S3 产物路径 | 2%～6%（Prepare） | 0.6%～1.8% | 当前可见间隔约 1.1～1.2 µs／pack，不能全当可消除墙钟 |
| P2 衰减折叠进 Cube | 暂不估正收益 | 暂不估正收益 | 独立精度项目，数值与额外 Cube 工作未验证 |
| P3 S5 direct L0C→L1 | 2%～6%（Prepare） | 0.6%～1.8% | 目标是去掉 GM 读回，同时保留必须的 GM A 输出 |
| P3 S7 复用 A 的 L0A 装载 | 0%～3%（Prepare） | 0%～0.9% | S7 已有跨包重叠，整体收益可能接近零 |
| P3 S7 跨 task 流水 | 2%～6%（Prepare） | 0.6%～2.4% | 与 P2 交接／重叠项不能累加 |
| P4 FwdO Mask 与 A′ 重排 | 5%～12%（FwdO） | 0.6%～1.4% | Mask 是最大命名 AIV 项，但并非全部位于关键路径 |
| P4 V_new 更深预取 | 0%～5%（FwdO） | 0%～0.6% | 已领先一个 head，需防止 L1 与首包成本反噬 |
| P4 剩余组边界解耦 | 0%～5%（FwdO） | 0%～0.6% | 只针对尚未消除的所有权／等待，不重复已有前移 |
| P4 QK／QH 复用 Q 的 L0A | 0%～3%（FwdO） | 0%～0.3% | 矩阵窗口较小，主要减少搬运而非 MMAD |
| P5 512 Init／S0／首包压缩 | 5%～15%（512 Prepare） | 2.4%～7.3%（512 合计） | 只针对小 shape；FwdH V 切列收益另见 P1 |

这些区间的用途是安排实验顺序：**P1 是高收益、高风险结构项目；P2／P3 是中等收益、可逐步验证的局部项目；P4 的整体上限受 FwdO 占比约束。** 任一项目实现后，都应以实测值替换本表估算，并记录实际 Task Duration，而不是继续沿用预估区间。

### P0：用修复后的 Timer Guard 重采，建立可信测量基线

这是后续性能分析的前置条件，不属于算子数学或流水优化。用同一输入、同一环境重新采集 512／8960，并保留无 timer 对照。只有采样一致性和无 timer 性能都稳定，才用 trace 比较优化前后的关键路径。

### P1：FwdH 按 value 维切逻辑列，突破 4 核组限制

**依据：** 它直接针对代码可确认的“只有 4 个核组 + 长递推链”；剔除尾跳变后 FwdH 仍占校正总和的 58.2%，但最终收益不依赖这个带 timer 比例，必须由无 timer A/B 验证。

对固定 head，将 `V=128` 切成独立列块 `Vs`：

```text
P[:,Vs]   = W @ H[:,Vs]
Vnew[:,Vs]= U[:,Vs] - P[:,Vs]
D[:,Vs]   = K^T @ right[:,Vs]
R[:,Vs]   = decay · R[:,Vs] + D[:,Vs]
```

列块之间没有跨 V 归约，每个列块的 chunk 递推可独立保持在同一核组；W/K 和 gate 可共享或重复读取，H/V_new/final_state 写回列区间互不重叠。**不能把同一 head 的不同 chunks 直接并行，但可以把同一 head 的不同 V 列块并行。**

| 候选 | 活跃核组数（当前 B×HV） | 优点 | 主要代价 |
|---|---:|---|---|
| 2 列块 × 64 | 8 | 改动和复制流量相对温和，Cube N 仍有 64 | 并行度只翻倍 |
| 4 列块 × 32 | 16 | 更接近可用核数，缩短单核递推工作量 | W/K 重复读取更多，Cube N=32 效率和同步次数需实测 |

这是结构变更，需先完成方案与资源评审；同步修改 host work 映射、逻辑 V offset、L0/L1/UB tile、AIV VF 宽度、workspace core 映射和 H/final_state 布局测试。预期是缩短关键路径，**8.1 节区间只是实验目标，不是加速承诺**；若复制 W/K、小 MMAD 和更多同步抵消收益，就保留原方案。

**P1 补充：评估 right 的 GM 往返，而不是再加普通预取。** 当前 AIV 生成的 right 必须经 GM workspace，再由 AIC 读入 L1；设计文档明确禁止现有 UB→L1 直搬路径。[design](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/docs/design.md&line=78&text=right "citation") 可以在 V 切列方案稳定后，评估新的 right L1 槽和 READY/FREE 协议，目标是消除内部中转的 GM 写读；这需要重新证明 L1 容量、AIV→AIC 可见性和 WAR 生命周期，不能只替换一条 DataCopy。`[W|U]@[H;-I]` 让 Cube 直接产生 V_new 的增广矩阵方案也可作为研究项，但会增加 L0B／MMAD 工作并改变数据路径，需在 right 路径测量后单独评审。

**不把“同核两个 head 交错”直接列为当前 case 的新方案。** 当前 `B=1、HV=4` 本来就是 4 核组各一个 head，没有同核第二个 head 可用来填等待；而现有多 head 路径已经按 chunk 在 round 内处理多个 head，并使用 L0 ping/pong。该建议只适用于需要复查的高 HV／多 head-per-core case，不能套用 `zq` 文档中 HV=32 的收益数字。

### P2：Prepare 先优化 Stage3 数据链和首次跨 pack 交接

1. **Stage3 只计算下游需要的严格下三角。** 当前 `MulKktGateVF` 对完整 64×64 矩阵执行 load／mul／store，但下游只消费严格下三角。可按行缩窄有效 mask、跳过无效上半区，并同步裁剪后续打包遍历；需要验证向量对齐和短行效率，不能简单假设节省恰好一半。[chunk_gated_delta_rule_fwd_prepare_vf](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare_vf.h&line=520&text=for%20%28uint16_t%20i "citation")
2. **减少 Stage3 中间数据遍历和重排。** 继续评估合并向量遍历、减少 LFull/LPacked/ResVcs 之间重复读写，以及更直接的叶块布局生成。现有 UB/L1 已接近容量上限，不能靠简单增加大 buffer 解决。
3. **合并 vb／kbg 的公共处理流程。** 当前 S6 分别执行 vb 和 kbg 的 GM→UB、行缩放与 L1 上传。可评估合并循环、地址／beta 公共准备和同步，并连续提交两个 L1 输出；两者输入分别是 v 和 k_hat，合并不必然减少输入字节，收益应来自减少公共开销和 UB／同步往返。[chunk_gated_delta_rule_fwd_prepare](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h&line=776&text=S6_VB "citation")
4. **首次交接与稳态分开调。** 临时细分 KKT dump／notify，确认 pack 1 的 6.460 µs notify→KKT 有多少来自上一包 AIC 后段、资源释放和下一包 S1 放行，不要先改 KKT 数学实现。
5. **资源生命期允许时，更深度交织下一包 KKT 与上一包 S5／S7。** 必须逐项证明 L0A/L0B/L0C、k_hat/NegL、event bank 不冲突，不能只移动 notify 或删除等待。
6. **缩短稳态 KKT end→S3 EPI 的约 1.1～1.2 µs 产物路径。** 重点看 L0C→AIV UB dump 和通知，而非 MMAD 本身。

衰减折叠为行／列缩放、再由 Cube 直接产出部分 L 的方案可以改变计算图，但会引入额外 BF16 舍入和指数数值范围问题；先完成上述不改变数学语义的优化，再把衰减折叠作为独立精度项目评审，不沿用其他 case 的收益估计。

### P3：Prepare 的 S5/S7 减少重复搬运并加强 task 间流水

- **S5 full-chunk A 路径：** 当前 `outputA!=0 && M==64` 走 `L0C→GM A→L1 A`。可评估在保留 GM 输出的同时使用 direct `L0C→L1 NZ`，消除 GM 读回；必须重新闭合 Fixpipe/MTE1 flag 和 L0C 生命周期。
- **S7 复用 A：** W 和 U 都以同一份 A 为左矩阵，可评估减少重复 L1→L0A 装载。
- **S7 跨 task 流水：** 可评估让下一 task 的独立 MMAD／装载覆盖上一 task 写回，先核对两个 L0 bank 和 AIV UB retained slot 的所有权。

这些优化主要帮助 Prepare AIC 后段，也可能间接缩短下一包 KKT 的供给延迟。

### P4：FwdO 优化 Mask、V_new 预取和组边界，放在后续

1. 优化 `Stage3Gate64VF` 的向量遍历和 A′ 的 UB→L1 重排，减少 V2 与 AttnV 之间空档。
2. 在现有“V_new 预取领先一个 head”基础上，评估提前到两个 head 或跨 group 预取。必须先证明新增 L1 槽、MTE2 顺序和 V_new 生命周期成立，并用组尾等待是否下降验收。
3. 继续拆解组边界，但不重复做已实现的传输前移：当前下一组独立 Q／K／H 的 GM→L1 已在等待上一组 Stage5 之前下发，剩余候选是进一步解耦 L0／UB 所有权，而不是简单移动同一个 wait。[chunk_fwd_o_cube](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/.kimi/feat-src/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_cube.h&line=373&text=Let%20the%20next%20task "citation")
4. QK 与 QH 都使用 Q，可评估 L0A 中 Q 的复用，减少重复 L1→L0 搬运。

“每个 HK 只算一次 QK”已经是当前实现，不再列为新优化：同一 HK 的 GVA value heads 共享 Q／K 装载和 QK 结果。

### P5：小 shape 单独处理启动和首包

512 的 Prepare 和 FwdO 都没有填满 28 核组，FwdH 也只有 4 核组。此 case 更适合缩减 Init/S0 和首包数据链上的非必要工作、用 FwdH V 列切分增加有效并行，并避免为过深预取拉长首包。不能用 8960 的优化设想直接承诺 512 同比例受益。

## 9. 不建议作为优化方案的做法

- **不做逐核 ANCHOR 对齐来提升“图的正确性”。** 对齐只是视图变换，不改变 kernel，也不能修复槽覆盖。
- **不再为本问题修改 4 处 `PIPE_S`。** 当前已定位到动态槽溢出后的覆盖行为。
- **不删除 ready/free 等待来压缩空泡。** 这些等待保护 P/D/right/H、L0/L1/UB 生命周期。
- **不把所有命名窗口相加后宣称利用率。** 它们跨核、跨 pipe 且定义不同。
- **不优先继续做普通 W/K 预取或 FP32 state 常驻。** FwdH 当前路径已经具备。
- **不根据三份独立切片推断 host launch gap，或据此优先做三 kernel 融合。** 是否需要融合必须由统一绝对时间轴或端到端性能数据另行证明。

## 10. 验证口径

每项实验只改变一个主要变量，使用相同输入、环境、预热和采样次数：

1. **采样正确性先过：** Guard 修复后检查每 Stage 只保留首次 0～63 槽，不再出现末尾槽跨到 kernel 末尾的跳变；跨 Stage 只比较同一轮次。
2. **精度先过：** 按各算子 CPU 标杆和 ATK 规则验证。V 列切分至少覆盖 2／4 列块、尾 chunk、varlen、initial/final state、两种 state layout 以及支持的 GVA 比例；Prepare 同步重排覆盖 pack 边界和 tail。
3. **统一性能验收：** 按仓库流程使用 `msprof op_summary` 的 `Task Duration(us)`，固定列出 Prepare／FwdH／FwdO 并采用同一聚合口径。Timer 用于定位，不替代性能验收。[03-方案设计](kimi-cite://md-line?path=F%3A/WorkSpace/Code/GDN/flash-linear-attention-npu/docs/agents/03-%E6%96%B9%E6%A1%88%E8%AE%BE%E8%AE%A1.md&line=27&text=Task%20Duration "citation")
4. **保留 `--no-timer` A/B：** 同时确认打点本身没有掩盖或制造收益；最终比较优化前后无 timer 的同口径性能。
5. **观察指标与优化假设对应：**
   - FwdH V 切列：活跃核组数、每核递推工作量、FwdH Task Duration；检查复制 W/K 后是否仍有净收益。
   - FwdH right 路径：right GM 写读是否消除、RIGHT_READY／FREE 等待、L1 占用和 FwdH Task Duration。
   - Prepare Stage3／S6：S3B_EPI、S6_VB、S6_KBG 与 S7_WAIT_S6；确认下三角裁剪没有让短行向量效率反而下降。
   - Prepare 交接：pack 1 notify→KKT、S3B_WAIT_KKT、稳态 KKT end→EPI；检查是否只是把空泡移到 S4/S7。
   - Prepare S5/S7：S5_DUMP、S7 W/U、下一包 S1/KKT 起点；检查 GM 读回是否真正消除。
   - FwdO：O_V2_MASK、Mask→AttnV、组尾等待、V_new MTE2 供给和 FwdO Task Duration；同时看整体收益是否值得保留复杂度。
6. **512 与 8960 都回归：** 防止大 shape 收益以小 shape 启动成本、tail 或负载不均衡恶化为代价。

## 11. 最终结论

当前证据支持的路线是：

```text
测量前提：Guard 溢出后丢弃采样，保留首次 0～63 槽；PIPE_S 不再改
第一候选：FwdH 按 V 逻辑列切分，增加递推并行核组
FwdH 后续：评估 right GM 往返和新的产物路径
第二候选：Prepare 严格下三角、Stage3 数据链和首次跨 pack 交接
第三候选：合并 vb/kbg 公共流程，减少 S5/S7 重复搬运
后续局部：FwdO Mask、V_new 深预取与剩余组边界；小 shape 单独优化首包
```

一句话概括：**512 的采集表观瓶颈是 Prepare；8960 剔除 159.800 µs 尾跳变后，FwdH 校正为 144.133 µs、占 58.2%，仍是首要结构优化候选。Prepare 应优化 Stage3 与交接，现有证据不支持“KKT MMAD 算力已打满”；校正值仍需 Guard 修复重采和无 timer A/B 最终确认。**
