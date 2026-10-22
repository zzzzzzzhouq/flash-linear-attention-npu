# GDN 计时打点 × 逻辑流水图 对照表

> 对照对象
> - 打点分支：`flash-linear-attention-npu-profiling` @ `codex/gdn-profiling`（66857968）
> - 流水图：`zq/gdn_fla_npu_pipeline.svg` / `gdn_fla_npu_pipeline.png`，说明见 `zq/GDN原理梳理_算法流程与硬件流水.md` §4
> - 路径：A5 三算子路径（`UsePreparePath = true`：BF16 + K=128 + V=128 + chunk=64 + GVA≤4），即两条性能 case 实际走的路径
> - 代码路径前缀：`fla/ops/ascendc/gdn/chunk_gdn_fwd/`
>
> 读表约定
> - **窗口（issue window）**：`TikNoBarrier/TokNoBarrier`，量的是下发时刻，不代表完成时延；
> - **等待（wait）**：只包 `CrossCoreWaitFlag / WaitFlag` 的返回；
> - 所有窗口可重叠，**不能相加**；总量只看 `KERNEL_TIMING`；
> - 每核每阶段最多 32 个槽（`MAX_DYNAMIC_ITER = 32`），超出截断；
> - 跨核绝对时间默认不可比（见 §5 时钟校验）。

---

## 0. 一句话结论

打点覆盖了三算子路径的绝大多数阶段，但有 **13 处与逻辑流程图/代码实际不一致**：
其中 **5 处是图错**（S1→S2 放行点、S3 前"等 KKT"、S4 prefill、S7 与 S6 关系、FwdO 的 GM 往返），
**6 处是打点语义/命名问题**（窗口含等待、W/U 不对称、等待合并、同名两套实现），
**5 处是打点缺口**（w/u 回写、S0、FwdH 预取、FwdO 组边界等待、S6 的 V/K 拆分）。

---

## 1. 打点全景：名字 → 位置 → 包什么 → 对应步骤

### 1.1 固定项（6 个，每核一份）

| 名字 | 核 | 含义 |
|---|---|---|
| `KERNEL_TIMING` | 全部 | `Init` → `Finish` 全 kernel 墙钟（唯一带 barrier，保证流水排空） |
| `MAIN_INIT` | 全部 | `Init` → `InitDone` 初始化窗口 |
| `MAIN_BODY` | 全部 | `InitDone` → kernel 结束 |
| `PHASE6_FRONT_DONE` | PHASE6 | front-batch 结束的瞬时点 |
| `PHASE6_H_DISPATCH_DONE` | PHASE6 | H 下发完成的瞬时点 |
| `PHASE6_HO_WAIT` | PHASE6 | H→O 串行模式的 SyncAll 等待 |

### 1.2 动态项（28 个）

#### Prepare（三算子路径的 Prepare kernel）

| 名字 | 核 | 打点位置（文件:行） | 包裹范围 | 对应步骤 |
|---|---|---|---|---|
| `PREPARE_NORM_GATE` | AIV | `chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h:422` | 整个 `Stage1_OneTask`：`Stage1_PadTailIfNeeded` + `Stage1_UploadK`（含 `NotifyAicStage1Done`）+ `Stage1_GateAndBeta` | **S1** |
| `KKT_MM` | AIC | 同文件 `:481`（`Stage2_AicOne` 首行） | `WaitFlag<M_MTE1>` + `MatmulToL0C` + `M_FIX` 等待 + `DumpKktToUb` | **S2** |
| `PREPARE_KKT_GATE` | AIV | 同文件 `:506`（`Stage3_PrepareGate`） | `GateLowerLVF`（exp 下三角）+ `I_vcs` 拷贝，**在 `WaitCubeKktDone` 之前** | **S3a**（gate 预计算） |
| `KKT_EPILOGUE` | AIV | 同文件 `:518`（`Stage3_AivOne` 首行） | **首句 `WaitCubeKktDone`** + `L = KKT⊙G` + 叶打包 + `-L`/leaf 上传 | **S3b** |
| `PREPARE_SOLVE_PREFILL` | AIC | 同文件 `:554`（`Stage4_PrefillI`） | `I@I` 预填充（在 `WaitAivStage3Done(0)` **之前**下发） | **S4-pre** |
| `PREPARE_SOLVE_ISSUE` | AIC | 同文件 `:566`（`Stage4_AccIssue`） | `Y = I + LeafLeft·(−L)` 的 MMAD 下发 | **S4-issue** |
| `PREPARE_SOLVE_DUMP` | AIC | 同文件 `:578`（`Stage4_Dump`）、`:626`（`Stage5_Dump`） | L0C → L1/UB 的 dump | **S4-dump / S5-dump** |
| `SOLVE` | AIC/AIV | 同文件 `:598`（`Stage5_Compute`） | `(I+L)⁻¹` 求逆 | **S5** |
| `WU_PREPARE` | AIV | 同文件 `:656`（`Stage6_AivOne`） | vb = v·β（GM→UB→L1）后接 kbg = k̂·β·e^g（GM→UB→L1），**一个窗口含两段** | **S6** |
| `W_MM` | AIC | 同文件 `:722` | `WuMatmulToL0C(A, kbg)`；`WuMatmulToL0C` 内部含 LoadData + `MTE1_M` + `FIX_M` 等待 | **S7-W** |
| `U_MM` | AIC | 同文件 `:736`、`:754` | `WuMatmulToL0C(A, vb)`；V=256 时第二个半块走 `:754` | **S7-U** |
| `FRONT_WAIT` | AIC/AIV | 同文件 `:324/345/365`（`WaitAicStage4Done`）、`:823/:860`（`WaitAicStage7Done`）、`:887`（`WaitAivStage1Done`）、`:898/:904`（`WaitAivStage3Done`）、`:924`（`WaitAivStage6Done`） | **5 种不同等待共用一个名字** | 包内/包间同步 |

> 没有打点的两处：`Stage0_GenerateResidentAux`（S0，只在 `MAIN_INIT` 里）和 `DrainPackWuToGm`（w/u 回写）。

#### FwdH（`chunk_fwd_h/op_kernel/arch35/`）

| 名字 | 核 | 打点位置 | 包裹范围 | 对应步骤 |
|---|---|---|---|---|
| `H_WH` | AIC | `chunk_fwd_h_cube.h:375` | 函数入口即开窗；内部先 `WaitFlag(WDoneEvent/HRightDoneEvent/L0AFree/L0BFree)`，再 L1→L0 拷贝 + MMAD | **C1**（P = w·S） |
| `H_KV` | AIC | `chunk_fwd_h_cube.h:563` | 同样含 `WaitFlag(M_MTE1 L0AFree/L0BFree)` + `MTE2_MTE1` 等待，再 MMAD | **C2**（h_work = kᵀ·v_new） |
| `H_VNEW` | AIV | `chunk_fwd_h_vec.h:647` | `epilogueGDNFwdHVnew` | **V1**（v_new） |
| `H_STATE_UPDATE` | AIV | `chunk_fwd_h_vec.h:873` | `epilogueGDNFwdHUpdate` | **V2**（状态更新） |
| `H_WAIT` | AIC/AIV | `chunk_fwd_h_cube.h:355/417/425/433/525/628/636/644/786/792/803/809/819/824`；`chunk_fwd_h_vec.h:673/879/1143/1180/1200/1217/1233` | ready / free / DIRECT_UB_FREE / round ACK 全部同名 | FwdH 内部同步 |

#### FwdO（`chunk_fwd_o/op_kernel/arch35/`）

| 名字 | 核 | 打点位置 | 包裹范围 | 对应步骤 |
|---|---|---|---|---|
| `O_GATE` | AIV | `chunk_fwd_o_vector.h:464`（`ComputeStage1Gate`） | cast + `PadGateInput64VF` + gateA/gateO 预计算（per chunk, per v-head） | **V1** |
| `O_QK` | AIC | `chunk_fwd_o_cube.h:442`（另有 `arch35/gemm/kernel/gdn_fwd_o_kernel.hpp:321`） | `blockMmadQK`：attn = q·kᵀ | **C1-a** |
| `O_QH` | AIC | `chunk_fwd_o_cube.h:508`（另有 `gemm/kernel/...hpp:434/440/448/454`） | `blockMmadQH128/256`（含 `ToUB` 变体）：O_s = q·h | **C1-b** |
| `O_MASK` | AIV | `chunk_fwd_o_vector.h:339/350`（另有 `gemm/kernel/...hpp:603`） | `Stage3Gate64VF`：tril + 衰减 → A′；O_s′ | **V2** |
| `O_ATTN_V` | AIC | `chunk_fwd_o_cube.h:279/298`（另有 `gemm/kernel/...hpp:499/506/515/522`） | `TileMmad` / `blockMmadAttenVNEW`：O_l = A′·v_new | **C3** |
| `O_OUTPUT` | AIV | `chunk_fwd_o_vector.h:410`（另有 `gemm/kernel/...hpp:639/650/662`） | `Stage5Fuse64VF`：o = scale·(O_s′+O_l) | **V3** |
| `O_WAIT` | AIC/AIV | `chunk_fwd_o_cube.h:170/182/195/386`；`chunk_fwd_o_vector.h:333/380/394` | ready / free / **组边界**（首个 C1 前的 `stage5GroupPending` 等待）全部同名 | FwdO 内部同步 |

#### 融合 Phase6 专用（本路径不执行，列全供参考）

| 名字 | 说明 |
|---|---|
| `CUMSUM` | AIV：chunk 内 gate cumsum（`RunPhase6Cumsum`） |
| `KKT_MM` / `KKT_EPILOGUE` | 融合核的 KKT 及其 gate 下三角 |
| `SOLVE` / `SOLVE_WAIT` | 求逆 / solve→WU 跨核等待 |
| `WU_PREPARE` / `WU_WAIT` / `W_MM` / `U_MM` | recompute：AIV 产 vb/kbg（GM ring）、AIC 逐 task **先 U 后 W** |
| `H_*` / `O_*` | 融合核内的 H/O 段 |
| `PHASE_WAIT` | 相位间 `SyncAll` |

---

## 2. 不一致清单（打点名 | 打点位置 | 图上怎么画 | 代码实际与正确写法 | 类型）

### 2.1 Prepare

| # | 打点名 | 打点位置 | 图上怎么画的 | 代码实际 / 正确应该是什么 | 类型 |
|---|---|---|---|---|---|
| P1 | `PREPARE_NORM_GATE` | `prepare.h:422`，包住 `PadTail + UploadK + GateAndBeta` | 图① AIV 行：`S0＋S1` 一整块；「k̂ 落 L1」箭头画在 S1 右边界，S2 在 S1 结束之后才开始 | `NotifyAicStage1Done` 在 `Stage1_UploadK` 内（k̂ 落 L1 即发），`Stage1_GateAndBeta` 还在其后。**正确**：S1 拆成 **S1a（pad + k̂ 上传，放行点）** 与 **S1b（gate/β）**；S2 从 S1a 末尾开始，与 S1b 重叠 | 图错 |
| P2 | `PREPARE_KKT_GATE` | `prepare.h:506`，`GateLowerLVF` + `I_vcs` 拷贝 | 图① AIV 行：S3 前有一块灰色「等 KKT」 | 这段是**可并行**的 gate 预计算，注释明说 "before WaitCubeKktDone so Exp and the MTE3 copy overlap Cube kkt"。**正确**：灰块改为实块 **S3a（gate 预计算，与 S2 并行）** | 图错 |
| P3 | `KKT_EPILOGUE` | `prepare.h:518`，函数首句 `WaitCubeKktDone` | 图① 的 S3 画成纯计算块 | 窗口 = 等待 + `L=KKT⊙G` + 打包 + 上传。**正确**：拆成 **`KKT_WAIT` + `KKT_EPILOGUE`**，否则时长不可解释 | 打点错 |
| P4 | `PREPARE_SOLVE_PREFILL` | `prepare.h:554`，在 `WaitAivStage3Done(0)`（`:898`）之前调用 | 图① AIC 行：`等 −L` 之后才进 S4 | prefill 在等 S3 之前就下发，与 AIV 的 S3b 重叠。**正确**：图上把 prefill 画进 S3b 的时间窗，`等 −L` 只挡 `AccIssue` | 图粗 |
| P5 | `PREPARE_SOLVE_ISSUE` / `PREPARE_SOLVE_DUMP` | `prepare.h:566` / `:578`（+`:626`） | 图① AIC 行：S4 是一整块 | 代码是 `AccIssue(t)` + `Dump(t-1)` 交错流水。**正确**：图上画成 issue/dump 两级（打点已分开，图跟上） | 图粗 |
| P6 | `W_MM` / `U_MM` + `FRONT_WAIT(:924)` | `prepare.h:722` / `:736,754`；AIC 循环内 `WaitAivStage6Done(t)` | 图① AIC 行：S7 画在 S6 全部结束之后 | 代码是 `for t: WaitAivStage6Done(t); Stage7(t)`，**逐 task**。**正确**：S7(t) 与 S6(t+1) 重叠，画成阶梯 | 图粗 |
| P7 | `W_MM` / `U_MM` | `prepare.h:722` vs `:736` | 图① S7 是一块对称的「W=A·kbg U=A·vb」 | `W_MM` 窗口含 LoadData + `MTE1_M` + `FIX_M`（`WuMatmulToL0C` 内部），`U_MM` 只有下发；实测 W_MM median 是 U_MM 的 35 倍。**正确**：改名 **`W_ISSUE` / `U_ISSUE`**，禁止直接比大小 | 打点错 |
| P8 | （无）`DrainPackWuToGm` | 函数体（`prepare.h:760-773`）无任何 `GDN_TIMER_SCOPE` | 图① AIV 行虚线区有实块「回写 w/u（包N）」，并用点线箭头标注「W/U 留 UB，S3′ 后排空」 | 回写是真实动作（两次 `CopyUbToGmElems`），但 **profile 里完全没有**。**正确**：加 **`WU_DRAIN`** scope | 打点缺口 |
| P9 | （无）`Stage0_GenerateResidentAux` | 无 scope，只被 `MAIN_INIT` 覆盖 | 图① 写「S0＋S1」 | S0 是常驻辅助生成（I_vcs/索引/清零）。**正确**：加 **`PREPARE_RESIDENT_AUX`**，或图上标注"S0 未单独计时" | 打点缺口 |
| P10 | `FRONT_WAIT` | 5 种等待同名：`WaitAicStage4Done`(:324/345/365)、`WaitAicStage7Done`(:823/860)、`WaitAivStage1Done`(:887)、`WaitAivStage3Done`(:898/904)、`WaitAivStage6Done`(:924) | 图① 分别画了「k̂ 就绪」「−L 就绪」「vb/kbg 就绪」「L1 槽释放」等不同依赖 | 打点只有一个名字，拆不开。**正确**：按对象拆名 **`WAIT_S4_FREE` / `WAIT_S1_DONE` / `WAIT_S3_DONE` / `WAIT_S6_DONE` / `WAIT_S7_DONE`** | 打点错 |
| P11 | `KKT_MM` | `prepare.h:481`；`ProcessAic` 里 `if (!OwnsHk(...)) continue`（`:883`） | 图① 的 S2 按 task 画（每 task 一个块） | KKT 按 **(chunk, HK)** 计算一次，`NotifyKktSiblings` 一次唤醒 HRatio 个 task；G=2 时一包只有 2 个 KKT。**正确**：图改为"每 HK 一个 KKT、一对多唤醒"；计数按 HK 解释 | 图粗 |
| P12 | `FRONT_WAIT`（tail） | `prepare.h:823/:860`（`WaitAicStage7Done`） | 图① 只在小字里写了「S1′ 的 L1 槽等 S4 释放」 | tail 包还会等上一包 S7 完成（`NotifyAivStage7Done`）。**正确**：图上补「tail 包等 S7 收尾」这条约束 | 图粗 |
| P13 | `WU_PREPARE` | `prepare.h:656`，一个窗口内先 V 后 K | 图① S6 是一块「vb=v·β  kbg=k̂·β·e^g」 | 两段独立搬运/缩放，但只打了一个窗。**正确**：拆 **`WU_PREP_V` / `WU_PREP_K`**，否则"让 K 与 V 重叠"的优化没有基线 | 打点缺口 |

### 2.2 FwdH

| # | 打点名 | 打点位置 | 图上怎么画的 | 代码实际 / 正确应该是什么 | 类型 |
|---|---|---|---|---|---|
| H1 | `H_WH` / `H_KV` | `chunk_fwd_h_cube.h:375` / `:563`，开窗在函数入口 | 图② AIC 行：C1/C2 画成纯矩阵乘块 | 窗口内先 `WaitFlag(WDoneEvent / HRightDoneEvent / L0AFree / L0BFree)`，再 L1→L0 拷贝、MMAD。**正确**：拆 **`H_WH_MM` / `H_WH_WAIT`**（`H_KV` 同理），或明确标注"含等待" | 打点错 |
| H2 | （无）预取 | 无 scope | 图② 在 C2 上方画了「MTE2 预取 W/K(c+1)」条带 | 预取（W/K lookahead）没有任何计时项。**正确**：加 **`H_PREFETCH`**（W、K 两个搬入点） | 打点缺口 |
| H3 | `H_WH` / `H_VNEW` 的 iter | 同上 | 图② 画出 h0/h1 双缓冲 | 没有 head/slot 维度，只能靠 iteration 序号反推。**正确**：timer args 里带 head/chunk/slot 标识 | 打点缺口 |
| H4 | `H_WAIT` | 24 处（`cube.h:355/417/...`、`vec.h:673/879/...`） | 图② 区分了「等 P」「等 v_update」「等 H flag」 | 打点全部同名。**正确**：拆 **`H_WAIT_READY` / `H_WAIT_FREE` / `H_WAIT_ACK`** | 打点错 |

### 2.3 FwdO

| # | 打点名 | 打点位置 | 图上怎么画的 | 代码实际 / 正确应该是什么 | 类型 |
|---|---|---|---|---|---|
| O1 | （无）接力通道 | — | 图③ 注释：「每次 C↔V 接力都过 GM workspace ＋ flag（A5：Fixpipe 直写 AIV UB，省一次 GM 往返）」 | A5 实际是 `blockMmadQH128ToUB` / `blockMmadAttenVNEW128ToUB`，**Fixpipe 直写 UB，没有 GM 往返**。**正确**：删掉 GM 说法，改为"Fixpipe→UB + flag" | 图错 |
| O2 | `O_QK` / `O_QH` | `chunk_fwd_o_cube.h:442` / `:508` | 图③ AIC 行：C1 画成一个「双矩阵乘」块 | 两段独立 matmul，QH 还有 `ToUB` 变体。**正确**：画成 QK / QH 两个子块 | 图粗 |
| O3 | `O_WAIT` | `chunk_fwd_o_cube.h:170/182/195/386`、`chunk_fwd_o_vector.h:333/380/394` | 图③ AIC 行单列了「等 V3 组收尾」（组边界回压） | 组边界等待和普通 ready/free 同名。**正确**：新增 **`O_GROUP_WAIT`**（`stage5GroupPending` 那一处） | 打点缺口 |
| O4 | `O_QK/O_QH/O_ATTN_V/O_MASK/O_OUTPUT` | 两套实现：`arch35/chunk_fwd_o_cube.h`+`vector.h` 与 `arch35/gemm/kernel/gdn_fwd_o_kernel.hpp` | 图③ 只画了一套 | A5 dispatch 走 `chunk_fwd_o_a5.h`（cube/vector 类），gemm kernel 那套也被 include；CSV 里只有名字，无法区分。**正确**：timer args 加实现/变体标签，或确认只编译一套 | 打点错 |
| O5 | `O_ATTN_V` | `cube.h:279/298`；`gemm/kernel/...hpp:499/506/515/522` | 图③ C3 是一块 | 实际有 128/256、ToUB/GM 四种实现同名。**正确**：加后缀 **`O_ATTN_V_128UB` / `_256` / `_GM`** | 打点错 |
| O6 | `O_GATE` | `chunk_fwd_o_vector.h:464` | 图③ V1 与 C1 并行（这条是对的），但 V1 没有出边 | gate（gateA/gateO）只被 `Stage3Gate64VF`（V2）消费。**正确**：补一根 **V1 → V2** 的箭头，不要画 V1 → C1 | 图粗 |

### 2.4 通用

| # | 项 | 现状 | 代码实际 / 正确应该是什么 | 类型 |
|---|---|---|---|---|
| G1 | kernel 间空隙 | 图① 下方画了 Prepare→FwdH→FwdO 的串行段 | 三个 launch 各自 `KERNEL_TIMING`、各自原点（trace metadata：`each kernel origin is independent`），**kernel 之间的空隙测不到**。**正确**：图上标注"kernel 间未计时"，需要时用 msprof 补 | 图粗 |
| G2 | 嵌套窗口 | 图未说明 | `PREPARE_NORM_GATE` 内嵌套 `FRONT_WAIT`；各项不可相加。**正确**：图上标注窗口层级，分析只用 `KERNEL_TIMING` 做总量 | 图粗 |
| G3 | 32 槽截断 | 图未说明 | 每核每阶段 `MAX_DYNAMIC_ITER = 32`，长序列（T=11274）可能截断。**正确**：分析前看 `metadata.iterations` 的 `truncated` 标记 | 图粗 |
| G4 | PHASE6 空 | 图未区分 | 两条性能 case 的 PHASE6 slice 为空（走三算子路径）。**正确**：图注"本图对应 PREPARE/FWD_H/FWD_O，PHASE6 未执行" | 图粗 |

---

## 3. 修正后的正确时序

### 3.1 Prepare（一包内，G=2）

```text
AIV: S0 | S1a(pad+k̂上传, notify) | S1b(gate/β) | S3a(gate 预计算) | S3b(等KKT + L=KKT⊙G + 打包/上传) | drain w/u(上一包) | S6(Vb→Kbg)
AIC:        [等 k̂] → S2(KKT, 每 HK 一次) | prefill I·I | [等 −L(t)] → S4 issue(t)/dump(t-1) | S5 | [等 S6(t)] → S7(t) [W→U]
```

重叠关系：
- **S2 ∥ S1b**：k̂ 一落 L1 就放行 KKT，gate/β 继续在 AIV 上算；
- **S3a ∥ S2 尾部**：gate 预计算不需要 KKT；
- **prefill ∥ S3b**：`Stage4_PrefillI` 在等 S3 之前下发；
- **S7(t) ∥ S6(t+1)**：逐 task 等 `WaitAivStage6Done(t)`；
- 包间：下一包 S1′ 与上一包 S5 尾/S7 重叠；`DrainPackWuToGm` 在下一包 S3b 之后、S6′ 之前。

### 3.2 FwdH

```text
AIC: C1(h0) | C1(h1) | C2(h0) | C2(h1) | [等H h0] | C1′(h0) | [等H h1] | C1′(h1) ...
AIV: [等P] | V1(h0) | V1(h1) | V2(h0) | V2(h1) | V1′(h0) | V1′(h1) | V2′(h0) | V2′(h1) ...
```

重叠关系：`V1(h0) ∥ C1(h1)`、`V2(h0) ∥ C2(h1)`、`C1′(c+1,h0) ∥ V2(c,h1)`（H flag 逐 head）。
预取（W/K）与 C2 并行，但**当前无打点**。

### 3.3 FwdO

```text
AIV: V1(gate, 逐 head) | V2(mask, 逐 head) | V3(fuse, 逐 head) | V1′(下一轮) ...
AIC: C1 = QK+QH (逐 head) | C3 = AttnV (逐 head) | [等 V3 组收尾] | C1′(下一组) ...
```

依赖：`C1` 不等 `V1`；`V2` 等 `C1`（attn）和 `V1`（gate）；`C3` 等 `V2`（A′）；`V3` 等 `C3`（O_l）；下一组 `C1′` 等本组 `V3` 全部收尾。

---

## 4. 已确认一致的部分

这些图/打点是对得上的，不用改：

- `S6 ∥ S4/S5`（AIV 的 Stage6 与 AIC 的 S4/S5 并行）；
- `w/u 回写推迟到下一包 S3′ 之后`（`DrainPackWuToGm` 位置正确，只是没打点）；
- `FwdO 的 V1 与 C1 并行`（C1 只等预取事件，不等 gate）；
- `FwdH 的 H 就绪 flag 逐 head 串行`；
- `FwdO 组边界等 V3 收尾`（机制存在，只是被 `O_WAIT` 合并）；
- `S7 = W 先、U 后`（Prepare 路径）。

---

## 5. 校验命令

**① 判断跨核时间基准是否可用（"KKT 全局第一"这类现象必查）**

```bash
awk -F, 'NR==1 || $6=="KERNEL_TIMING"{print $3,$4,$5,$8,$9}' gdn_npu_time_parsed.csv
```

比较 AIC 与 AIV 的 `KERNEL_TIMING` 起点：
- 只差几 µs → 时钟基本对齐；
- 差出几十 µs 以上的常数 → 计数器原点偏移，跨核先后不可用。

**② 看某阶段的调用序号（区分 task / pack / sibling）**

```bash
awk -F, 'NR==1 || ($6=="KKT_MM" || $6=="PREPARE_NORM_GATE") && $3==0' gdn_npu_time_parsed.csv | head -20
```

- KKT 序号 `0,1` + norm-gate 序号 `2,3` → sibling 任务；
- KKT 序号 `2,3` + norm-gate 回到 `0,1` → 跨包。

**③ 重新生成 Perfetto trace（必须验证共享时钟）**

```bash
python tests/atk/chunk_gated_delta_rule_fwd/scripts/profile_performance.py --device 0 --cycles-per-us <实测频率> --shared-clock
```

**④ 确认走的是三算子路径**

两条性能 case 的 GVA 分别为 2 和 1（均 ≤4），`UsePreparePath = true`。若换成 GVA>4 / V=256 / chunk=128 / fp16，才会进融合 Phase6，届时 `PREPARE/FWD_H/FWD_O` 三个 slice 为空、`PHASE6` 才有数据。

---

## 6. 修改建议

### 打点（按性价比排序）

1. `WU_DRAIN`：包住 `DrainPackWuToGm`（W、U 两次 `CopyUbToGmElems`）；
2. `H_PREFETCH`：W/K lookahead 两个搬入点；
3. `O_GROUP_WAIT`：`stage5GroupPending` 的组边界等待；
4. 拆名：`FRONT_WAIT` → 5 个、`H_WAIT` → 3 类、`O_WAIT` → ready/free/group；
5. 拆窗口：`PREPARE_NORM_GATE` → `K_UPLOAD` + `GATE_BETA`；`KKT_EPILOGUE` → `KKT_WAIT` + 计算；`W_MM/U_MM` → `W_ISSUE/U_ISSUE`；`WU_PREPARE` → `WU_PREP_V/K`；
6. 变体标签：`O_ATTN_V` 加 128/256、ToUB/GM 后缀；`H_WH/H_KV` 带 head/slot。

### 流水图（`zq/gdn_fla_npu_pipeline.svg`）

1. ① Prepare：S1 拆 S1a/S1b，S2 起点前移；「等 KKT」改实块 S3a；prefill 画进 S3b 窗口；S7 改逐 task 阶梯；
2. ② FwdH：预取条带标注"未计时"（或补打点后更新）；C1/C2 标注"窗口含等待"；
3. ③ FwdO：删掉"C↔V 过 GM"，改"Fixpipe→UB"；C1 拆 QK/QH；补 V1→V2 箭头；
4. 全局：标注"kernel 间未计时"、"窗口不可相加"、PHASE6 未执行。

---

## 7. 附：本次对照使用的打点枚举

固定 6 项 + 动态 28 项，定义于
`fla/ops/ascendc/common/kernel_utils/gdn_timer/AscendTimerV2.hpp`：

- 固定：`KERNEL_TIMING_IDX`、`MAIN_INIT`、`MAIN_BODY`、`PHASE6_FRONT_DONE`、`PHASE6_H_DISPATCH_DONE`、`PHASE6_HO_WAIT`
- 动态：`CUMSUM`、`KKT_MM`、`KKT_EPILOGUE`、`SOLVE`、`SOLVE_WAIT`、`WU_PREPARE`、`WU_WAIT`、`W_MM`、`U_MM`、`PREPARE_NORM_GATE`、`PREPARE_KKT_GATE`、`PREPARE_SOLVE_PREFILL`、`PREPARE_SOLVE_ISSUE`、`PREPARE_SOLVE_DUMP`、`H_WH`、`H_VNEW`、`H_KV`、`H_STATE_UPDATE`、`H_WAIT`、`O_QK`、`O_QH`、`O_MASK`、`O_GATE`、`O_ATTN_V`、`O_OUTPUT`、`O_WAIT`、`FRONT_WAIT`、`PHASE_WAIT`
- Launch slice：`PHASE6`、`PREPARE`、`FWD_H`、`FWD_O`（各自独立原点）
