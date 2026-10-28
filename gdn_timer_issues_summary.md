# GDN C/V 流水计时问题总结

## 1. 背景

本次目标是用 AscendTimerV2 观察 GDN 三算子路径 `prepare → fwd_h → fwd_o` 的 AIC/AIV 流水，并让 MindStudio Insight 中的跨核先后顺序与代码中的数据依赖一致。

采集使用 `GetSystemCycle()` 记录原始 cycle，A5 按已核实的 `1800 cycles/us` 换算显示。由于现有 msprof 采集没有提供可用的算子内部 per-pipe 原始数据，本次图的本质是：

> **共享硬件时钟上的标量控制流时间轴 + 各 Stage 的指令下发窗口。**

它不是 MTE1/MTE2/FIX/V 等硬件 pipe 的逐指令执行 trace。Stage 窗口可以重叠，也不能直接相加得到总算子耗时。

---

## 2. 问题一：跨核等待没有挡住标量，AIC 的计时点整体跑到 AIV 前面

### 现象

最初的 Prepare trace 中出现了明显违反代码依赖的图形：

- AIC 的 KKT 看起来早于 AIV 的 K 上传和 notify。
- AIC 多个 Stage 看起来总是比 AIV 早开始。
- `S2_WAIT_S1` 很短，但 KKT 与 notify 之间仍相差数微秒。
- 图上像是“K 还没准备好，KKT 就已经开始”。

一组典型数据中，AIC 的 KKT 起点约为 `5.37 us`，AIV notify 却约为 `10.56 us`。

### 根因

相关代码原来使用：

```cpp
CrossCoreWaitFlag<0x4, PIPE_MTE1>(flag);
CrossCoreWaitFlag<0x4, PIPE_V>(flag);
```

这些写法将等待挂到指定 pipe，并没有让标量控制流停住。真正的数据 pipe 仍会等待，因此原算子的数据依赖不一定有功能错误；但标量已经越过等待并继续执行后续的 `TikNoBarrier()`。

计时器读取的是标量时间，所以记录到的是“标量提前下发后续指令”的时刻，而不是“跨核数据已经就绪”的时刻。

这也解释了为什么等待窗口很短：它测到的是标量越过等待指令的时间，不是数据 pipe 真正阻塞的时间。

### 错误修法：直接使用 PIPE_ALL

将等待改为默认 `PIPE_ALL` 后，跨核先后关系有所改善，但 Prepare 的 body 时间增加约 23%。

原因是它引入了过大的流水约束，不只是让标量等待，还影响了其他 pipe 的原有重叠。因此它不适合作为本次修复方案。

### 最终修改

对于需要让标量控制流等待“数据已就绪”的跨核点，改为：

```cpp
CrossCoreWaitFlag<0x4, PIPE_S>(flag);
```

这样计时点不会在跨核就绪条件满足前被标量提前记录，同时避免 `PIPE_ALL` 的过度约束。

Prepare 中相关的 Stage1/Stage3/Stage6 等跨核等待按这一原则处理。生产侧的 `CrossCoreSetFlag` 仍保留实际生产 pipe，例如 `PIPE_FIX`、`PIPE_MTE3`，没有全部改成 `PIPE_S`。

### 验证结果

512 tokens 的 trace 中，Prepare 的 240 项跨核依赖检查全部通过，包括：

- K 上传完成 → KKT。
- notify → KKT。
- KKT → S3 EPI。
- S3 EPI → S4。
- S6 VB/KBG → S7 W/U。

---

## 3. 问题二：ANCHOR 对齐把“每核局部时间”误当成了真实全局流水

### 现象

在排查跨核倒挂时，曾尝试给每个核减去自己的 `ANCHOR`，让各核从 body 入口开始对齐。但这个视图无法回答真实的 C/V 先后关系，也会改变各核之间原本存在的启动时间差。

### 根因

`ANCHOR` 只是各核执行到 body 入口时各自记录的时间点，并不是一次全核 rendezvous。分别减去每个核的 ANCHOR 后，得到的只是：

> 每个核各自进入 body 后的局部经过时间。

它适合观察核内阶段分布，不适合验证跨核因果顺序。

### 最终处理

真实 C/V 流水统一使用：

```text
--shared-clock --cycles-per-us 1800 --no-align-anchor
```

具体规则是：

- AIC/AIV 共同时钟已经核实后，才放在同一绝对时间轴。
- 每个 rank 只使用一个全局 min-start 作为显示零点。
- 不再分别减去各核 ANCHOR。
- ANCHOR 对齐仅保留为可选的“每核 body 内视图”，不用于判断跨核依赖。

`cycles_per_us` 只影响显示单位，不能修复因果倒挂；真正的时间关系始终使用原始 `start_cycle/end_cycle` 检查。

---

## 4. 问题三：Prepare 修好后，fwd_h 仍出现多组跨核倒挂

### 现象

Prepare 的 KKT/notify 关系修正后，fwd_h 仍有大量消费者早于生产者的情况：

- `H C1_WH(i) → V1_VNEW(i+1)`：12/28 倒挂，最大 2539 cycles。
- `H V1_VNEW(i) → C2_KV(i)`：13/28 倒挂，最大 3200 cycles。
- `H C2_KV(i) → V2_STATE(i)`：12/28 倒挂，最大 3413 cycles。

同一文件的 fwd_o 四组依赖均为 32/32 通过，说明问题不能简单归因于整个时钟或整个 JSON 都错误。

### 根因

fwd_h 中存在与 Prepare 相同的标量等待问题，只是涉及不同的数据就绪 flag：

| 数据等待 | 原 pipe | 修改为 |
|---|---|---|
| `FWD_H_P_READY_FLAG` | `PIPE_V` | `PIPE_S` |
| `FWD_H_D_READY_FLAG` | `PIPE_V` | `PIPE_S` |
| `FWD_H_H_READY_FLAG` | `PIPE_MTE2` | `PIPE_S` |
| `FWD_H_RIGHT_READY_FLAG` | `PIPE_MTE2` | `PIPE_S` |

本轮只修改这 4 个数据就绪等待，没有批量修改 `P_FREE/D_FREE/RIGHT_FREE` 等槽位信用等待，也没有调整 Stage 的 Tik/Tok 位置。

### 验证结果

修改后的 512 tokens trace：

- fwd_h 三组依赖均为 28/28 通过，无缺失、无倒挂。
- fwd_o 四组依赖均为 32/32 通过，没有引入回退。
- 加上 Prepare，全图共 452/452 项依赖检查通过。

这说明 512 场景下，全局时间轴上的代码级 C/V 因果关系已经一致。

---

## 5. 问题四：长序列只有 63 号迭代异常，根因是计时槽溢出覆盖

### 现象

8960 tokens 的原始 trace 中：

- Prepare 全部依赖通过。
- fwd_o 四组均为 560/560 通过。
- fwd_h 只有 63 号迭代出现 4 条倒挂和 4 个缺失。
- 其余已记录区间没有同类倒挂。

4 条倒挂全部表现为：

```text
H_V1_VNEW_63 → H_C2_KV_63
```

### 根因

计时器每核、每动态 Stage 的容量为：

```cpp
MAX_DYNAMIC_ITER = 64;
```

原 `NextIter()` 在达到上限后，不断把后续实际迭代钳制到 63：

```cpp
int32_t iter = next_iter_[t];
if (iter >= AscendTimer::MAX_DYNAMIC_ITER) {
    iter = AscendTimer::MAX_DYNAMIC_ITER - 1;
} else {
    next_iter_[t] = iter + 1;
}
return iter;
```

因此，长序列中后续真实事件会反复覆盖 63 号槽。

不同 Stage 的实际调用次数并不完全相同，尤其 fwd_h 首末 chunk 的 V1、C1、C2、V2 数量存在正常差异。各 Stage 最后覆盖进 63 号槽的事件可能来自不同实际迭代，解析器再把相同的 `_63` 后缀配对，就会产生假倒挂和假缺失。

这不是 fwd_h 的 `PIPE_S` 修复失效，而是计时缓冲的采样别名。

### 验证方式

排除所有涉及 63 号溢出槽的依赖后：

- fwd_h 三组有效依赖全部通过，无倒挂。
- Prepare 和 fwd_o 仍全部通过。

这证明异常集中在计时器尾槽，而不是长序列有效采集区间的普遍同步问题。

### 最新修改

`NextIter()` 改为返回真实递增计数：

```cpp
return next_iter_[t]++;
```

当返回值达到 `MAX_DYNAMIC_ITER` 后，现有 Tik/Tok 边界检查直接丢弃后续采样，不再覆盖 0～63 槽。首次溢出时打印的 warning 是预期提示，表示后续采样未记录，不是算子功能错误。

该修复保留每个 Stage 首次采集的 0～63 次事件；它并不表示长序列的全部迭代都被记录。

### 当前验证状态

- 溢出分配逻辑和 Tik/Tok 丢弃语义已完成源码检查与模拟验证。
- 最新溢出补丁后的 8960 设备 trace 尚未回本，不能把这一项写成“设备复测已通过”。

---

## 6. 其他容易误读的问题

### 6.1 Stage 的 NoBarrier 窗口主要是 issue window

`TikNoBarrier()` 和 `TokNoBarrier()` 不会主动排空异步 pipe。MMAD、MTE2、MTE3 等 Stage 的起止点主要表示标量下发窗口，不能直接等同于硬件计算完成时延。

因此：

- 不用这些窗口推导 per-pipe 的精确执行耗时。
- 跨核检查使用代码中明确的生产/消费关系，而不是只看两个色块是否 visually 对齐。
- 等待本身不新增打点；计算 Stage 的 Tik 保持在功能性等待之后。

### 6.2 MAIN_BODY 的上下层级是显示问题

`MAIN_BODY` 是包含多个 Stage 的父区间。Perfetto/MindStudio Insight 的轨道和嵌套排列可能让它显示在 Stage 中间或下方，这不代表真实执行顺序变化。

这类问题只调整 JSON 的轨道组织，保留原始 cycle 和 Stage 区间，不需要修改 kernel 同步。

### 6.3 分支、wheel 和采集脚本必须一致

排查过程中还遇到过 `npu_chunk_gated_delta_rule_fwd() got an unexpected keyword argument 'timer'`。原因是 Python wrapper 与 kernel/UT 不在同一特性分支，并非流水算法问题。

后续使用同一 `feat/gdn-timer-align` 源码构建 GDN 单算子 wheel，同时更新 wrapper 和 kernel，避免只替换其中一个组件。

### 6.4 Prepare 曾出现计时数据只落盘一部分

早期还遇到过 Prepare 固定项和动态计数大量为 0、其他两段正常的问题。最终通过启用显式 GM publish，并在各 kernel 收尾、被测窗口之外调用 `timer_.Publish()` 解决。

该问题属于计时缓冲写回完整性，与跨核等待和 ANCHOR 对齐是不同问题，不能混在一起判断。

---

## 7. 最终结论

1. **AIC 看起来永远早于 AIV 的首要原因，是跨核等待只挡数据 pipe、没有挡标量，标量计时点提前越过了等待。**
2. **`PIPE_ALL` 虽能改变先后关系，但过度约束流水；本次数据就绪等待使用 `PIPE_S`。**
3. **真实跨核比较必须使用共享原始时钟和统一零点，不能分别按各核 ANCHOR 对齐。**
4. **Prepare 修复后，fwd_h 还需要独立检查并修复 4 个数据就绪等待；不能假设同一类问题只存在于一个 kernel。**
5. **长序列 63 号槽异常来自 `NextIter()` 溢出覆盖，不是剩余的跨核同步倒挂。**
6. **512 tokens 已完成全图 452/452 依赖验证；8960 的有效区间已验证，最新溢出补丁仍等待设备复测闭环。**
