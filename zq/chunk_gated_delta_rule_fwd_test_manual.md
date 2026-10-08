# chunk_gated_delta_rule_fwd 算子测试手册（stock ATK 环境适用版）

> 适用对象：在**普通 pip 版 ATK**（非开发者定制版）环境下，对融合算子 `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd` 做精度与性能测试的工程师。
>
> 环境基线（本文所有命令与结果均在此环境验证）：
> - 代码：main @ `3f4c016a`（2026-09-29 更新后）
> - Wheel：`flash_linear_attention_npu_a5-26.10.0+main.dev3f4c016`（`FLA_NPU_SOC=ascend950` 编译）
> - ATK：26.9.8（pip 安装的 stock 版，**不含**开发者定制能力）
> - SoC：Ascend 950（A5），单卡 device 0，Debian
>
> 两个前提：① 下文所有命令都在 **NPU 服务器的仓库根目录**下执行（示例环境 `/home/z00943842/flash-linear-attention-npu`）；② CANN / torch_npu / Python 3.11 等基础环境已就绪（新机器搭建参考仓内 docs，本手册不覆盖）。
>
> 仓内 `tests/atk/chunk_gated_delta_rule_fwd/README.md` 是按开发者定制 ATK 写的；本手册补充 stock 环境下**需要单独处理的地方**，全部用 ⚠️ 标出，共 3 个前置补丁 + 1 个自备工具，缺一不可。

---

## 0. 先看这张表：跑通测试需要做什么

| # | 事项 | 性质 | 不做的后果 | 详见 |
|---|------|------|-----------|------|
| ⚠️ 1 | `six_aclnn_benchmark.py` dense 路径 `bhtd`→`bnsd` | 上游 bug，需上报 | 六算子链 dense case 全灭（layout 白名单拒绝） | §3.1 |
| ⚠️ 2 | 三个 case JSON + yaml 的 g 生成范围 `[-1,1]`→`[-1,0]` | 上游 bug，需上报 | 大 T 必产 NaN（GDN 数学要求 g≤0） | §3.2 |
| ⚠️ 3 | executor 追加比较器 stub shim | 本地补丁，等定制 ATK 后移除 | ATK 后处理 KeyError，连性能报告都出不来 | §3.3 |
| 🔧 4 | 自备 `compare_offline.py` 离线比对脚本 | 本地工具 | 精度只有"执行了"没有"比过了"的结论 | §4 |

---

## 1. 环境准备

### 1.1 编译安装 wheel

```bash
FLA_NPU_SOC=ascend950 python scripts/build_wheel.py
```

安装时**指定完整文件名**，不要用 `dist/*.whl` 通配符——dist 里存有新旧多个 wheel 时会版本冲突。文件名里的 `dev3f4c016` 是 commit 短哈希，随代码更新而变，先 `ls dist/` 看刚编出来的实际名字，替换下面命令再执行（本文环境为 `dev3f4c016`）：

```bash
python3 -m pip install --force-reinstall --no-deps --no-cache-dir dist/flash_linear_attention_npu_a5-26.10.0+main.dev3f4c016-py3-none-manylinux_2_34_x86_64.whl
```

顺手清掉旧 wheel（换成实际的旧文件名），避免下次误装：

```bash
rm dist/flash_linear_attention_npu_a5-26.10.0+main.dev7a99eb2-*.whl
```

**注意**：每次 `git pull` 更新代码后必须重编重装；同时检查本地补丁是否被冲掉（见 §6 补丁自检）。

### 1.2 确认 ATK 版本与来源

```bash
atk --version && which atk && pip3 show atk 2>/dev/null | head -5
```

要求 ≥26.8.8。若 `which atk` 不是 pip 安装的那个（Docker 内外混装），先解决环境再继续。

---

## 2. 测试体系总览

| 维度 | 用例文件 | 条数 | 三路机制 |
|------|---------|------|---------|
| 精度冒烟（MSS） | `atk_chunk_gated_delta_rule_fwd_mss.json` | 6 条小 shape（T=1~318） | DUT（融合算子，NPU 节点 `phase6`）/ benchmark（六算子链，NPU 节点 `gold`）/ golden（CPU FP64 递推） |
| 精度正式 | `atk_chunk_gated_delta_rule_fwd.json` | 500 条冻结矩阵 | 同上，分片入口 `run_matrix.sh` |
| 性能 | `atk_chunk_gated_delta_rule_fwd_perf.json` | 2 条 A5 模型 shape（推理 T=11274 / 训练 T=8192） | 仅 DUT |

精度用小 shape 是**组合覆盖**（dtype × V 维度 × chunk × 定长/变长 × 状态），性能必须用真实模型规模——两边的用例文件不同，**精度 6 条、性能 2 条是设计如此，不是漏跑**。

统一入口（**本手册只用到 `-scope=performance`**；精度走 §4.1 的专用三路脚本，不走这个入口）：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=performance --soc=ascend950
```

---

## 3. 前置补丁（⚠️ 三项，按顺序打）

### 3.1 ⚠️ 补丁一：dense 路径 layout 拼写（上游 bug）

**症状**：六算子链所有 dense case 报 `RuntimeError: npu_solve_tri: layout must be one of ['bnsd', 'bsnd', 'ntd', 'tnd'], got 'bhtd'`。

**根因**：`six_aclnn_benchmark.py:92` 写了 `layout="bhtd"`。这个拼写只在 ctypes 版包装器下合法（字符串原样透传给 aclnnSolveTri）；stock wheel 的 public `solve_tri` 路由到 stable 版包装器，白名单只有 `bsnd/bnsd/tnd/ntd` 四种。`bnsd` 的形状约定 `[B, H, T, N]` 与 `a_raw` 完全一致，改字符串即可，张量不用动。

varlen 路径（`tnd` 拼写）**不需要动**：上游 #748 已修复 kernel segfault 并移除拒绝护栏。

**处理**：

```bash
sed -i 's/layout="bhtd"/layout="bnsd"/' tests/atk/chunk_gated_delta_rule_fwd/six_aclnn_benchmark.py
```

**验证**：

```bash
grep -n "layout=" tests/atk/chunk_gated_delta_rule_fwd/six_aclnn_benchmark.py
```

期望：92 行 `bnsd`、98 行 `tnd`。

**⚠️ 勿用 bsnd 替代 varlen 的 tnd**：实测 `solve_tri` 用 `bsnd` + `cu_seqlens` + **多序列**时**不报错、安静地算错**（第二序列的 A 整块 O(1) 偏差，详见 §7 问题 5）。这是最危险的一类静默错误，已回退规避，但值得报给 kernel 负责人。

### 3.2 ⚠️ 补丁二：g 生成范围（上游 bug，精度性能都要）

**症状**：性能 case（T=8192/11274）executor 报 `RuntimeError: output[0] 包含 NaN/Inf`；直接调用算子在大 T 下 o 从中途开始 NaN（实测 T=8192 从 token 3648 起 229 万元素 NaN）。

**根因**：case JSON 里 g 的 `range_values` 是 `[-1, 1]`，含正值。GDN 数学要求 g≤0（log 衰减因子，exp(g_cumsum) 必须 ≤1）；g 为正时 chunk 内累计随机游走冲到 +17 量级，指数项连乘在长序列上溢出 fp32。小 T 用例靠运气不炸，大 T 必炸——**"上一轮没炸"≠"数学正确"**。

**对照实验结论**（可作为上报证据）：g∈[-1,1] 时 T=512/2048 finite、T=8192 NaN；g≤0 时全 T finite。

**处理**（三个 JSON + 一个 yaml，按 name 精确定位 g，beta 的 [-1,1] 不动）：

```bash
python3 -c "
import json, re
base='tests/atk/chunk_gated_delta_rule_fwd/'
for name in ['atk_chunk_gated_delta_rule_fwd.json','atk_chunk_gated_delta_rule_fwd_mss.json','atk_chunk_gated_delta_rule_fwd_perf.json']:
    p=base+name
    cases=json.load(open(p,encoding='utf-8'))
    n=0
    for c in cases:
        for i in c['inputs']:
            if i['name']=='g' and i.get('range_values')==[-1,1]:
                i['range_values']=[-1,0]; n+=1
    json.dump(cases,open(p,'w',encoding='utf-8'),ensure_ascii=False)
    print(name,'patched',n,'cases')
p=base+'chunk_gated_delta_rule_fwd.yaml'
s=open(p,encoding='utf-8').read()
s2=re.sub(r'(- name: g\n(?:\s+.*\n)*?\s+values: )\[\[-1, 1\]\]', r'\g<1>[[-1, 0]]', s, count=1)
assert s2!=s, 'yaml pattern not found'
open(p,'w',encoding='utf-8').write(s2)
print('yaml patched')
"
```

**注意**：JSON 是"冻结矩阵"，改完哈希就变了。正式出报告时在结论里注明"g 范围修正版"，不要与开发者历史结果直接混比。

### 3.3 ⚠️ 补丁三：比较器 stub shim（本地补丁）

**症状**：任何 scope（含 performance）的 post_process 阶段报 `KeyError: 'cv_fused_double_benchmark'`。

**根因**：ATK 的 `post_process` 无论精度还是性能任务都会先构造 `CompareExecutor`（`opp_tasks.py:316`），后者到 `atk.tasks.post_process.ACCURACY_REGISTRY` 查 `cv_fused_double_benchmark`——这个三方比较器只存在于开发者的定制 ATK，stock 26.9.8 没有，仓里也没有。

**两个坑要避开**：
1. 不能照抄 `executor_solve_tri.py` 的 shim 写法：它从 `atk.tasks.task_plugins_register` 导入注册表，而 post_process 用的是 `atk.tasks.post_process` 里的——**两个不同的 Registry 实例**，注册进前者无效（该 shim 在 26.8.8+ 上是死代码）。
2. stub 的 `accuracy_calc` 必须 raise 而不是静默通过，防止"比对缺失"被当成 PASS。

**处理**（追加到 executor 末尾）：

```bash
python3 -c "
p='tests/atk/chunk_gated_delta_rule_fwd/executor_chunk_gated_delta_rule_fwd.py'
s=open(p,encoding='utf-8').read()
if 'register_with_key' in s:
    print('already patched'); raise SystemExit
shim='''

def _register_cv_fused_double_benchmark_stub() -> None:
    try:
        from atk.tasks.post_process import ACCURACY_REGISTRY
    except Exception:
        return
    if \"cv_fused_double_benchmark\" in ACCURACY_REGISTRY:
        return

    class _CvFusedDoubleBenchmarkStub:
        def __init__(self, config, need_md5=None, **thresholds):
            self.config = config
            self.thresholds = thresholds
            self.bm_path = None
            self.bm_remote_path = None

        def accuracy_calc(self, *_args, **_kwargs):
            raise NotImplementedError(
                \"cv_fused_double_benchmark comparator absent in stock ATK; \"
                \"use offline comparison or the vendor ATK build.\")

    ACCURACY_REGISTRY.register_with_key(
        \"cv_fused_double_benchmark\", _CvFusedDoubleBenchmarkStub)


_register_cv_fused_double_benchmark_stub()
'''
open(p,'a',encoding='utf-8').write(shim)
print('shim appended')
"
```

**效果**：性能任务只构造不调用比较器 → 完整报告可出；精度任务的三路执行与数据落盘正常，post-process 会响亮报 `NotImplementedError`（预期行为），精度结论改由离线比对给出（§4）。

---

## 4. 精度测试

### 4.1 跑 MSS 冒烟

前置：§3 三个补丁已打。

```bash
GDN_ATK_CASE_JSON="$PWD/tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd_mss.json" bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_double_benchmark.sh 0
```

两个参数说明：
- 末尾的 `0` 是 **NPU 设备号**（默认就是 0，单卡环境可不传）
- `GDN_ATK_CASE_JSON` 指定用例文件；不设时默认跑正式 500 条（`atk_chunk_gated_delta_rule_fwd.json`），冒烟必须显式指到 `_mss.json`

脚本内置了 `--save_data input/output/profile`（数据自动落盘，§4.2 用）、ATK 版本检查和并发度控制（`GDN_ATK_MAX_TASK`，默认 5），跑完在终端打印 `双标杆精度测试完成：<run_dir>`。

**怎么判断跑对了**：
- run 目录下 `runtime_role_contract.json` 生成且日志出现 `[gdn-double-atk] 角色合同通过：6 case，角色=['dut', 'benchmark', 'golden']` —— 三路任务全部初始化并执行
- run 目录下 `summary.json` 里 `execution_failed: 6` 是**没有比较器结果的记法**（stub 的预期表现），不是执行失败；执行失败的真凶看同目录 `atk_task.log`（里面应当只有 NotImplementedError、没有别的 Traceback）

### 4.2 🔧 自备工具：离线三方比对

三路数据由脚本自动落盘（`--save_data output` 已内置，无需手动加参数），目录结构：

```
atk_output/double_benchmark/<时间戳>/atk_output/<API目录>/
├── input/.../<case>/input.bin                  # 冻结输入
└── output/
    ├── npu_phase6/.../<case>/output_N.pt       # DUT（融合算子）
    ├── npu_gold/.../<case>/output_N.pt         # benchmark（六算子链）
    └── cpu_benchmark/.../<case>/output_N.pt    # golden（CPU FP64）
```

输出个数随 scenario 变：3 个 = `[o, g_cumsum, A]`；4 个 = `[o, final_state, g_cumsum, A]`。

比对脚本 `compare_offline.py` 放仓根（完整源码见附录 A）。两种等价用法——传 run 目录时间戳，或不带参数自动取最新一轮：

```bash
python3 compare_offline.py $(ls -t tests/atk/chunk_gated_delta_rule_fwd/atk_output/double_benchmark/ | head -1)
```

```bash
python3 compare_offline.py
```

### 4.3 结果怎么读

- `max_d/bm`、`avg_d/bm`、`rms_d/bm`：DUT 误差 ÷ 六算子链误差的三个统计量（逐元素相对误差 `|a-ref|/max(|ref|,1e-6)`，分子分母同公式，比值对误差定义不敏感）
- 判定阈值（仓内 `atk_role_contract.py`）：**max ≤ 5、avg ≤ 1.5、rms ≤ 1.5**，即"DUT 误差不得超过六算子链这个现实噪声地板的 5×/1.5×/1.5×"
- `d_err` / `bm_err`：DUT、链各自对 FP64 golden 的**最大绝对误差**——这两个数是不依赖任何阈值解释的硬结论，最有说服力
- 比值 `0.000` = DUT 误差真为零（输出落 dtype 后与 golden 逐元素一致，小 T 退化场景正常）；比值 `1.000` = 两边输出逐位一致（如 g_cumsum 两边同一套 cumsum 实现），**不代表误差为零**，要看 d_err 列确认

### 4.4 本次实测结果（2026-09-29，run 20260929_162035）

| case | output | max_d/bm | avg_d/bm | rms_d/bm | d_err | bm_err | verdict |
|---|---|---|---|---|---|---|---|
| 0 | o | 1.000 | 1.000 | 1.000 | 7.451e-09 | 7.451e-09 | PASS |
| 0 | g_cumsum | 0.000 | 0.000 | 0.000 | 0 | 0 | PASS |
| 0 | A | 0.000 | 0.000 | 0.000 | 0 | 0 | PASS |
| 1 | o | 1.000 | 1.000 | 1.000 | 1.192e-07 | 1.192e-07 | PASS |
| 1 | g_cumsum | 0.000 | 0.000 | 0.000 | 0 | 0 | PASS |
| 1 | A | 0.000 | 0.000 | 0.000 | 0 | 0 | PASS |
| 2 | o | 1.000 | 1.000 | 1.000 | 2.384e-07 | 2.384e-07 | PASS |
| 2 | final_state | 1.394 | 0.784 | 1.099 | 9.426e-07 | 1.662e-06 | PASS |
| 2 | g_cumsum | 0.000 | 0.000 | 0.000 | 0 | 0 | PASS |
| 2 | A | 0.000 | 0.000 | 0.000 | 0 | 0 | PASS |
| 3 | o | 1.000 | 0.994 | 0.990 | 3.815e-06 | 3.815e-06 | PASS |
| 3 | final_state | 0.545 | 0.946 | 0.816 | 2.058e-05 | 3.492e-05 | PASS |
| 3 | g_cumsum | 1.000 | 1.000 | 1.000 | 3.815e-06 | 3.815e-06 | PASS |
| 3 | A | 1.000 | 1.000 | 1.000 | 6.104e-05 | 6.104e-05 | PASS |
| 4 | o | 1.000 | 1.000 | 1.000 | 2.384e-07 | 2.384e-07 | PASS |
| 4 | g_cumsum | 1.000 | 1.000 | 1.000 | 1.192e-07 | 1.192e-07 | PASS |
| 4 | A | 0.000 | 0.000 | 0.000 | 0 | 0 | PASS |
| 5 | o | 1.000 | 1.000 | 1.001 | 1.192e-07 | 1.192e-07 | PASS |
| 5 | g_cumsum | 1.000 | 1.000 | 1.000 | 1.526e-05 | 1.526e-05 | PASS |
| 5 | A | 1.000 | 1.000 | 1.000 | 7.629e-06 | 7.629e-06 | PASS |

**OVERALL: PASS（6/6 case，全部输出）**

结论要点：
- **DUT（融合算子）对 FP64 golden 的最大绝对误差 ≤ 6.1e-5**，多数输出在 1e-7 量级——不依赖阈值解释的硬精度结论
- 覆盖维度：BF16/FP16、V128/V256、chunk 64/128、定长/变长（单序列/多序列）、GVA、初末状态
- case 3（T=48 多序列 V=256）上 DUT 误差显著小于六算子链（曾测得 1/50~1/5）：链在算子间把中间量落成 bf16（`a_raw.to(q.dtype)`），融合核内部保持 fp32 累加——融合的价值有直接数据支撑
- 阈值语义为对仓内 `atk_role_contract.py` 的 `METRIC_TENSOR_PAIRS` + 阈值名的解读，非官方比较器逐行复刻；要与开发者历史结果严格对齐需其定制 ATK 对照

### 4.5 正式 500 条

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

分片入口（末尾 `0` 同样是设备号），每 25 条起一个 fresh ATK 进程，已完成的分片自动复用、可断点续跑。**注意输出布局与冒烟不同**：结果在 `atk_output/generalized500_<时间戳>/shard_<起>_<止>/`，每个 shard 目录内部结构同 §4.2（`atk_output/<API>/...`），且 ATK 任务名没有 `_mss` 后缀。离线比对需对每个 shard 各跑一次 `python3 compare_offline.py <shard目录>`（附录 A 脚本支持直接传 run 目录），并把脚本外层循环 `range(6)` 改成对应分片的 `range(start, end)`。

---

## 5. 性能测试

前置：§3.2（g 范围，否则 NaN）与 §3.3（stub，否则报告生成失败）**同样必须生效**。

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=performance --soc=ascend950
```

**注意**：`--soc=ascend950` 要显式给；不传时脚本从 npu-smi 解析型号，失败会默认按 ascend910b 处理。

指标解读：
- `device_perf`：e2e 每迭代耗时（µs）；`aicore_time`：AI Core 耗时，两者之比是 host 开销占比
- `aic_mac_ratio`：cube 流水利用指标
- `perf_fluctuation_result`：波动检查，采样轮数少时容易不过，正式结论建议多轮重跑取稳定值

### ⚠️ 设备状态陷阱（先读这个再看数字）

同代码同输入下，性能测量出现过整段"系统性偏慢"的窗口（慢 1.5~1.9 倍且波动大）。**代码因素已排除**：wheel 安装时间为 9/29 09:25，早于包括慢轮在内的所有测量轮——全部轮次跑的是同一个二进制。复核方法：`ls -ldt /usr/local/python3.11.10/lib/python3.11/site-packages/flash_linear_attention_npu_a5-*.dist-info`，dist-info 目录时间即安装时间。

五轮数据（case 0 推理 b1_t11274_gva2 / case 1 训练 b2_t8192_mha）：

| 时间 | 设备状态 | case 0 | case 1 | case 0 std |
|---|---|---|---|---|
| 9/29 09:46 | 独占 | 2879 µs | （中断未跑） | 193 µs |
| 9/29 10:08 | 独占 | 3193 µs | 4045 µs | 218 µs |
| 9/29 17:41 | 有并发负载 | 1701 µs | 2872 µs | 2.1 µs |
| 9/30 09:03 | 独占 | **1703.2 µs** | **2703.8 µs** | 2.0 µs |
| 9/30 09:04 | 独占 | **1706.2 µs** | **2705.2 µs** | 2.4 µs |

解读：
- **稳定真值：case 0 ≈ 1704 µs、case 1 ≈ 2704 µs**——9/30 两轮背靠背独占复现，差异 <0.2%、std 个位数 µs
- 慢窗口只出现在 9/29 上午，且 std 高达 218 µs（6.8%）、伴随波动检查失败——设备当时处于异常状态（空闲降频未被短测试唤醒，或共享服务器上其他用户的隐蔽负载），与代码无关，具体原因已无法回溯
- 9/29 17:41 的 case 1（2872 µs，std 93）比独占真值慢 6%——真实并发共享的开销；同轮 case 0 未受影响

**正式测速的规矩**：
- 单轮数字不作数，至少两轮背靠背复现；**std 是可信度指标**——个位数 µs 才是标杆状态，上百 µs 说明环境有干扰
- 数字与历史值差 >10% 或 std 异常时，先查环境再报数：`npu-smi info` 看 AI Core 频率、确认有没有别人的进程
- 结论必须记录设备状态与复现轮数；对比优化前后须在同一状态下测

### 本次实测结果（正式值：9/30 两轮背靠背独占复现）

| 指标 | 推理 `a5_inference_b1_t11274_gva2`<br>(B=1,Hk=16,Hv=32,T=11274,varlen,chunk64) | 训练 `a5_training_b2_t8192_mha`<br>(B=2,Hk=Hv=32,T=8192,dense,chunk64) |
|---|---|---|
| **device_perf（e2e）** | **1703.2 / 1706.2 µs**（两轮） | **2703.8 / 2705.2 µs**（两轮） |
| AI Core 耗时 | 1658.2 µs（占比 97.4%） | 2644.2 µs（占比 97.8%） |
| aic_mac_ratio | 1.327 | 1.237 |
| 折算吞吐 | ≈661 万 token/s | ≈606 万 token/s |
| 显存峰值（reserved） | 1255 MB | 1885 MB |
| 标准差 | 2.0~2.4 µs（0.1%） | 2.9~3.1 µs（0.1%） |

报告文件（9/30 两轮，均为 `Total Task: 2, success 2, failed 0`）：
- `atk_output/perf/atk_output/atk_chunk_gated_delta_rule_fwd_perf_2026-09-30-09-03-15-277182/report/atk_chunk_gated_delta_rule_fwd_perf_reports_2026-09-30-09-03-15.xlsx`
- `atk_output/perf/atk_output/atk_chunk_gated_delta_rule_fwd_perf_2026-09-30-09-04-02-874386/report/atk_chunk_gated_delta_rule_fwd_perf_reports_2026-09-30-09-04-02.xlsx`

历史轮归档：9/29 17:41（有并发，case 0 与正式值一致，case 1 因共享 +6%）、9/29 上午两轮（设备异常慢窗口，仅作对照保留，不作结论依据）。

解读要点：
- AI Core 占比 97%+：host 侧调度无瓶颈，两轮一致
- TFlops/MFU 显示 0 是因为 executor 未定义 `cal_cube_computation()`，非真实值

---

## 6. 诊断命令速查

```bash
# 捞最近一次精度 run 的执行错误（排除 stub 的 NotImplementedError 后应无其他 Traceback）
grep -nE "Traceback|RuntimeError|ValueError|KeyError|Segmentation" $(ls -td tests/atk/chunk_gated_delta_rule_fwd/atk_output/double_benchmark/*/ | head -1)/atk_task.log | head -60
```

```bash
# 绕开 ATK 直接最小验证融合算子（含 varlen 拍平 chunk_indices 的正确格式）
python3 -c "import torch, torch_npu; from fla_npu.ops import ascendc; q=(torch.rand(1,1,3,128,device='npu:0')*0.1-0.05).to(torch.bfloat16); k=(torch.rand(1,1,3,128,device='npu:0')*0.1-0.05).to(torch.bfloat16); v=(torch.rand(1,1,3,128,device='npu:0')*0.1-0.05).to(torch.bfloat16); g=-torch.rand(1,3,1,device='npu:0')*0.5; beta=torch.rand(1,3,1,device='npu:0').to(torch.bfloat16); out=ascendc.chunk_gated_delta_rule_fwd(q,k,v,g,beta,initial_state=None,output_final_state=False,chunk_size=64,cu_seqlens=[0,3],chunk_indices=[0,0],scale=0.08838834764831843); print('OK', out[0].shape)"
```

> `chunk_indices` 必须是**拍平的一维列表** `[seq_id, chunk_id, seq_id, chunk_id, ...]`；传嵌套 `[[0,0]]` 会在 `_host_ints` 处 `TypeError: unhashable type: 'list'`。

```bash
# 绕开 ATK 单独验证六算子链（dense / varlen 各一）
PYTHONPATH=tests/atk/chunk_gated_delta_rule_fwd python3 -c "import torch, torch_npu; from fla_npu.ops import ascendc; from six_aclnn_benchmark import run_six_aclnn_core; q=(torch.rand(1,1,3,128,device='npu:0')*0.1-0.05).to(torch.bfloat16); k=(torch.rand(1,1,3,128,device='npu:0')*0.1-0.05).to(torch.bfloat16); v=(torch.rand(1,1,3,128,device='npu:0')*0.1-0.05).to(torch.bfloat16); g=-torch.rand(1,3,1,device='npu:0')*0.5; beta=torch.rand(1,3,1,device='npu:0').to(torch.bfloat16); out=run_six_aclnn_core(ascendc,q,k,v,g,beta,initial_state=None,output_final_state=False,chunk_size=64,cu_seqlens=None,scale=0.08838834764831843); print('DENSE OK', out[0].shape)"
```

（varlen 版把 `cu_seqlens=None` 换成 `cu_seqlens=[0,3]`。）

```bash
# git pull 后补丁自检：三处都应在
grep -n "layout=" tests/atk/chunk_gated_delta_rule_fwd/six_aclnn_benchmark.py   # 期望 bnsd + tnd
grep -c "register_with_key" tests/atk/chunk_gated_delta_rule_fwd/executor_chunk_gated_delta_rule_fwd.py   # 期望 ≥1
python3 -c "import json; d=json.load(open('tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd_perf.json')); print([i['range_values'] for c in d for i in c['inputs'] if i['name']=='g'])"   # 期望 [[-1, 0], [-1, 0]]
```

---

## 7. 已发现问题清单（建议上报）

| # | 问题 | 位置 | 性质 | 处理状态 |
|---|------|------|------|---------|
| 1 | dense 路径 `layout="bhtd"` 不在 stable 包装器白名单，dense 用例必炸 | `six_aclnn_benchmark.py:92` | 上游 bug | 本地已补（§3.1），待上游修 |
| 2 | g 生成范围 `[-1,1]` 违背 GDN 数学约束（g≤0），大 T 必产 NaN | 3 个 case JSON + yaml | 上游 bug | 本地已补（§3.2），待上游修 |
| 3 | `cv_fused_double_benchmark` 比较器依赖定制 ATK，stock 版连性能报告都出不来；README "ATK≥26.8.8" 要件不足 | 测试基建 | 环境可复现性 | 本地 stub 顶住（§3.3），待定制 ATK 或比较器进仓 |
| 4 | `executor_solve_tri.py` 的 shim 注册进 `task_plugins_register` 注册表，与 post_process 实际使用的**不是同一对象**，26.8.8+ 上是死代码 | `tests/atk/solve_tri/` | 上游 bug | 未修，报给同事 |
| 5 | `solve_tri` 用 `bsnd`+`cu_seqlens`+**多序列**时**静默算错**（第二序列 A 整块 O(1) 偏差，不崩溃不报错）；stable 包装器报错信息还在建议用 bsnd | kernel / 包装器提示语 | 上游 bug（最危险的一类） | 已回退 tnd 规避（§3.1 注意事项） |
| 6 | tnd 拒绝护栏已过期（kernel #481 已修、#748 已移除护栏）——历史确认项，无需处理 | `_stable.py` | 已修复 | 关闭 |

上报时的证据材料：§4.4 表格、§5 表格、§3.2 的对照实验数据（g 正值 T=8192 NaN from token 3648 / g≤0 全 finite）、§8 排查记录。

---

## 8. 附：本次排查过程时间线（供复盘）

1. **角色合同失败 `role_count=14 != 18`** → 实为六算子链崩溃连带（benchmark 崩 → ATK 撕掉同 case 任务组 → DUT 角色打印缺失）
2. **`layout='tnd' is refused`** → 旧版护栏；上游 #748 已删（更新代码后消失）
3. **`layout must be one of [...], got 'bhtd'`** → 补丁一（§3.1）
4. **`KeyError: 'cv_fused_double_benchmark'`** → 补丁三（§3.3）；期间踩坑：shim 注册表来源必须用 `atk.tasks.post_process`（两个 Registry 实例问题）
5. **性能 case `output[0] 包含 NaN/Inf`** → 补丁二（§3.2），对照实验定位 g 正值
6. **离线比对 case 3/5 `bm_err≈1.0`** → 曾误用 bsnd 修 varlen（问题 5），错误精确覆盖第二序列（case 5 token 127~317 全中、4 头全中），回退 tnd 后恢复
7. **性能五轮不一致（9/29 上午 2879/3193 µs vs 其余三轮 ~1704 µs）** → 排除代码因素（wheel 安装于 9/29 09:25，早于全部轮次，同一二进制）；慢窗口仅出现在 9/29 上午且 std 高达 218 µs，属设备异常状态（降频或共享服务器隐蔽负载）；9/30 两轮背靠背独占复现出稳定快值（1703/1706、2704/2705，std 2~3 µs）定为正式结论，详见 §5 设备状态陷阱
8. 最终：精度 OVERALL PASS + 性能正式报告（9/30 两轮独占复现定稿），全部结论有落盘数据支撑

---

## 附录 A：compare_offline.py 完整源码

```python
#!/usr/bin/env python3
"""离线三方精度比对: DUT(融合) vs golden(FP64), benchmark(六算子链) vs golden(FP64).

用法: python3 compare_offline.py [run目录 | double_benchmark下的时间戳]
      不带参数时自动取 double_benchmark 下最新一轮。
      run 目录 = 含 atk_output 子目录的那层（冒烟时间戳目录或 500 条矩阵的 shard 目录）。
"""
import glob
import os
import sys
import torch

ROOT = "tests/atk/chunk_gated_delta_rule_fwd/atk_output/double_benchmark"
arg = sys.argv[1] if len(sys.argv) > 1 else sorted(
    p for p in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, p)))[-1]
if os.path.isdir(os.path.join(arg, "atk_output")):
    run_dir = arg                      # 直接传 run 目录（含 atk_output 的那层，如 shard 目录）
else:
    run_dir = os.path.join(ROOT, arg)  # 传 double_benchmark 下的时间戳
print("run dir:", run_dir)
BASE = os.path.join(run_dir, "atk_output")
BASE = os.path.dirname(sorted(glob.glob(f"{BASE}/*/output"))[0])
TASK = "atk_chunk_gated_delta_rule_fwd_mss"
THRESH = {"max": 5.0, "avg": 1.5, "rms": 1.5}
NAMES = {3: ["o", "g_cumsum", "A"], 4: ["o", "final_state", "g_cumsum", "A"]}


def rel_err(a, ref):
    a = a.double()
    ref = ref.double()
    return (a - ref).abs() / ref.abs().clamp_min(1e-6)


def stats(e):
    return e.max().item(), e.mean().item(), e.pow(2).mean().sqrt().item()


def ratio(dv, bv):
    if bv <= 0:
        return 0.0 if dv <= 0 else float("inf")
    return dv / bv


def load_case(role, cid):
    d = f"{BASE}/output/{role}/{TASK}/{cid}"
    outs, i = [], 0
    while os.path.exists(f"{d}/output_{i}.pt"):
        outs.append(torch.load(f"{d}/output_{i}.pt", map_location="cpu", weights_only=False))
        i += 1
    return outs


print(f"{'case':<6}{'output':<14}{'max_d/bm':<11}{'avg_d/bm':<11}{'rms_d/bm':<11}verdict")
overall = True
for cid in range(6):
    dut = load_case("npu_phase6", cid)
    bm = load_case("npu_gold", cid)
    gd = load_case("cpu_benchmark", cid)
    if not gd:
        print(f"{cid:<6}NO GOLDEN OUTPUTS FOUND")
        overall = False
        continue
    if not (len(dut) == len(bm) == len(gd)):
        print(f"{cid:<6}ROLE COUNT MISMATCH dut={len(dut)} bm={len(bm)} gd={len(gd)}")
        overall = False
        continue
    names = NAMES.get(len(gd), [f"out{i}" for i in range(len(gd))])
    for i, name in enumerate(names):
        if dut[i].shape != gd[i].shape or bm[i].shape != gd[i].shape:
            print(f"{cid:<6}{name:<14}SHAPE MISMATCH {tuple(dut[i].shape)}/{tuple(bm[i].shape)}/{tuple(gd[i].shape)}")
            overall = False
            continue
        sd = stats(rel_err(dut[i], gd[i]))
        sb = stats(rel_err(bm[i], gd[i]))
        r = [ratio(sd[k], sb[k]) for k in range(3)]
        ok = r[0] <= THRESH["max"] and r[1] <= THRESH["avg"] and r[2] <= THRESH["rms"]
        overall = overall and ok
        ad = (dut[i].double() - gd[i].double()).abs().max().item()
        ab = (bm[i].double() - gd[i].double()).abs().max().item()
        print(f"{cid:<6}{name:<14}{r[0]:<11.3f}{r[1]:<11.3f}{r[2]:<11.3f}"
              f"d_err={ad:.3e} bm_err={ab:.3e}  {'PASS' if ok else 'FAIL'}")

print("\nOVERALL:", "PASS" if overall else "FAIL")
```

注：① 500 条矩阵的 ATK 任务名没有 `_mss` 后缀（`TASK` 改为 `atk_chunk_gated_delta_rule_fwd`），且结果是分片布局（见 §4.5）——对每个 shard 目录各跑一次、`range(6)` 改成对应分片的 `range(start, end)`；② `NAMES` 映射只认 3/4 个输出，用例同构时无需改动。

---

## 附录 B：结论存档要件

按仓内 README 要求，正式结论必须记录：代码 commit（`3f4c016a`）、ATK/CANN 版本（26.9.8）、SoC（ascend950）、实际加载的 OPP 路径、case JSON 哈希（g 范围修正版）、原始报告路径（§4.4 / §5 已列）。
