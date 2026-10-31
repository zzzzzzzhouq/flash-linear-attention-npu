# GDN A5 阶段计时接入报告

## 已实现

- 分支基于同步后的 `main`，只插桩解耦 A5 Prepare／FwdH／FwdO 路径；arch22 保持空 timer。
- 共 1 个每核 kernel 固定项和 19 个动态阶段类型；Prepare 的 S4 按两个 MMAD bank 分为 `S4_Y0`／`S4_Y1`。
- 所有阶段使用 `TikNoBarrier`，终点复用原实现已有的同步点或异步下发边界；没有新增独立 wait／notify 项，也没有为阶段新增 `SyncAll`／`PipeBarrier`。
- 每个 kernel 使用独立 timer slice；动态 iter 在每物理核、每 kernel 内单调递增，容量为 1024。
- 新增 `aclnnChunkGatedDeltaRuleFwdProfile*` 诊断 ABI；原 `aclnnChunkGatedDeltaRuleFwd*` 参数列表和默认执行路径不变。
- Python `timer=None` 时调用原入口；传入连续 int64 timer 时调用 Profile 入口，并校验三个完整 slice 的容量。
- 采集脚本执行同输入普通／诊断输出精确对照，导出 raw、每 kernel CSV、metadata；trace 只在显式确认频率和共同时钟后生成。

## 静态验证

- 六个修改／新增 Python 文件均通过 `py_compile`。
- C++ 枚举与 Python 名称逐项一致：19 个动态阶段，`MAX_DYNAMIC_ITER=1024`，`N_CORE_COUNT=96`。
- 布局校验：`TIMER_SLICE_NUMEL=3740160`，`TIMER_BUFFER_NUMEL=11220480`。
- 使用三 slice 合成 raw 验证 Prepare／FwdH／FwdO 的 CSV 和 trace：时间区间合法，trace 使用三个独立 pid，没有跨 launch 对齐。
- `git diff --check` 无格式错误。
- 计时模板资产校验在 UTF-8 环境下通过 6 项，另 1 项因本地无 C++17 编译器跳过；该结果不是 CANN 编译结果。

## 未完成与边界

- 本地没有 CANN／NPU，尚未执行 Ascend C 编译、安装、真实 kernel 运行、数值对照或大 case 采集。
- 1024 是采样容量而不是把动态索引扩大到 `INT_MAX`；超过容量的迭代不记录，达到 1024 时需结合设备告警判断是否截断。
- 阶段窗口可能包含原实现已有的资源复用或数据依赖等待，不应简单相加为 kernel 总时长。
- 当前未执行独立子代理审查；已完成本地枚举、配对、参数链和解析交叉检查。

设备侧验收按 `timer_run.md` 执行，真实结果产生后再更新本报告。
