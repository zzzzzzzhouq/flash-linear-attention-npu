# GDN A5 阶段计时运行说明

## 构建与安装

在仓库根目录、Ascend950 环境执行。四个目标必须一起构建：三个子 kernel 和新的融合 Profile host 入口。普通 Stable ABI 不需要切换分支或替换调用方式。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu \
  --ops=chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_fwd_prepare,chunk_fwd_h,chunk_fwd_o

./build_out/fla_npu_linux-*.run --install

(
  cd torch_custom/fla_npu
  python3 setup.py bdist_wheel
)
WHEEL_PATH="$(ls -t torch_custom/fla_npu/dist/fla_npu-*.whl | head -n 1)"
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"
```

安装后重启 Python 进程，确保加载本轮 `libcust_opapi.so` 和 Python wrapper。

## 采集

以下命令仍从仓库根目录执行。普通／诊断调用使用同一组输入，脚本要求全部非空输出逐元素完全一致。

```bash
DRIVER_DIR="$PWD/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/tests/ut"

python3 "$DRIVER_DIR/profile_gdn.py" \
  --device 0 \
  --batch 1 \
  --tokens 512 \
  --k-heads 16 \
  --v-heads 32 \
  --k-dim 128 \
  --v-dim 128 \
  --out profiling_results/gdn_stage_timer_512
```

覆盖可选 initial-state cast 阶段时增加 `--initial-state`。大 case 例如：

```bash
python3 "$DRIVER_DIR/profile_gdn.py" \
  --device 0 \
  --batch 1 \
  --tokens 11264 \
  --k-heads 16 \
  --v-heads 32 \
  --k-dim 128 \
  --v-dim 128 \
  --out profiling_results/gdn_stage_timer_11264
```

输出：

- `rank_0_gdn_timer.npy`：形状为 `[3, TIMER_SLICE_NUMEL]` 的原始 int64 数据；
- `rank_0_{prepare,fwd_h,fwd_o}_timer_parsed.csv`：每核每阶段 start/end/duration；
- `metadata.json`：输入参数、布局常量、输出对照与文件索引。

## 生成 MindStudio／Chrome trace

只有在时钟频率和同一 kernel 内 AIC/AIV 共同时钟均已确认时，才增加 trace 选项。将 `<verified_cycles_per_us>` 换成已核实数值：

```bash
python3 "$DRIVER_DIR/profile_gdn.py" \
  --device 0 \
  --tokens 512 \
  --k-heads 16 \
  --v-heads 32 \
  --cycles-per-us <verified_cycles_per_us> \
  --shared-clock \
  --out profiling_results/gdn_stage_timer_trace
```

生成 `rank_0_prepare_timer.json`、`rank_0_fwd_h_timer.json` 和 `rank_0_fwd_o_timer.json`。三个文件使用各自 kernel 的零点；不要把三次 launch 的 cycle 直接拼成一条绝对时间轴。

## 离线重放

```bash
python3 "$DRIVER_DIR/parse_timer_csv.py" \
  profiling_results/gdn_stage_timer_512/rank_0_gdn_timer.npy \
  --kernel prepare \
  --out profiling_results/gdn_stage_timer_512/rank_0_prepare_timer_parsed.csv
```

`--kernel` 可改为 `fwd_h` 或 `fwd_o`。解析会拒绝负时长、半写记录、非法 iter 数和落在本核 `KERNEL_TIMING` 之外的事件。
