# 自研 PoolingSFA 算子模型整网替换接入指导（A3 单节点）

> 适用对象：局点部署工程师
> 适用硬件：**仅 Atlas A3 系列（`ascend910_93`）**
> 适用模型：GLM-5.2 W8A8C8（带 `index_topk` 的稀疏注意力模型）
> 参考流程：vLLM-Ascend 官方 GLM-5.2/5.3 部署指南（A3 系列页）
> 文档版本：v3.0（2026-10）

**整体顺序**：

```text
前提条件 → 下载权重 → 准备容器 → 检查/安装 MemFabric
→ 部署单节点基线模型 → 基线功能验证
→ 编译算子包 → 替换算子包 → 启用 PoolingSFA 配置
→ 重启并确认命中 → 精度/性能验证 → 回退
```

> 单节点示例用于**功能打通与验证**：PoolingSFA 用 `keep_device_kv_cache=true`，不产生容量收益。生产环境的多节点 PD 部署见第 12 节备注。
>
> 基线模型验证通过之前，**不要**动算子包和模型配置。先证明官方镜像能把模型跑起来，再替换。

---

## 0. 变量约定

每个新 shell 先执行一次，按实际环境替换尖括号内容：

```bash
export MODEL_PATH=/data/models/GLM-5.2-w8a8c8
export REPO=/home/<user>/vllm-ascend
export CHECK=/home/<user>/pooling_sfa_check.py
export PORT=8000
export LOG=$HOME/single.log
```

> 以下命令均在 **A3 容器内、bash** 下执行；宿主机操作会单独标注。

---

## 1. 前提条件

### 1.1 硬件

| 项 | 要求 |
|---|---|
| 节点 | 1 个 A3 节点（128GB × 8 起） |
| 设备 | 容器暴露 16 个逻辑设备（`davinci0`–`davinci15`） |
| 上下文 | 单节点示例 `--max-model-len 32768`（更长上下文需多节点） |

### 1.2 软件与依赖

| 项 | 要求 | 检查命令 |
|---|---|---|
| 镜像 | 官方 `quay.io/ascend/vllm-ascend:{tag}` | `docker images \| grep vllm-ascend` |
| CANN | 与镜像匹配 | `echo $ASCEND_HOME_PATH` |
| Python | 与镜像一致（通常 3.11） | `python3 -V` |
| MemFabric | 存在且 `offload` 可用（本算子 CPU KV 池依赖） | 见步骤 3 |
| fastokens | 已安装（官方 A3 配置使用） | `python3 -c "import fastokens; print('ok')"` |
| 编译器/构建工具 | `clang++`、`cmake`、`ninja` | `which clang++ cmake ninja` |
| 锁页内存 | 建议 ≥ 数 GB | `ulimit -l; ulimit -Hl` |

### 1.3 模型权重

| 权重版本 | 硬件 | 下载地址 |
|---|---|---|
| `GLM-5.2-w8a8c8` | 1 个 A3 节点（128GB × 8）起 | <https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8c8> |

---

## 步骤 1：下载模型权重

**目标**：把权重下载到单节点的本地目录，并核对文件完整。

**操作 1**：安装 ModelScope 客户端。

**命令**：

```bash
python3 -m pip install modelscope
```

**操作 2**：下载权重。

**命令**：

```bash
python3 -c "from modelscope import snapshot_download; snapshot_download('Eco-Tech/GLM-5.2-w8a8c8', local_dir='$MODEL_PATH')"
```

**操作 3**：核对权重。

**命令**：

```bash
ls $MODEL_PATH | head -20
du -sh $MODEL_PATH
python3 -c "import json; c=json.load(open('$MODEL_PATH/config.json')); t=c.get('text_config',c); print('model_type=',c.get('model_type')); print('index_topk=',t.get('index_topk')); print('compress_ratios=',t.get('compress_ratios'))"
```

**预期结果**：

- 目录中有 `config.json`、`*.safetensors`、tokenizer 等文件；
- `index_topk` 非 `None`；
- `compress_ratios` 为 `None`（若存在，本算子不支持，停止）。

---

## 步骤 2：准备容器

**目标**：在 A3 节点启动带 16 个设备的容器。

**操作 1**：在**宿主机**拉取镜像。

**命令**：

```bash
docker pull quay.io/ascend/vllm-ascend:{tag}
```

**操作 2**：在**宿主机**启动容器。

**命令**：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{tag}
export NAME=vllm-ascend

docker run -itd --rm \
  --name $NAME \
  --net=host \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci8 \
  --device /dev/davinci9 \
  --device /dev/davinci10 \
  --device /dev/davinci11 \
  --device /dev/davinci12 \
  --device /dev/davinci13 \
  --device /dev/davinci14 \
  --device /dev/davinci15 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v /data/models:/data/models \
  $IMAGE bash
```

**操作 3**：进入容器并确认设备。

**命令**：

```bash
docker exec -it vllm-ascend bash -lc "npu-smi info | head -20; echo CONTAINER_OK"
```

**预期结果**：能看到 16 个设备，容器内 `$MODEL_PATH` 可访问。

> 若模型目录不在 `/data/models`，同步修改上面的挂载路径。

---

## 步骤 3：检查并安装 MemFabric

**目标**：确认 `memfabric_hybrid.offload` 可用；不满足则安装/升级。

**操作 1**：检查安装路径与环境脚本。

**命令**：

```bash
ls -l /usr/local/memfabric_hybrid/set_env.sh
python3 -m pip show memfabric_hybrid memcache_hybrid | grep -E 'Name|Version|Location'
```

**操作 2**：检查 `offload` 与关键字段。

**命令**：

```bash
python3 -c "import torch; from memfabric_hybrid import offload; c=offload.OffloadConfig(); f=[n for n in dir(c) if not n.startswith('_')]; print('fields=', f); print('ok=', all(x in f for x in ['device_id','size','world_size','rank_id']))"
```

**预期结果**：

```text
fields= ['device_id', 'size', 'world_size', 'rank_id', ...]
ok= True
```

**操作 3**：若检查失败，安装/升级 MemFabric。

> 示例版本 `1.1.4` 为参考环境验证版本；实际以交付包/镜像说明为准。

**命令**：

```bash
python3 -m pip freeze | grep -E '^(torch|torch-npu|torchvision|torchaudio)==' > /tmp/torch-constraints.txt

python3 -m pip install --upgrade 'memfabric-hybrid==1.1.4' 'memcache-hybrid==1.1.4' \
  -c /tmp/torch-constraints.txt \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

**操作 4**：加载环境并复验。

**命令**：

```bash
source /usr/local/memfabric_hybrid/set_env.sh
python3 -c "import torch; from memfabric_hybrid import offload; c=offload.OffloadConfig(); f=[n for n in dir(c) if not n.startswith('_')]; print('fields=', f); print('ok=', all(x in f for x in ['device_id','size','world_size','rank_id']))"
echo "MEMFABRIC=$MEMFABRIC_HYBRID_EXTEND_LIB_PATH"
```

**预期结果**：`ok= True`，`MEMFABRIC_HYBRID_EXTEND_LIB_PATH` 非空。

> 若 `set_env.sh` 不存在：优先使用交付的带 MemFabric 镜像；或按交付包说明设置 `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` 指向 MemFabric 扩展库目录。不要跳过此步，否则启动会报 `offload launch load library failed`。

---

## 步骤 4：部署单节点基线模型

**目标**：用官方镜像把 GLM-5.2 的 DP1/TP16 服务跑起来，**暂不使用 PoolingSFA**。

**操作 1**：在容器内创建 `start_single.sh`。

**文件内容**：

```bash
#!/usr/bin/env bash

export MODEL_PATH=/data/models/GLM-5.2-w8a8c8

export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-server-count 1 \
  --data-parallel-size 1 \
  --tensor-parallel-size 16 \
  --seed 1024 \
  --served-model-name glm \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --enable-auto-tool-choice \
  --max-num-seqs 12 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-cache-dtype int8 \
  --attention_config.indexer_kv_dtype int8 \
  --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
```

**操作 2**：启动基线服务（同时备份原始脚本，供回退）。

**命令**：

```bash
chmod +x start_single.sh
mkdir -p /tmp/poolingsfa
cp -a start_single.sh /tmp/poolingsfa/start_single.sh.baseline
bash start_single.sh 2>&1 | tee $LOG
```

**预期结果**：日志出现模型加载完成、服务监听 `0.0.0.0:8000`。

**操作 3**：确认服务就绪。

**命令**：

```bash
curl -s http://127.0.0.1:8000/v1/models | head -c 300; echo
```

**预期结果**：返回包含 `"glm"` 的模型列表。

---

## 步骤 5：基线功能验证（替换前必须做）

**目标**：确认官方镜像 + 单节点基线服务正常。此步**不涉及** PoolingSFA。

**操作 1**：发一条固定请求。

**命令**：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"用一句话解释注意力机制"}],"max_tokens":32,"temperature":0}'
```

**预期结果**：HTTP 200，返回 JSON 中包含 `choices[0].message.content`，内容非空、语义正常。

**操作 2**：用自检脚本记录基线。

**命令**：

```bash
python3 $CHECK service --base-url http://127.0.0.1:8000 --model glm --prompt "用一句话解释注意力机制" --max-tokens 32 | tee baseline_smoke.txt
```

**预期结果**：`FAIL 0`。

**操作 3**（建议）：保存基线批量结果，供替换后 A/B。

**命令**：

```bash
python3 $CHECK service --base-url http://127.0.0.1:8000 --model glm --prompt-file prompts.txt --output baseline.jsonl --max-tokens 64 --temperature 0
```

**预期结果**：生成 `baseline.jsonl`。若还没有评测集，先准备 `prompts.txt`（每行一条，建议 ≥100 条）。

**操作 4**（建议）：保存基线性能。

**命令**：

```bash
vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8000 --model glm --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 100 --request-rate inf > baseline_perf.txt 2>&1; tail -20 baseline_perf.txt
```

> 基线验证通过后，再进入算子包编译/替换。若基线本身不通，先排查模型部署（权重、量化和显存），不要怀疑算子。

---

## 步骤 6：编译算子包

**目标**：在 A3 环境编译 PoolingSFA 的 Python 扩展 + CANN kernel。

**前提**：编译环境的 Python ABI、torch/torch_npu、CANN 必须与运行环境一致。建议在宿主机另起一个**同镜像容器**编译，避免影响正在运行的服务；构建容器需含 `cmake`、`ninja`、`clang++`。

**操作 1**：把算子源码放到容器内（`$REPO`），核对版本。

**命令**：

```bash
cd $REPO
git log --oneline -1
python3 -V
python3 -c "import torch, torch_npu, vllm; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('vllm', vllm.__version__)"
```

**预期结果**：源码存在，Python 版本与运行环境一致（通常 3.11），torch/torch_npu/vllm 版本与官方镜像一致。

**操作 2**：加载 CANN 编译环境。

**命令**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
```

**操作 3**：清理旧构建（增量编译不可靠，必须全量）。

**命令**：

```bash
cd $REPO
rm -rf build csrc/build
export MAX_JOBS=32
```

**操作 4**：编译。

**命令**：

```bash
cd $REPO
python3 setup.py build_ext --inplace
```

**预期结果**：

- 退出码 0；
- `$REPO/vllm_ascend/` 下生成 `vllm_ascend_C*.so`；
- `$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/fused_sparse_attention_overlap/` 下有 `.o` 文件。

**操作 5**：记录构建产物指纹。

**命令**：

```bash
ls -lh $REPO/vllm_ascend/vllm_ascend_C*.so
mkdir -p /tmp/poolingsfa
md5sum $REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/fused_sparse_attention_overlap/*.o | tee /tmp/poolingsfa/kernel_md5.txt
```

**预期结果**：`.so` 非空；kernel `.o` 的 md5 已记录。

> 首次编译约 10~30 分钟。合入官仓后，构建入口以官仓的 `build.sh` / `setup.py` 为准；本文的 `build_ext --inplace` 用于现场验证。

---

## 步骤 7：替换算子包

**目标**：让服务加载本仓编译出的 `vllm_ascend` 与 kernel vendor 包。

**操作 1**：停止基线服务。

**命令**：

```bash
pkill -f 'vllm serve' || true
sleep 3
npu-smi info | head -20
```

**预期结果**：无残留服务进程，显存释放。

**操作 2**：在 `start_single.sh` 的 `exec vllm serve` 之前加入两行。

**内容**：

```bash
export PYTHONPATH="$REPO:${PYTHONPATH}"
source $REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
```

> 作用：让 Python 优先加载本仓的 `vllm_ascend`，并让 CANN 找到本仓的 kernel vendor 包。

**操作 3**：确认加载路径。

**命令**：

```bash
python3 -c "import vllm_ascend,os; print(os.path.realpath(os.path.dirname(vllm_ascend.__file__)))"
```

**预期结果**：输出 `$REPO/vllm_ascend`，不是 `site-packages`。

**操作 4**：记录回退信息。

**命令**：

```bash
mkdir -p /tmp/poolingsfa
ls -l /tmp/poolingsfa/start_single.sh.baseline
md5sum $REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/fused_sparse_attention_overlap/*.o > /tmp/poolingsfa/kernel_md5.txt
```

**预期结果**：基线脚本备份存在（步骤 4 已创建），md5 文件生成。回退时恢复该备份即可。

---

## 步骤 8：启用 PoolingSFA 配置

**目标**：只改 `start_single.sh`，把注意力切到融合 offload 路径（单节点用 `keep_device_kv_cache=true` 做功能验证）。

**操作 1**：确认脚本没有不兼容开关。

**命令**：

```bash
grep -c "enable_dsa_cp" start_single.sh
grep -c "enable_sparse_sfa_c8" start_single.sh
```

**预期结果**：`1`、`0`（第一条是官方基线的 `enable_dsa_cp`，下一步删除；`enable_sparse_sfa_c8` 应为 0）。删除后重新执行本命令，应为 `0`、`0`。

**操作 2**：修改 `--additional-config`：

- 删除 `enable_dsa_cp`；
- 追加 `kv_offload_decode_config`，单节点调试用 `keep_device_kv_cache=true`。

改前：

```json
{"enable_dsa_cp": true, "enable_balance_scheduling": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true}
```

改后：

```json
{"enable_balance_scheduling": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true, "kv_offload_decode_config": {"enabled": true, "use_fused_overlap": true, "topk_buffer_size": 4096, "dram_size_per_dp_GB": 128, "keep_device_kv_cache": true}}
```

**参数取值**：

| 参数 | 取值 |
|---|---|
| `topk_buffer_size` | ≥ 步骤 1 读到的 `index_topk`，且是 128 的整数倍；例如 2048 → 4096 |
| `dram_size_per_dp_GB` | 先填 128；启动后按日志 `planned=... GiB` 调整；必须 ≥ planned |
| `keep_device_kv_cache` | 单节点调试填 `true`；生产 PD 填 `false` |

**操作 3**：校验配置。

**命令**：

```bash
python3 -c "import re,json; s=open('start_single.sh').read(); m=re.search(r\"--additional-config\s+'([^']*)'\", s); assert m, 'missing --additional-config'; c=json.loads(m.group(1)); k=c.get('kv_offload_decode_config') or {}; assert k.get('enabled') is True, 'enabled!=true'; assert k.get('use_fused_overlap') is True, 'use_fused_overlap!=true'; assert 'enable_dsa_cp' not in c, 'enable_dsa_cp present'; assert not c.get('enable_sparse_sfa_c8', False), 'enable_sparse_sfa_c8 on'; print('CONFIG_OK', k)"
```

**预期结果**：

```text
CONFIG_OK {'enabled': True, 'use_fused_overlap': True, 'topk_buffer_size': 4096, 'dram_size_per_dp_GB': 128, 'keep_device_kv_cache': True}
```

---

## 步骤 9：重启并确认算子命中

**操作 1**：重启服务。

**命令**：

```bash
bash start_single.sh 2>&1 | tee -a $LOG
```

**操作 2**：检查启动日志中的 CPU 池初始化。

**命令**：

```bash
grep -E "KVOffloadDecodeManager starts|Traceback|NotImplementedError|offload launch" $LOG | tail -20
```

**预期结果**：

```text
KVOffloadDecodeManager starts CPU KV pool initialization: planned=... GiB, configured_limit=... GiB, num_blocks=...
```

且没有 `Traceback` / `NotImplementedError` / `offload launch load library failed`。

**操作 3**：发一次请求（触发 decode）并检查命中。

**命令**：

```bash
python3 $CHECK service --base-url http://127.0.0.1:8000 --model glm --prompt "用一句话解释注意力机制" --max-tokens 32 --log-file $LOG
```

**预期结果**：输出包含 `[PASS] 融合算子命中标记`，日志中出现 `[fused_overlap_offload][decode]`。

**操作 4**：如未命中，手工确认。

**命令**：

```bash
grep "\[fused_overlap_offload\]\[decode\]" $LOG | tail -3
python3 $CHECK env --model-config $MODEL_PATH/config.json | tail -20
```

**预期结果**：至少 3 行融合标记；`env` 中 `custom_op=True`、算子 schema 为 15 位置参数版。

---

## 步骤 10：精度与性能验证

### 10.1 算子级精度

**操作**：运行随版本提供的融合算子精度用例（合入官仓后位于 `tests/ut/attention/` 下；若交付版本未包含该文件，以 10.2 的模型级验证为准）。

**命令**：

```bash
cd $REPO
pytest -sv tests/ut/attention/a2/test_fused_sparse_attention_overlap_precision.py
```

**预期结果**：全部通过；信号相对误差 ≤1e-2。

### 10.2 整网精度（A/B）

**操作 1**：确认步骤 5 已保存 `baseline.jsonl`。

**操作 2**：对 PoolingSFA 服务跑同一批请求。

**命令**：

```bash
python3 $CHECK service --base-url http://127.0.0.1:8000 --model glm --prompt-file prompts.txt --output fused.jsonl --max-tokens 64 --temperature 0
```

**操作 3**：对比。

**命令**：

```bash
python3 $CHECK compare --file-a baseline.jsonl --file-b fused.jsonl --report accuracy.json
```

**预期结果**：

```text
[PASS] 条数一致 | 100
[PASS] 完全一致率 | 0.99xx（阈值 0.99）
[PASS] 相似度低于 0.95 的条数 | 0
```

### 10.3 性能

**操作 1**：对 PoolingSFA 服务做压测。

**命令**：

```bash
vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8000 --model glm --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 100 --request-rate inf > fused_perf.txt 2>&1; tail -20 fused_perf.txt
```

**操作 2**：与步骤 5 保存的 `baseline_perf.txt` 对比。

**命令**：

```bash
grep -E "TTFT|TPOT|Throughput|Request throughput|Output token throughput" baseline_perf.txt fused_perf.txt
```

**预期结果**：TTFT/TPOT/吞吐不劣于 baseline；长稳 ≥1 小时无内存持续增长、无超时。

> 单节点只验证功能与相对趋势；容量收益需在生产 PD 环境验证。

---

## 步骤 11：回退

**操作 1**：恢复基线启动脚本。

**命令**：

```bash
cp -a /tmp/poolingsfa/start_single.sh.baseline start_single.sh
pkill -f 'vllm serve' || true
sleep 3
bash start_single.sh 2>&1 | tee $LOG
```

**操作 2**：确认 baseline 恢复。

**命令**：

```bash
python3 $CHECK service --base-url http://127.0.0.1:8000 --model glm --prompt "你好" --max-tokens 16
```

**预期结果**：请求成功；日志中不再新增 `[fused_overlap_offload][decode]`。

---

## 12. 多节点 PD 生产部署（备注）

单节点示例只用于功能验证。生产环境需要使用 **PD 分离**，在 Decode 节点启用本算子，才能获得容量收益。多节点部署的完整步骤以官方指南 **A3 系列 → PD 分离** 页为准：

<https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html>

与单节点流程相比，只有以下差异：

| 项 | 单节点验证 | 多节点生产 |
|---|---|---|
| 部署方式 | DP1/TP16 单引擎 | A3 PD：P0/P1（TP16/PP2）+ D0/D1（DP16/TP2） |
| 算子生效节点 | 本节点 | 仅 D 节点（Prefill 保持官方基线） |
| `keep_device_kv_cache` | `true` | `false` |
| 启动脚本 | `start_single.sh` | 官方 P/D 脚本；在 D 脚本中做步骤 7/8 的同样改动 |
| 网络要求 | 无 | 需验证多节点通信、Mooncake 连接器 |
| MemFabric | 本节点 | 每个 D 节点都要检查/安装 |

> 多节点时，把本文步骤 3（MemFabric）、步骤 6（编译）、步骤 7（PYTHONPATH/vendor）、步骤 8（配置）原样应用到 D 节点；Prefill 不要改。PD 脚本、代理和连接器配置全部沿用官方原文。

---

## 13. 常见问题排查

| 症状 | 根因 | 处理 |
|---|---|---|
| 基线服务起不来 | 权重/量化/显存问题 | 先按步骤 5 排查，确认 `/v1/models` 正常 |
| `MemFabric` 导入失败 | 未安装、版本过旧或未 source | 按步骤 3 检查/安装；`source /usr/local/memfabric_hybrid/set_env.sh` |
| `offload launch load library failed` | MemFabric 环境未加载 | 同上一行 |
| `offload.empty failed` | `ulimit -l` 锁页内存不足 | 提高 `ulimit -l`（同时检查宿主机限制） |
| `custom ops will be disabled` | `VLLM_BATCH_INVARIANT=1` 或扩展加载失败 | 关闭 batch invariant；检查 `vllm_ascend_C*.so` |
| `fused_overlap offload requires custom op ...` | 包未替换成功 / 加载到旧包 | 按步骤 7 操作 3 确认 `vllm_ascend.__file__` 指向本仓 |
| `KV Offload Decode only support sparse attention model` | 模型无 `index_topk` | 模型不适用 |
| `KV Offload Decode don't support compress now` | 模型有 `compress_ratios` | 模型不适用 |
| `KV Offload Decode is only supported in PD ...` | 单节点未开 `keep_device_kv_cache` | 单节点调试填 `true`；生产用 PD |
| `topk_buffer_size must be >= topk` / `divisible by block_size` | 容量配置不合法 | 调大并保证 128 倍数 |
| `requires TP without context parallelism` | 脚本含 `enable_dsa_cp` 或 CP size>1 | 删除 `enable_dsa_cp` |
| `does not support the sparse SFA C8 main cache` | `enable_sparse_sfa_c8=true` | 关闭 |
| `requires one TopK head` | TopK 头数≠1 | 使用单 TopK 头模型/配置 |
| 日志无 `[fused_overlap_offload][decode]` | 只跑 prefill / `use_fused_overlap=false` / 配置未生效 | 发 decode 请求；按步骤 8 操作 3 重新校验 |
| `Failed to stat ...<hash>.o` | kernel 与配置 json/扩展版本不一致 | 重装同源整套包 |
| 启动很慢 / JIT 编译失败 | 缺 clang++、CANN 头文件 | 补齐编译环境；首次构建预留时间 |
| 性能不如预期 | 单节点本身无容量收益；或实际走了非融合 | 单节点只做功能验证；生产看 PD 的命中日志与 A/B |

**现场取证**：

```bash
export VLLM_ASCEND_SFA_DUMP_DIR=/tmp/sfa_dump
export VLLM_ASCEND_SFA_DUMP_LAYER=0
export VLLM_ASCEND_SFA_DUMP_STEP=0
```

重启并发一次 decode 请求后，算子输入/输出以 `.pt` 落盘，可离线复算。

---

## 14. 常用命令速查

```bash
# 基线健康检查
curl -s http://127.0.0.1:8000/v1/models | head -c 300

# 基线单请求
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"temperature":0}'

# MemFabric 检查
python3 -c "import torch; from memfabric_hybrid import offload; c=offload.OffloadConfig(); print([n for n in dir(c) if not n.startswith('_')])"

# 确认加载的是本仓包
python3 -c "import vllm_ascend,os; print(os.path.realpath(os.path.dirname(vllm_ascend.__file__)))"

# 确认 custom op 与 schema
python3 -c "import torch, vllm_ascend.vllm_ascend_C; from vllm_ascend.utils import enable_custom_op; print(enable_custom_op()); print(torch.ops._C_ascend.npu_fused_sparse_attention_overlap.default._schema)"

# 环境自检
python3 $CHECK env --model-config $MODEL_PATH/config.json

# 融合命中日志
grep "\[fused_overlap_offload\]\[decode\]" $LOG | tail -3

# 单请求 + 命中检查
python3 $CHECK service --base-url http://127.0.0.1:8000 --model glm --prompt "你好" --max-tokens 32 --log-file $LOG

# 整网精度对比
python3 $CHECK compare --file-a baseline.jsonl --file-b fused.jsonl --report accuracy.json

# 算子级精度（若用例随版本提供）
cd $REPO && pytest -sv tests/ut/attention/a2/test_fused_sparse_attention_overlap_precision.py

# kernel 指纹
md5sum $REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/fused_sparse_attention_overlap/*.o
```

---

## 15. 附录

### A. PoolingSFA 算子契约摘要

```text
npu_fused_sparse_attention_overlap(
    Tensor query,
    Tensor(a!) selection_k_rope, Tensor(b!) selection_kv_cache,
    Tensor(c!) selection_kv_block_table, Tensor(d!) selection_kv_block_status,
    Tensor(e!) selection_membership_map,
    Tensor selection_topk_indices,
    Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table,
    Tensor full_kv_actual_seq, Tensor full_q_actual_seq,
    float scale_value, int sparse_block_size=1, int selection_topk_block_size=1,
    *, layout_query="TND", layout_kv="PA_BSND", sparse_mode=3
) -> Tensor
```

- `selection_*` 五个张量是副作用输入，算子原地更新；
- 单 TopK 头；KV block size 128；`layout_query="TND"`、`layout_kv="PA_BSND"`、`sparse_mode=3`。

### B. 关键配置项

| 配置 | 位置 | 说明 |
|---|---|---|
| `kv_offload_decode_config.*` | `--additional-config` | 本文核心开关 |
| `enable_dsa_cp` | `--additional-config` | 必须不存在 |
| `enable_sparse_sfa_c8` | `--additional-config` | 必须关闭 |
| `VLLM_BATCH_INVARIANT` | 环境变量 | 必须未开启 |
| `VLLM_USE_V2_MODEL_RUNNER` | 环境变量 | 官方 A3 配置未设置，保持默认 |
| `VLLM_ASCEND_SFA_DUMP_*` | 环境变量 | 算子级取证 |

### C. 交付记录模板

```text
局点：            ________
日期：            ________
硬件/SoC：        A3 / ascend910_93
镜像 tag：        ________
权重：            ________
MemFabric 版本：  ________
基线验证：        PASS / FAIL
算子包构建 commit：________
kernel .o md5：   ________
配置：            topk_buffer_size=____, dram_size_per_dp_GB=____, keep_device_kv_cache=____
L0/L2/L3/L4：     ________
遗留问题：        ________
```

### D. 参考

- GLM-5.2/5.3 官方部署指南（使用 **A3 系列** 页）：<https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html>
- 多节点通信验证：<https://docs.vllm.ai/projects/ascend/zh-cn/latest/getting_started/installation.html#installation-multi-node-interconnect>
