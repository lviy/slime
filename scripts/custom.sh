#!/bin/bash

LOG_DIR="/root/dump"
mkdir -p "${LOG_DIR}"
LOG_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/log_${LOG_TIME}.log"
exec > "${LOG_FILE}" 2>&1

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT=1
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# 确保能找到 models/moonlight.sh，假设脚本在 scripts 目录下运行
source "${SCRIPT_DIR}/models/moonlight.sh"
# /gfs/platform/public/infra/cp_slime/scripts/models/qwen3-30B-A3B.sh
# /gfs/platform/public/infra/cp_slime/scripts/run-glm4.5-355B-A32B.sh
# /gfs/platform/public/infra/slime/scripts/models/qwen3-30B-A3B.sh
# /gfs/platform/public/infra/cp_slime/scripts/models/moonlight.sh
CKPT_ARGS=(
   --hf-checkpoint /gfs/platform/public/infra/Moonlight-16B-A3B-Instruct
   # 下面这些在 debug-rollout-only 模式下可能不需要，但保留作为参考
   #--ref-load /root/models/Moonlight-16B-A3B-Instruct_torch_dist
   #--load /root/checkpoints/Moonlight-16B-A3B-Instruct-GRPO
   # --save /root/checkpoints/Moonlight-16B-A3B-Instruct-GRPO
   #--save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "/gfs/platform/public/infra/cp_slime/examples/ten_samples.jsonl"
   --input-key messages
   --label-key answer
   --apply-chat-template
   --rm-type math
   --num-rollout 1
   --rollout-batch-size 512
   --n-samples-per-prompt 1
   --rollout-max-response-len 5120
   --rollout-temperature 0.7
   --rollout-seed 42

   --over-sampling-batch-size 512
   
   --num-steps-per-rollout 4
   #--balance-data   
   
)

EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192

   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.85
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256) 512
   --sglang-enable-deterministic-inference
   --sglang-data-parallel-size 1
   --sglang-expert-parallel-size 4
   --sglang-enable-eplb
   --sglang-moe-a2a-backend none
   --sglang-deepep-mode normal
   --sglang-eplb-rebalance-num-iterations 1540
   --sglang-eplb-algorithm auto
   --sglang-ep-dispatch-algorithm static
   --sglang-eplb-min-rebalancing-utilization-threshold 1.0
   --sglang-ep-num-redundant-experts 0
   # == 串行执行 ==
   --sglang-server-concurrency 1

)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
   --seed 42
   
   # === Requested Debug Flags === DEBUG eplb
   --debug-rollout-only
   # --use-rollout-routing-replay
   --use-slime-router
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ../train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
