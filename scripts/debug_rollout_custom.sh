#!/bin/bash

# for rerun the task
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

CKPT_ARGS=(
   --hf-checkpoint /root/.cache/modelscope/hub/models/moonshotai/Moonlight-16B-A3B-Instruct
   # 下面这些在 debug-rollout-only 模式下可能不需要，但保留作为参考
   --ref-load /root/models/Moonlight-16B-A3B-Instruct_torch_dist
   --load /root/checkpoints/Moonlight-16B-A3B-Instruct-GRPO
   # --save /root/checkpoints/Moonlight-16B-A3B-Instruct-GRPO
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /home/lviy/dapo-math-17k_dedup_no_prompt_math_verify.parquet
   --input-key messages
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type f1
   --num-rollout 3000
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --over-sampling-batch-size 256
   
   --num-steps-per-rollout 4
   --balance-data   
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
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-eplb-algorithm auto
   --sglang-eplb-min-rebalancing-utilization-threshold 1.0
   --sglang-ep-num-redundant-experts 0
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
   
   # === Requested Debug Flags ===
   --debug-rollout-only
   --use-rollout-routing-replay
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
