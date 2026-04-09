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
export TENSORBOARD_DIR="/afs/chatrl/users/wxe/slime/tensorboard_logs/telechatmoe-105B/$(date +%Y-%m-%d_%H-%M-%S)"
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

export val_data_path="/afs/chatrl/users/hxh/data/slime_rl_data/math_verify_aime2024_sample32_no_prompt.jsonl"
export train_data_path="/afs/chatrl/users/hxh/data/slime_rl_data/olympiads_sky_processed_model_judge_step1834_1119.jsonl"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/telechat3-moe.sh"

CKPT_ARGS=(
   --hf-checkpoint /afs/chatrl/public/models/telechat-moe/telechat-moe-105b-step7500
   --ref-load /afs/chatrl/public/models/telechat-moe/telechat-moe-105b-step7500_torch_dist
   --load /afs/chatrl/public/models/telechat-moe/telechat-moe-105b-step7500
   --save /afs/chatrl/public/models/telechat-moe/telechat-moe-105b-step7500-test
   --save-interval 4
)

ROLLOUT_ARGS=(
   --prompt-data ${train_data_path}
   --input-key messages
   --label-key answer
   --apply-chat-template
   # --rollout-shuffle
   --rm-type dapo
   --num-rollout 3000
   --rollout-batch-size 256
   --n-samples-per-prompt 16
   --rollout-max-prompt-len 2048
   --rollout-max-response-len 30720
   --rollout-temperature 1

   --over-sampling-batch-size 4096
   --sglang-server-concurrency 1024
#    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --num-steps-per-rollout 8
   # --global-batch-size 256
   --balance-data   
)

EVAL_ARGS=(
   --eval-interval 4
   --eval-prompt-data ${val_data_path}
   --eval-input-key messages
   --eval-label-key answer
   --eval-temperature 1
   --eval-top-p 1
   # --eval-reward-key score
   --eval-max-response-len 2048
   --eval-max-context-len 30720
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --decoder-first-pipeline-num-layers 11
   --decoder-last-pipeline-num-layers 12

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-tis
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 3e-4
   --eps-clip-high 4e-4
   --eps-clip-c 10
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
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group moomlight-16B-A3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-dp-size 4
   --sglang-ep-size 8
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   --sglang-enable-eplb
   --sglang-eplb-rebalance-num-iterations 1000

   # --sglang-enable-ep-moe

   # dp attention
   --sglang-enable-dp-attention
   # --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
)

REMOTE_RM_ARGS=(
   --rm-api-key EMPTY
   --rm-base-url http://app-042eb891fec6475c9935b5b3b836aebb.ns-bjdianxin-cb517126.svc.cluster.local:6669/v1
   --rm-model-name Qwen3-30B-A3B
   --custom-rm-path slime.rollout.rm_hub.remote_reward_model.remote_reward_function
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   # --attention-backend flash

   # use deepep for megatron
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# ray start --head \
#     --node-ip-address "127.0.0.1" \
#     --num-gpus 8 \
#     --disable-usage-stats \
#     --dashboard-host=0.0.0.0 \
#     --dashboard-port=8265
if [ "$RANK" -eq 0 ]; then
   ray start --head \
        --node-ip-address "127.0.0.1" \
        --num-gpus 8 \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265
   # if [ "$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX" -eq 0 ]; then
   # Build the runtime environment JSON with proper variable substitution
   RUNTIME_ENV_JSON="{
   \"env_vars\": {
      \"PYTHONPATH\": \"/root/Megatron-LM/\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
      \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
      \"SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK\":\"True\",
      \"SGLANG_LOGITS_PROCESSOR_CHUNK_SIZE\":\"2048\"
   }
   }"
      # \"SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK\":\"True\",
      # \"SGLANG_LOGITS_PROCESSOR_CHUNK_SIZE\":\"1024\"
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train.py \
      --actor-num-nodes 4 \
      --actor-num-gpus-per-node 8 \
      --colocate \
      --reward-key score \
      --use-tensorboard \
      --save-debug-rollout-data ./debug_32k_new/exp1_rollout_{rollout_id}.pt \
      ${MODEL_ARGS[@]} \
      ${CKPT_ARGS[@]} \
      ${ROLLOUT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${GRPO_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${EVAL_ARGS[@]} \
      ${SGLANG_ARGS[@]} \
      ${MISC_ARGS[@]} \
      ${REMOTE_RM_ARGS[@]}

else
   ray start --address "${MASTER_ADDR}:6379" --num-gpus 8 --disable-usage-stats
fi
