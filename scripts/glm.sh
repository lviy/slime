#!/bin/bash
# 强清理：杀掉所有可能残留的进程
pkill -9 sglang || true
pkill -9 ray || true  
pkill -9 python || true
pkill -9 redis || true
sleep 5
pkill -9 ray || true
pkill -9 python || true

# 清理 Ray 状态
ray stop --force || true
sleep 3

# 多机 NCCL 配置
export NCCL_IB_HCA="mlx5_gdr_0,mlx5_gdr_1,mlx5_gdr_2,mlx5_gdr_3,mlx5_gdr_4,mlx5_gdr_5,mlx5_gdr_6,mlx5_gdr_7"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_GRAPH_REGISTER=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# 升级 transformers 以支持 glm4_moe_lite 模型类型
pip install -U transformers --break-system-packages
# 升级sgl-kernel
pip install -U --force-reinstall sglang-kernel
# 减少显存碎片
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ========================
# 模型路径
# ========================
export model_load_path=${model_load_path:-"/gfs/platform/public/infra/Moonlight-16B-A3B-Instruct"}
export torch_dist_model_load_path=/gfs/platform/public/infra/Moonlight-16B-A3B-Instruct_torch_dist
export root_dir="/gfs/platform/public/infra/lxr/slime"
export save_dir="/gfs/platform/public/infra/lxr/checkpoint"
export DEBUG_SGLANG_ROOT="${root_dir}/third_party/sglang_pod_20260322"
export LOG_ROOT_DIR="/gfs/platform/public/infra/lxr/logs/megatron"
export RUNTIME_LOG_DIR="${LOG_ROOT_DIR}/runtime_logs"
mkdir -p "${save_dir}" "${LOG_ROOT_DIR}" "${RUNTIME_LOG_DIR}"
export PYTHONPATH="/gfs/platform/public/infra/lxr/sglang/python:${PYTHONPATH}"
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_LOGITS_PROCESSOR_CHUNK_SIZE=2048
export SGLANG_ENABLE_LOGITS_PROCESSOR_CHUNK=True

# ========================
# 训练超参数
# ========================
export ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
export MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-4096}
MAX_LEN_K=$(( MAX_RESPONSE_LEN / 1024 ))k

export lr=${lr:-5e-7}
export rollout_batch_size=${rollout_batch_size:-16}
export train_prompt_batch_size=${train_prompt_batch_size:-8}  # 每步更新用多少条 prompt
export n_samples=${n_samples:-8}
export global_batch_size=$(( train_prompt_batch_size * n_samples ))   
export num_steps_per_rollout=$(( rollout_batch_size / train_prompt_batch_size ))  

export eval_interval=${eval_interval:-4}
export over_sampling_batch_size=${over_sampling_batch_size:-256}

export eps_clip=${eps_clip:-3e-4}
export eps_clip_high=${eps_clip_high:-4e-4}

# ========================
# SGLang 稳定性参数（关键修改）
# ========================
export sglang_server_concurrency=${sglang_server_concurrency:-32}
export sglang_max_running_requests=${sglang_max_running_requests:-32}

echo "✅ 稳定版配置: Nodes=${ACTOR_NUM_NODES}, MaxLen=${MAX_LEN_K}, lr=${lr}"
echo "   rollout_bs=${rollout_batch_size}, train_prompt_bs=${train_prompt_batch_size}, n_samples=${n_samples}"
echo "   sglang_concurrency=${sglang_server_concurrency}, sglang_max_running=${sglang_max_running_requests}"

set -ex
export model_prefix_name="${model_prefix_name:-$(date +%Y%m%d-%H%M)}"
export model_suffix_name=${model_suffix_name:-"${ACTOR_NUM_NODES}node-${MAX_LEN_K}"}

if [ "$train_from_resume" = "True" ]; then
    export model_save_name=$(basename "$(dirname "$model_load_path")")
    export model_save_path=${save_dir}/Moonlight-PTM/${model_save_name}
else
   export model_save_name=${model_prefix_name}-gspo-lr${lr}-bs${rollout_batch_size}-tpbs${train_prompt_batch_size}-n${n_samples}-epsclip${eps_clip}-high${eps_clip_high}
   export model_save_path=${save_dir}/Moonlight-PTM/${model_save_name}-${model_suffix_name}
fi
echo "✅ Model will be saved to: ${model_save_path}"

export TENSORBOARD_DIR=${model_save_path}
export LOG_FILE=${LOG_ROOT_DIR}/train_log_$(date +%Y-%m-%d_%H-%M-%S).log
export DEBUG_DUMP_DIR=${DEBUG_DUMP_DIR:-"${model_save_path}/debug"}
mkdir -p "${DEBUG_DUMP_DIR}"

if [ ! -d "$model_save_path" ]; then
   mkdir -p "$model_save_path"
   echo "已创建目录: $model_save_path"
else
   echo "目录已存在: $model_save_path"
fi

export save_interval=${save_interval:-4}

export val_data_path=${val_data_path:-"/gfs/space/chatrl/users/hxh/data/rule_based_rl/AIME-2024/data/AIME_DAPO_math.jsonl"}
export train_data_path=${train_data_path:-"/gfs/space/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k_dedup.jsonl"}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# 加载 Moonlight-16B-A3B 模型配置
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/moonlight.sh"

CKPT_ARGS=(
   --hf-checkpoint ${model_load_path}
   --ref-load ${torch_dist_model_load_path}
   --load ${model_load_path}
)

ROLLOUT_ARGS=(
   --prompt-data ${train_data_path}
   --input-key messages
   --label-key answer
   --apply-chat-template
   --rm-type dapo
   --num-rollout 1
   --rollout-batch-size ${rollout_batch_size}
   --n-samples-per-prompt ${n_samples}
   --rollout-max-prompt-len 2048
   --rollout-max-response-len ${MAX_RESPONSE_LEN}
   --rollout-temperature 1

   --over-sampling-batch-size ${over_sampling_batch_size}
   --sglang-server-concurrency ${sglang_server_concurrency}

   --num-steps-per-rollout ${num_steps_per_rollout}
   --global-batch-size ${global_batch_size}
)

EVAL_ARGS=(
   --eval-interval ${eval_interval}
   --eval-prompt-data ${val_data_path}
   --eval-input-key messages
   --eval-label-key answer
   --eval-temperature 1
   --eval-top-p 1
   --eval-max-response-len ${MAX_RESPONSE_LEN}
   --eval-max-prompt-len 2048
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 2
   --expert-tensor-parallel-size 1
   --sequence-parallel

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-tis
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip ${eps_clip}
   --eps-clip-high ${eps_clip_high}
   --eps-clip-c 10
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${lr}
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
   # --wandb-group glm4.7-flash-dapo
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.5
   --sglang-tp-size 2
   --sglang-cuda-graph-max-bs 32
   --sglang-max-running-requests ${sglang_max_running_requests}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
)

PTM_ARGS=()
_ptm_enable=$(echo "${PTM_ENABLE:-0}" | tr '[:upper:]' '[:lower:]')
if [[ "${_ptm_enable}" == "1" || "${_ptm_enable}" == "true" || "${_ptm_enable}" == "yes" ]]; then
   PTM_ARGS+=(--slime-prefix-tree-merging)
   PTM_ARGS+=(--slime-prefix-min-group-size "${PTM_MIN_GROUP_SIZE:-2}")
   if [[ -n "${PTM_PREFIX_MAX_LEN:-}" ]]; then
      PTM_ARGS+=(--slime-prefix-max-len "${PTM_PREFIX_MAX_LEN}")
   fi
fi

EXTRA_TRAIN_ARGS_ARRAY=()
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
   # shellcheck disable=SC2206
   EXTRA_TRAIN_ARGS_ARRAY=(${EXTRA_TRAIN_ARGS})
fi

export WORLD_SIZE=${ACTOR_NUM_NODES}
export MASTER_ADDR=${GEMINI_IP_taskrole1_0:-"127.0.0.1"}
export RANK=${GEMINI_TASK_INDEX:-0}
# if [ "$RANK" -eq 0 ]; then
#    export NODE_LOG_FILE=${RUNTIME_LOG_DIR}/node_rank${RANK}_${RUN_TS}.log
#    exec > >(tee -a "${NODE_LOG_FILE}") 2>&1
# else
#    exec >/dev/null 2>&1
# fi

if [ "$RANK" -eq 0 ]; then
   ray start --head \
        --node-ip-address "127.0.0.1" \
        --num-gpus 8 \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

   RUNTIME_ENV_JSON="{
   \"env_vars\": {
      \"PYTHONPATH\": \"/gfs/platform/public/infra/lxr/sglang/python:/gfs/platform/public/infra/lxr/Megatron-LM\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
      \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
      \"SGLANG_ENABLE_LOGITS_PROCESSOR_CHUNK\":\"True\",
      \"SGLANG_LOGITS_PROCESSOR_CHUNK_SIZE\":\"2048\",
      \"PYTORCH_ALLOC_CONF\":\"expandable_segments:True\"
   }
   }"

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train.py \
      --actor-num-nodes ${ACTOR_NUM_NODES} \
      --actor-num-gpus-per-node 2 \
      --colocate \
      --use-fault-tolerance \
      --reward-key score \
      --use-tensorboard \
      --rollout-health-check-interval 10 \
      --rollout-health-check-timeout 30 \
      --rollout-health-check-first-wait 300 \
      --save-debug-rollout-data ${DEBUG_DUMP_DIR}/rollout_{rollout_id}.pt \
      --save-debug-train-data ${DEBUG_DUMP_DIR}/train_{rollout_id}_{rank}.pt \
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
      ${PTM_ARGS[@]} \
      ${EXTRA_TRAIN_ARGS_ARRAY[@]} \
      2>&1 | tee "${LOG_FILE}"

else
   # 对于其余节点，加入 Ray worker 集群
   ray start --address="${MASTER_ADDR}:6379" --num-gpus 2 --disable-usage-stats --block
fi




