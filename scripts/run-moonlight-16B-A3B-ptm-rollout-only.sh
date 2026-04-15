#!/usr/bin/env bash
set -euo pipefail

for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do
  unset "${proxy_var}" || true
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Clean up the previous local Ray / SGLang state for a fresh rollout-only run.
pkill -9 sglang || true
ray stop --force || true
pkill -9 ray || true
pkill -9 redis || true

export PYTHONBUFFERED=16

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l | tr -d ' ')"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

export SLIME_PTM_ROLLOUT_SGLANG_ROOT="${SLIME_PTM_ROLLOUT_SGLANG_ROOT:-/gfs/platform/public/infra/lxr/sglang}"
export SLIME_PTM_ROLLOUT_SGLANG_PYTHON_PATH="${SLIME_PTM_ROLLOUT_SGLANG_PYTHON_PATH:-${SLIME_PTM_ROLLOUT_SGLANG_ROOT}/python}"
export SLIME_PTM_ROLLOUT_MEGATRON_PATH="${SLIME_PTM_ROLLOUT_MEGATRON_PATH:-/root/Megatron-LM}"
export SLIME_PTM_ROLLOUT_MODEL_PATH="${SLIME_PTM_ROLLOUT_MODEL_PATH:-/gfs/platform/public/infra/Moonlight-16B-A3B-Instruct}"
export SLIME_PTM_ROLLOUT_REF_PATH="${SLIME_PTM_ROLLOUT_REF_PATH:-/gfs/platform/public/infra/Moonlight-16B-A3B-Instruct_torch_dist}"
export SLIME_PTM_ROLLOUT_PROMPT_DATA="${SLIME_PTM_ROLLOUT_PROMPT_DATA:-${REPO_DIR}/examples/ptm_long_context_agentic/glm47_flash_ptm_rollout_prompts.jsonl}"
export SLIME_PTM_ROLLOUT_DUMP_DIR="${SLIME_PTM_ROLLOUT_DUMP_DIR:-/tmp/moonlight16b_a3b_ptm_rollout}"
export SLIME_PTM_ROLLOUT_LD_LIBRARY_PATH="${SLIME_PTM_ROLLOUT_LD_LIBRARY_PATH:-/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}"
export LD_LIBRARY_PATH="${SLIME_PTM_ROLLOUT_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

export SLIME_PTM_ROLLOUT_NUM_GPUS="${SLIME_PTM_ROLLOUT_NUM_GPUS:-2}"
export SLIME_PTM_ROLLOUT_NUM_GPUS_PER_ENGINE="${SLIME_PTM_ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export SLIME_PTM_ROLLOUT_BATCH_SIZE="${SLIME_PTM_ROLLOUT_BATCH_SIZE:-4}"
export SLIME_PTM_ROLLOUT_NUM_SAMPLES="${SLIME_PTM_ROLLOUT_NUM_SAMPLES:-8}"
export SLIME_PTM_ROLLOUT_MAX_RESPONSE_LEN="${SLIME_PTM_ROLLOUT_MAX_RESPONSE_LEN:-2048}"
export SLIME_PTM_ROLLOUT_MAX_TOKENS_PER_GPU="${SLIME_PTM_ROLLOUT_MAX_TOKENS_PER_GPU:-4096}"
export SLIME_PTM_ROLLOUT_MEM_FRACTION="${SLIME_PTM_ROLLOUT_MEM_FRACTION:-0.6}"
export SLIME_PTM_ROLLOUT_CUDA_GRAPH_MAX_BS="${SLIME_PTM_ROLLOUT_CUDA_GRAPH_MAX_BS:-32}"
export SLIME_PTM_ROLLOUT_SEED="${SLIME_PTM_ROLLOUT_SEED:-42}"
export SLIME_PTM_ROLLOUT_TEMPERATURE="${SLIME_PTM_ROLLOUT_TEMPERATURE:-0.8}"
export SLIME_PTM_ROLLOUT_MAX_RUNNING_REQUESTS="${SLIME_PTM_ROLLOUT_MAX_RUNNING_REQUESTS:-64}"

mkdir -p "${SLIME_PTM_ROLLOUT_DUMP_DIR}"

source "${SCRIPT_DIR}/models/moonlight.sh"

CKPT_ARGS=(
  --hf-checkpoint "${SLIME_PTM_ROLLOUT_MODEL_PATH}"
  --ref-load "${SLIME_PTM_ROLLOUT_REF_PATH}"
  --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
  --prompt-data "${SLIME_PTM_ROLLOUT_PROMPT_DATA}"
  --input-key messages
  --label-key answer
  --apply-chat-template
  --rollout-shuffle
  --rm-type f1
  --num-rollout 1
  --rollout-batch-size "${SLIME_PTM_ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${SLIME_PTM_ROLLOUT_NUM_SAMPLES}"
  --rollout-max-response-len "${SLIME_PTM_ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${SLIME_PTM_ROLLOUT_TEMPERATURE}"
  --global-batch-size 32
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 2
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${SLIME_PTM_ROLLOUT_MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --eps-clip 0.2
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus "${SLIME_PTM_ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${SLIME_PTM_ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-tp-size "${SLIME_PTM_ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SLIME_PTM_ROLLOUT_MEM_FRACTION}"
  --sglang-cuda-graph-max-bs "${SLIME_PTM_ROLLOUT_CUDA_GRAPH_MAX_BS}"
  --sglang-max-running-requests "${SLIME_PTM_ROLLOUT_MAX_RUNNING_REQUESTS}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --moe-enable-deepep
  --moe-token-dispatcher-type flex
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${SLIME_PTM_ROLLOUT_NUM_GPUS}"
  --colocate
  --debug-rollout-only
  --save-debug-rollout-data "${SLIME_PTM_ROLLOUT_DUMP_DIR}/rollout_{rollout_id}.pt"
  --seed "${SLIME_PTM_ROLLOUT_SEED}"
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${SLIME_PTM_ROLLOUT_NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_PTM_ROLLOUT_SGLANG_PYTHON_PATH}:${SLIME_PTM_ROLLOUT_MEGATRON_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"LD_LIBRARY_PATH\": \"${SLIME_PTM_ROLLOUT_LD_LIBRARY_PATH}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"no_proxy\": \"127.0.0.1,localhost\",
    \"OTEL_SDK_DISABLED\": \"true\",
    \"OTEL_METRICS_EXPORTER\": \"none\",
    \"OTEL_TRACES_EXPORTER\": \"none\",
    \"OTEL_LOGS_EXPORTER\": \"none\"
  }
}"

ray job submit --address=\"http://127.0.0.1:8265\" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "$@"
