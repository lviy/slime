#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"

for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do
  unset "${proxy_var}" || true
done

export SLIME_PTM_E2E_SGLANG_ROOT="${SLIME_PTM_E2E_SGLANG_ROOT:-/gfs/platform/public/infra/lxr/sglang}"
export SLIME_PTM_E2E_SGLANG_PYTHON_PATH="${SLIME_PTM_E2E_SGLANG_PYTHON_PATH:-${SLIME_PTM_E2E_SGLANG_ROOT}/python}"
export SLIME_PTM_E2E_LD_LIBRARY_PATH="${SLIME_PTM_E2E_LD_LIBRARY_PATH:-/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}"
export LD_LIBRARY_PATH="${SLIME_PTM_E2E_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

if [[ -z "${SLIME_PTM_E2E_NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    _detected_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -n "${_detected_gpus}" && "${_detected_gpus}" -gt 0 ]]; then
      export SLIME_PTM_E2E_NUM_GPUS="${_detected_gpus}"
    fi
  fi
fi

if [[ $# -eq 0 && -z "${SLIME_PTM_FORWARD_SPEED_ROLLOUT_PT:-}" ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/run_ptm_forward_speed.sh --rollout-pt /path/to/rollout_0.pt [extra args]
  scripts/run_ptm_forward_speed.sh --rollout-pt '/path/to/rollout_{rollout_id}.pt' --num-rollout 2

Environment fallback:
  SLIME_PTM_FORWARD_SPEED_ROLLOUT_PT=/path/to/rollout_0.pt scripts/run_ptm_forward_speed.sh [extra args]
EOF
  exit 2
fi

declare -a cmd=(python3 tests/test_ptm_forward_speed.py)
if [[ -n "${SLIME_PTM_FORWARD_SPEED_ROLLOUT_PT:-}" ]]; then
  cmd+=(--rollout-pt "${SLIME_PTM_FORWARD_SPEED_ROLLOUT_PT}")
fi
cmd+=("$@")

"${cmd[@]}"
