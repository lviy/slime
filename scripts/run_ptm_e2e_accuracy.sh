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

python3 tests/test_qwen2.5_0.5B_ptm_e2e_accuracy.py "$@"
