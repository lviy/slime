#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"

for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do
  unset "${proxy_var}" || true
done

python3 tests/test_qwen2.5_0.5B_ptm_no_grad_e2e_accuracy.py "$@"
