"""
Three-phase PTM no-grad E2E accuracy comparison:

1) Generate fixed rollout data once (debug-rollout-only).
2) Run train-only on the same rollout data with PTM OFF and dump train data.
3) Run train-only on the same rollout data with PTM ON and dump train data.

Finally, compare no-grad outputs (default: log_probs/ref_log_probs) between PTM OFF/ON.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

import slime.utils.external_utils.command_utils as U

TIGHT_DEVICE_MEMORY = U.get_bool_env_var("SLIME_TEST_TIGHT_DEVICE_MEMORY", "1")

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
MODEL_ROOT = os.environ.get("SLIME_PTM_E2E_MODEL_ROOT", "/root/models")
MODEL_PATH = os.environ.get("SLIME_PTM_E2E_MODEL_PATH", f"{MODEL_ROOT}/{MODEL_NAME}")
SGLANG_ROOT = os.environ.get("SLIME_PTM_E2E_SGLANG_ROOT", "/gfs/platform/public/infra/lxr/sglang")
SGLANG_PYTHON_PATH = os.environ.get("SLIME_PTM_E2E_SGLANG_PYTHON_PATH", f"{SGLANG_ROOT}/python")
MEGATRON_PATH = os.environ.get("SLIME_PTM_E2E_MEGATRON_PATH", "/root/Megatron-LM")
RUNTIME_PYTHONPATH = os.environ.get("SLIME_PTM_E2E_PYTHONPATH", f"{SGLANG_PYTHON_PATH}:{MEGATRON_PATH}")
_DETECTED_CUDA_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
NUM_GPUS = int(os.environ.get("SLIME_PTM_E2E_NUM_GPUS", str(max(_DETECTED_CUDA_GPUS, 1))))
NUM_ROLLOUT = int(os.environ.get("SLIME_PTM_E2E_NUM_ROLLOUT", "1"))

COMPARE_KEYS = tuple(
    key.strip()
    for key in os.environ.get("SLIME_PTM_E2E_COMPARE_KEYS", "log_probs").split(",")
    if key.strip()
)
STRICT_COMPARE_KEYS = os.environ.get("SLIME_PTM_E2E_STRICT_COMPARE_KEYS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
ATOL = float(os.environ.get("SLIME_PTM_E2E_ATOL", "1e-6"))
RTOL = float(os.environ.get("SLIME_PTM_E2E_RTOL", "1e-6"))
PTM_MIN_GROUP_SIZE = int(os.environ.get("SLIME_PTM_E2E_MIN_GROUP_SIZE", "2"))
PTM_PREFIX_MAX_LEN = os.environ.get("SLIME_PTM_E2E_PREFIX_MAX_LEN")


def prepare() -> None:
    U.exec_command(f"mkdir -p {MODEL_ROOT} /root/datasets")
    model_path = Path(MODEL_PATH)
    if model_path.exists() and any(model_path.iterdir()):
        print(f"Skip model download since model path already exists: {MODEL_PATH}")
    else:
        U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_PATH}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def _validate_runtime_gpus() -> None:
    if _DETECTED_CUDA_GPUS <= 0:
        raise RuntimeError(
            "No CUDA GPU detected in current process. "
            "Please run in a GPU environment or set SLIME_PTM_E2E_NUM_GPUS explicitly after checking device visibility."
        )
    if NUM_GPUS > _DETECTED_CUDA_GPUS:
        raise RuntimeError(
            f"Configured NUM_GPUS={NUM_GPUS} is larger than detected CUDA GPUs={_DETECTED_CUDA_GPUS}. "
            "Set SLIME_PTM_E2E_NUM_GPUS to a valid value (for example, the count from `nvidia-smi -L | wc -l`)."
        )
    print(
        f"PTM E2E runtime config: detected_cuda_gpus={_DETECTED_CUDA_GPUS}, "
        f"num_gpus={NUM_GPUS}, model_path={MODEL_PATH}"
    )


def _runtime_env_vars() -> dict[str, str]:
    return {
        "PYTHONPATH": RUNTIME_PYTHONPATH,
    }


def _common_args() -> str:
    ckpt_args = f"--hf-checkpoint {MODEL_PATH}/ " f"--ref-load {MODEL_PATH}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = "--advantage-estimator grpo " "--eps-clip 0.2 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--megatron-to-hf-mode bridge "
        "--seed 42 "
    )

    return f"{ckpt_args} " f"{rollout_args} " f"{optimizer_args} " f"{grpo_args} " f"{perf_args} " f"{misc_args} "


def _ptm_args() -> str:
    args = f"--slime-prefix-tree-merging --slime-prefix-min-group-size {PTM_MIN_GROUP_SIZE} "
    if PTM_PREFIX_MAX_LEN:
        args += f"--slime-prefix-max-len {PTM_PREFIX_MAX_LEN} "
    return args


def execute_rollout_only(debug_data_dir: str) -> None:
    # Generate rollout files once. Keep this phase PTM-off so that PTM ON train
    # path validates fallback metadata construction from fixed rollout tokens.
    sglang_args = (
        f"--rollout-num-gpus {NUM_GPUS} "
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {0.6 if TIGHT_DEVICE_MEMORY else 0.7} "
        "--sglang-cuda-graph-max-bs 32 "
    )
    phase_args = (
        f"{_common_args()} "
        f"{sglang_args} "
        "--debug-rollout-only "
        f"--save-debug-rollout-data {debug_data_dir}/rollout_{{rollout_id}}.pt "
    )
    print("=" * 80)
    print("Phase 1: debug-rollout-only (generate fixed rollout data)")
    print("=" * 80)
    U.execute_train(
        train_args=phase_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars=_runtime_env_vars(),
    )


def execute_train_only(debug_data_dir: str, *, ptm_enabled: bool, tag: str) -> None:
    phase_args = (
        f"{_common_args()} "
        f"--load-debug-rollout-data {debug_data_dir}/rollout_{{rollout_id}}.pt "
        f"--save-debug-train-data {debug_data_dir}/{tag}_train_{{rollout_id}}_{{rank}}.pt "
        "--ci-test "
    )
    if ptm_enabled:
        phase_args += _ptm_args()

    print("=" * 80)
    print(f"Phase {'3' if ptm_enabled else '2'}: train-only ({'PTM ON' if ptm_enabled else 'PTM OFF'})")
    print("=" * 80)
    U.execute_train(
        train_args=phase_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars=_runtime_env_vars(),
    )


def _load_dump_index(debug_data_dir: str, tag: str) -> dict[tuple[int, int], dict]:
    index: dict[tuple[int, int], dict] = {}
    for path in sorted(Path(debug_data_dir).glob(f"{tag}_train_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rollout_id = int(payload["rollout_id"])
        rank = int(payload["rank"])
        index[(rollout_id, rank)] = payload["rollout_data"]
    if not index:
        raise AssertionError(f"No train dump files found for tag={tag} under {debug_data_dir}")
    return index


def _compare_tensor_lists(
    off_vals: list[torch.Tensor],
    on_vals: list[torch.Tensor],
    *,
    key: str,
    rollout_id: int,
    rank: int,
) -> tuple[float, float]:
    if len(off_vals) != len(on_vals):
        raise AssertionError(
            f"Length mismatch for key={key} at rollout_id={rollout_id}, rank={rank}: "
            f"{len(off_vals)} != {len(on_vals)}"
        )

    local_max_abs = 0.0
    local_mean_abs_sum = 0.0
    local_mean_abs_count = 0
    for i, (off_t, on_t) in enumerate(zip(off_vals, on_vals, strict=True)):
        if not isinstance(off_t, torch.Tensor) or not isinstance(on_t, torch.Tensor):
            raise AssertionError(
                f"Expected tensor list for key={key}, got {type(off_t)} and {type(on_t)} "
                f"at rollout_id={rollout_id}, rank={rank}, idx={i}"
            )
        if off_t.shape != on_t.shape:
            raise AssertionError(
                f"Shape mismatch for key={key} at rollout_id={rollout_id}, rank={rank}, idx={i}: "
                f"{tuple(off_t.shape)} != {tuple(on_t.shape)}"
            )

        off_fp = off_t.detach().float().cpu()
        on_fp = on_t.detach().float().cpu()
        if not torch.allclose(off_fp, on_fp, rtol=RTOL, atol=ATOL):
            diff = (off_fp - on_fp).abs()
            raise AssertionError(
                f"PTM accuracy mismatch for key={key} at rollout_id={rollout_id}, rank={rank}, idx={i}: "
                f"max_abs={diff.max().item():.6e}, mean_abs={diff.mean().item():.6e}, "
                f"rtol={RTOL}, atol={ATOL}"
            )

        diff = (off_fp - on_fp).abs()
        local_max_abs = max(local_max_abs, diff.max().item())
        local_mean_abs_sum += diff.mean().item()
        local_mean_abs_count += 1

    local_mean_abs = local_mean_abs_sum / max(local_mean_abs_count, 1)
    return local_max_abs, local_mean_abs


def compare_no_grad_outputs(debug_data_dir: str) -> None:
    off_index = _load_dump_index(debug_data_dir, "ptm_off")
    on_index = _load_dump_index(debug_data_dir, "ptm_on")
    if set(off_index.keys()) != set(on_index.keys()):
        only_off = sorted(set(off_index.keys()) - set(on_index.keys()))
        only_on = sorted(set(on_index.keys()) - set(off_index.keys()))
        raise AssertionError(f"Dump key mismatch. only_off={only_off}, only_on={only_on}")

    global_max_abs = 0.0
    global_mean_abs_sum = 0.0
    global_mean_abs_count = 0
    skipped_keys: set[str] = set()
    first_pair_keys = None

    for key_tuple in sorted(off_index.keys()):
        rollout_id, rank = key_tuple
        off_data = off_index[key_tuple]
        on_data = on_index[key_tuple]
        if first_pair_keys is None:
            first_pair_keys = (sorted(off_data.keys()), sorted(on_data.keys()))

        for cmp_key in COMPARE_KEYS:
            if cmp_key not in off_data or cmp_key not in on_data:
                if STRICT_COMPARE_KEYS:
                    if cmp_key not in off_data:
                        raise AssertionError(
                            f"Missing key={cmp_key} in PTM OFF dump for rollout_id={rollout_id}, rank={rank}"
                        )
                    raise AssertionError(
                        f"Missing key={cmp_key} in PTM ON dump for rollout_id={rollout_id}, rank={rank}"
                    )
                skipped_keys.add(cmp_key)
                continue

            off_vals = off_data[cmp_key]
            on_vals = on_data[cmp_key]
            if not isinstance(off_vals, (list, tuple)) or not isinstance(on_vals, (list, tuple)):
                raise AssertionError(
                    f"Only list/tuple tensor values are supported for compare key={cmp_key}, "
                    f"got {type(off_vals)} and {type(on_vals)}"
                )

            max_abs, mean_abs = _compare_tensor_lists(
                list(off_vals),
                list(on_vals),
                key=cmp_key,
                rollout_id=rollout_id,
                rank=rank,
            )
            global_max_abs = max(global_max_abs, max_abs)
            global_mean_abs_sum += mean_abs
            global_mean_abs_count += 1

    if global_mean_abs_count == 0:
        off_keys = first_pair_keys[0] if first_pair_keys is not None else []
        on_keys = first_pair_keys[1] if first_pair_keys is not None else []
        raise AssertionError(
            "No comparable keys found in PTM OFF/ON dumps. "
            f"requested_keys={COMPARE_KEYS}, off_keys_sample={off_keys}, on_keys_sample={on_keys}"
        )

    global_mean_abs = global_mean_abs_sum / max(global_mean_abs_count, 1)
    print("=" * 80)
    print("PTM no-grad E2E accuracy PASSED")
    print(f"Compared keys: {COMPARE_KEYS}")
    if skipped_keys:
        print(f"Skipped missing keys (non-strict mode): {sorted(skipped_keys)}")
    print(f"Global max_abs_diff={global_max_abs:.6e}, global mean_abs_diff={global_mean_abs:.6e}")
    print(f"Thresholds: rtol={RTOL}, atol={ATOL}")
    print("=" * 80)


def execute() -> None:
    _validate_runtime_gpus()
    debug_data_dir = tempfile.mkdtemp(prefix="slime_ptm_nograd_e2e_")
    print(f"Using temp dir: {debug_data_dir}")

    execute_rollout_only(debug_data_dir)
    execute_train_only(debug_data_dir, ptm_enabled=False, tag="ptm_off")
    execute_train_only(debug_data_dir, ptm_enabled=True, tag="ptm_on")
    compare_no_grad_outputs(debug_data_dir)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
