"""
Three-phase PTM E2E accuracy comparison:

1) Generate fixed rollout data once (debug-rollout-only).
2) Run train-only on the same rollout data with PTM OFF and save HF weights.
3) Run train-only on the same rollout data with PTM ON and save HF weights.

Finally, compare PTM OFF/ON saved HF weights and verify every tensor matches.
"""

from __future__ import annotations

import json
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
ATOL = float(os.environ.get("SLIME_PTM_E2E_ATOL", "0.0"))
RTOL = float(os.environ.get("SLIME_PTM_E2E_RTOL", "0.0"))
PTM_MIN_GROUP_SIZE = int(os.environ.get("SLIME_PTM_E2E_MIN_GROUP_SIZE", "2"))
PTM_PREFIX_MAX_LEN = os.environ.get("SLIME_PTM_E2E_PREFIX_MAX_LEN")


def _build_runtime_ld_library_path() -> str:
    configured = os.environ.get("SLIME_PTM_E2E_LD_LIBRARY_PATH")
    if configured:
        base_paths = configured.split(":")
    else:
        base_paths = [
            "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib",
            "/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
        ]

    merged_paths: list[str] = []
    for raw_path in [*base_paths, *os.environ.get("LD_LIBRARY_PATH", "").split(":")]:
        path = raw_path.strip()
        if not path or path in merged_paths or not os.path.isdir(path):
            continue
        merged_paths.append(path)

    return ":".join(merged_paths)


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
        "LD_LIBRARY_PATH": _build_runtime_ld_library_path(),
        "CUDNN_LOGERR_DBG": "1",
        "CUDNN_LOGDEST_DBG": "stderr",
        # This test only validates rollout/train correctness. Disable
        # OpenTelemetry exporters in Ray workers to avoid unrelated native
        # metrics threads crashing rollout-only startup.
        "OTEL_SDK_DISABLED": "true",
        "OTEL_METRICS_EXPORTER": "none",
        "OTEL_TRACES_EXPORTER": "none",
        "OTEL_LOGS_EXPORTER": "none",
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
    save_root = Path(debug_data_dir) / f"{tag}_megatron_ckpt"
    save_hf_root = Path(debug_data_dir) / f"{tag}_hf_{{rollout_id}}"
    phase_args = (
        f"{_common_args()} "
        f"--load-debug-rollout-data {debug_data_dir}/rollout_{{rollout_id}}.pt "
        f"--save {save_root} "
        "--save-interval 1 "
        "--no-save-optim "
        "--no-save-rng "
        f"--save-hf {save_hf_root} "
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


def _saved_hf_dirs(debug_data_dir: str, tag: str) -> dict[int, Path]:
    prefix = f"{tag}_hf_"
    saved_dirs: dict[int, Path] = {}
    for path in sorted(Path(debug_data_dir).glob(f"{tag}_hf_*")):
        if not path.is_dir():
            continue
        suffix = path.name[len(prefix) :]
        if not suffix.isdigit():
            continue
        saved_dirs[int(suffix)] = path
    if not saved_dirs:
        raise AssertionError(f"No saved HF model directories found for tag={tag} under {debug_data_dir}")
    return saved_dirs


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_safetensors_state_dict(model_dir: Path, filenames: list[str]) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "Loading PTM E2E HF checkpoints requires `safetensors` in the runtime environment."
        ) from exc

    state_dict: dict[str, torch.Tensor] = {}
    for filename in filenames:
        state_dict.update(load_file(str(model_dir / filename), device="cpu"))
    return state_dict


def _load_pytorch_state_dict(model_dir: Path, filenames: list[str]) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for filename in filenames:
        shard = torch.load(model_dir / filename, map_location="cpu", weights_only=False)
        if isinstance(shard, dict) and "state_dict" in shard and isinstance(shard["state_dict"], dict):
            shard = shard["state_dict"]
        if not isinstance(shard, dict):
            raise AssertionError(f"Unsupported PyTorch checkpoint shard format: {model_dir / filename}")

        tensor_items = {key: value for key, value in shard.items() if isinstance(value, torch.Tensor)}
        if len(tensor_items) != len(shard):
            non_tensor_keys = sorted(key for key, value in shard.items() if not isinstance(value, torch.Tensor))
            raise AssertionError(
                f"Expected only tensor entries in HF checkpoint shard {model_dir / filename}, "
                f"but found non-tensor keys: {non_tensor_keys}"
            )
        state_dict.update(tensor_items)
    return state_dict


def _load_saved_hf_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_index = model_dir / "model.safetensors.index.json"
    if safetensors_index.exists():
        index_data = _load_json(safetensors_index)
        filenames = sorted(set(index_data["weight_map"].values()))
        return _load_safetensors_state_dict(model_dir, filenames)

    safetensors_files = sorted(path.name for path in model_dir.glob("*.safetensors"))
    if safetensors_files:
        return _load_safetensors_state_dict(model_dir, safetensors_files)

    pytorch_index = model_dir / "pytorch_model.bin.index.json"
    if pytorch_index.exists():
        index_data = _load_json(pytorch_index)
        filenames = sorted(set(index_data["weight_map"].values()))
        return _load_pytorch_state_dict(model_dir, filenames)

    pytorch_files = sorted(path.name for path in model_dir.glob("pytorch_model*.bin"))
    if pytorch_files:
        return _load_pytorch_state_dict(model_dir, pytorch_files)

    raise AssertionError(f"Could not find supported HF weight files under {model_dir}")


def _compare_weight_tensors(
    off_tensor: torch.Tensor,
    on_tensor: torch.Tensor,
    *,
    rollout_id: int,
    weight_name: str,
) -> tuple[float, float]:
    if off_tensor.shape != on_tensor.shape:
        raise AssertionError(
            f"Shape mismatch for weight={weight_name} at rollout_id={rollout_id}: "
            f"{tuple(off_tensor.shape)} != {tuple(on_tensor.shape)}"
        )
    if off_tensor.dtype != on_tensor.dtype:
        raise AssertionError(
            f"Dtype mismatch for weight={weight_name} at rollout_id={rollout_id}: "
            f"{off_tensor.dtype} != {on_tensor.dtype}"
        )

    off_cpu = off_tensor.detach().cpu()
    on_cpu = on_tensor.detach().cpu()

    if not off_cpu.is_floating_point():
        if not torch.equal(off_cpu, on_cpu):
            mismatch_count = int((off_cpu != on_cpu).sum().item())
            raise AssertionError(
                f"Non-floating weight mismatch for weight={weight_name} at rollout_id={rollout_id}: "
                f"mismatch_count={mismatch_count}"
            )
        return 0.0, 0.0

    if torch.equal(off_cpu, on_cpu):
        return 0.0, 0.0

    diff = (off_cpu.float() - on_cpu.float()).abs()
    max_abs = diff.max().item() if diff.numel() > 0 else 0.0
    mean_abs = diff.mean().item() if diff.numel() > 0 else 0.0
    if not torch.allclose(off_cpu.float(), on_cpu.float(), rtol=RTOL, atol=ATOL):
        raise AssertionError(
            f"PTM weight mismatch for weight={weight_name} at rollout_id={rollout_id}: "
            f"max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, rtol={RTOL}, atol={ATOL}"
        )
    return max_abs, mean_abs


def compare_saved_weights(debug_data_dir: str) -> None:
    off_dirs = _saved_hf_dirs(debug_data_dir, "ptm_off")
    on_dirs = _saved_hf_dirs(debug_data_dir, "ptm_on")
    if set(off_dirs.keys()) != set(on_dirs.keys()):
        only_off = sorted(set(off_dirs.keys()) - set(on_dirs.keys()))
        only_on = sorted(set(on_dirs.keys()) - set(off_dirs.keys()))
        raise AssertionError(f"Saved HF rollout mismatch. only_off={only_off}, only_on={only_on}")

    global_max_abs = 0.0
    global_mean_abs_sum = 0.0
    global_weight_count = 0

    for rollout_id in sorted(off_dirs.keys()):
        off_state = _load_saved_hf_state_dict(off_dirs[rollout_id])
        on_state = _load_saved_hf_state_dict(on_dirs[rollout_id])
        if set(off_state.keys()) != set(on_state.keys()):
            only_off = sorted(set(off_state.keys()) - set(on_state.keys()))
            only_on = sorted(set(on_state.keys()) - set(off_state.keys()))
            raise AssertionError(
                f"Weight key mismatch at rollout_id={rollout_id}: "
                f"only_off={only_off[:10]}, only_on={only_on[:10]}"
            )

        for weight_name in sorted(off_state.keys()):
            max_abs, mean_abs = _compare_weight_tensors(
                off_state[weight_name],
                on_state[weight_name],
                rollout_id=rollout_id,
                weight_name=weight_name,
            )
            global_max_abs = max(global_max_abs, max_abs)
            global_mean_abs_sum += mean_abs
            global_weight_count += 1

    if global_weight_count == 0:
        raise AssertionError("No HF weights were compared between PTM OFF and PTM ON runs.")

    global_mean_abs = global_mean_abs_sum / global_weight_count
    print("=" * 80)
    print("PTM E2E weight accuracy PASSED")
    print(f"Compared rollout_ids: {sorted(off_dirs.keys())}")
    print(f"Compared weights: {global_weight_count}")
    print(f"Global max_abs_diff={global_max_abs:.6e}, global mean_abs_diff={global_mean_abs:.6e}")
    print(f"Thresholds: rtol={RTOL}, atol={ATOL}")
    print("=" * 80)


def execute() -> None:
    _validate_runtime_gpus()
    debug_data_dir = tempfile.mkdtemp(prefix="slime_ptm_e2e_")
    print(f"Using temp dir: {debug_data_dir}")

    execute_rollout_only(debug_data_dir)
    execute_train_only(debug_data_dir, ptm_enabled=False, tag="ptm_off")
    execute_train_only(debug_data_dir, ptm_enabled=True, tag="ptm_on")
    compare_saved_weights(debug_data_dir)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
