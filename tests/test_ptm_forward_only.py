"""
Run only the PTM ON train-only phase using pre-generated rollout .pt data.

Compared with `test_qwen2.5_0.5B_ptm_no_grad_e2e_accuracy.py`, this script:
1) Skips rollout generation entirely.
2) Loads rollout data from a user-provided `.pt` path or template.
3) Runs only the PTM ON train-only phase and optionally dumps train data.

Examples:
  python3 tests/test_ptm_forward_only.py \
      --rollout-pt /tmp/rollout_0.pt

  python3 tests/test_ptm_forward_only.py \
      --rollout-pt '/tmp/rollout_{rollout_id}.pt' \
      --num-rollout 2 \
      --save-dir /tmp/ptm_phase3

  python3 tests/test_ptm_forward_only.py \
      --model qwen3-4B \
      --rollout-pt /tmp/rollout_0.pt

  python3 tests/test_ptm_forward_only.py \
      --rollout-pt /tmp/rollout_0.pt \
      --model-type glm4.7-30B-A3B \
      --model-path /root/GLM-4.7-Flash \
      --ref-load /root/GLM-4.7-Flash_torch_dist \
      --megatron-to-hf-mode raw
"""

from __future__ import annotations

import os
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import torch

import slime.utils.external_utils.command_utils as U

MODEL_ROOT = os.environ.get("SLIME_PTM_E2E_MODEL_ROOT", "/root/models")
SGLANG_ROOT = os.environ.get("SLIME_PTM_E2E_SGLANG_ROOT", "/gfs/platform/public/infra/lxr/sglang")
SGLANG_PYTHON_PATH = os.environ.get("SLIME_PTM_E2E_SGLANG_PYTHON_PATH", f"{SGLANG_ROOT}/python")
MEGATRON_PATH = os.environ.get("SLIME_PTM_E2E_MEGATRON_PATH", "/root/Megatron-LM")
RUNTIME_PYTHONPATH = os.environ.get("SLIME_PTM_E2E_PYTHONPATH", f"{SGLANG_PYTHON_PATH}:{MEGATRON_PATH}")
_DETECTED_CUDA_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEFAULT_NUM_GPUS = int(os.environ.get("SLIME_PTM_E2E_NUM_GPUS", str(max(_DETECTED_CUDA_GPUS, 1))))
DEFAULT_NUM_ROLLOUT = int(os.environ.get("SLIME_PTM_E2E_NUM_ROLLOUT", "1"))
PTM_MIN_GROUP_SIZE = int(os.environ.get("SLIME_PTM_E2E_MIN_GROUP_SIZE", "2"))
PTM_PREFIX_MAX_LEN = os.environ.get("SLIME_PTM_E2E_PREFIX_MAX_LEN")
DEFAULT_TENSOR_MODEL_PARALLEL_SIZE = int(os.environ.get("SLIME_PTM_E2E_TENSOR_MODEL_PARALLEL_SIZE", "1"))
DEFAULT_PIPELINE_MODEL_PARALLEL_SIZE = int(os.environ.get("SLIME_PTM_E2E_PIPELINE_MODEL_PARALLEL_SIZE", "1"))
DEFAULT_CONTEXT_PARALLEL_SIZE = int(os.environ.get("SLIME_PTM_E2E_CONTEXT_PARALLEL_SIZE", "1"))
DEFAULT_EXPERT_MODEL_PARALLEL_SIZE = int(os.environ.get("SLIME_PTM_E2E_EXPERT_MODEL_PARALLEL_SIZE", "1"))
DEFAULT_EXPERT_TENSOR_PARALLEL_SIZE = int(os.environ.get("SLIME_PTM_E2E_EXPERT_TENSOR_PARALLEL_SIZE", "1"))
DEFAULT_MAX_TOKENS_PER_GPU = int(os.environ.get("SLIME_PTM_E2E_MAX_TOKENS_PER_GPU", "4096"))
DEFAULT_DECODER_LAST_PIPELINE_NUM_LAYERS = os.environ.get("SLIME_PTM_E2E_DECODER_LAST_PIPELINE_NUM_LAYERS")

MODEL_PRESETS = {
    "glm4.7-flash": {
        "model_name": "GLM-4.7-Flash",
        "model_type": "glm4.7-30B-A3B",
        "download_model_id": None,
    },
    "qwen2.5-0.5B": {
        "model_name": "Qwen2.5-0.5B-Instruct",
        "model_type": "qwen2.5-0.5B",
        "download_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    "qwen3-4B": {
        "model_name": "Qwen3-4B",
        "model_type": "qwen3-4B",
        "download_model_id": "Qwen/Qwen3-4B",
    },
    "qwen3-8B": {
        "model_name": "Qwen3-8B",
        "model_type": "qwen3-8B",
        "download_model_id": "Qwen/Qwen3-8B",
    },
}


def _resolve_model_config(
    model: str, model_name: str | None, model_type: str | None, model_path: str | None
) -> dict[str, str | None]:
    preset = MODEL_PRESETS.get(model)
    if preset is None:
        raise ValueError(
            f"Unsupported --model {model}. Available presets: {sorted(MODEL_PRESETS.keys())}"
        )

    resolved_model_name = model_name or preset["model_name"]
    resolved_model_type = model_type or preset["model_type"]
    resolved_model_path = model_path or os.environ.get(
        "SLIME_PTM_E2E_MODEL_PATH",
        f"{MODEL_ROOT}/{resolved_model_name}",
    )
    return {
        "model_name": resolved_model_name,
        "model_type": resolved_model_type,
        "model_path": resolved_model_path,
        "download_model_id": preset["download_model_id"],
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-0.5B",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model preset to use. Default: qwen2.5-0.5B.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional override for HuggingFace checkpoint directory/model name.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Optional override for Slime/Megatron model type.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional override for the full local model checkpoint path.",
    )
    parser.add_argument(
        "--ref-load",
        type=str,
        default=None,
        help=(
            "Optional override for the Megatron checkpoint directory used by --ref-load. "
            "Required for models that should run with --megatron-to-hf-mode=raw."
        ),
    )
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="bridge",
        help=(
            "How to initialize/update HF weights. "
            "Use `bridge` for models supported by megatron.bridge, "
            "and `raw` when using a converted Megatron torch_dist checkpoint."
        ),
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_MODEL_PARALLEL_SIZE,
        help=f"Tensor parallel size. Default: {DEFAULT_TENSOR_MODEL_PARALLEL_SIZE}.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=DEFAULT_PIPELINE_MODEL_PARALLEL_SIZE,
        help=f"Pipeline parallel size. Default: {DEFAULT_PIPELINE_MODEL_PARALLEL_SIZE}.",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=DEFAULT_CONTEXT_PARALLEL_SIZE,
        help=f"Context parallel size. PTM currently requires 1. Default: {DEFAULT_CONTEXT_PARALLEL_SIZE}.",
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=DEFAULT_EXPERT_MODEL_PARALLEL_SIZE,
        help=f"Expert parallel size. Default: {DEFAULT_EXPERT_MODEL_PARALLEL_SIZE}.",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=DEFAULT_EXPERT_TENSOR_PARALLEL_SIZE,
        help=f"Expert tensor parallel size. Default: {DEFAULT_EXPERT_TENSOR_PARALLEL_SIZE}.",
    )
    parser.add_argument(
        "--max-tokens-per-gpu",
        type=int,
        default=DEFAULT_MAX_TOKENS_PER_GPU,
        help=f"Max tokens per GPU for dynamic batching. Default: {DEFAULT_MAX_TOKENS_PER_GPU}.",
    )
    parser.add_argument(
        "--decoder-last-pipeline-num-layers",
        type=int,
        default=(
            int(DEFAULT_DECODER_LAST_PIPELINE_NUM_LAYERS)
            if DEFAULT_DECODER_LAST_PIPELINE_NUM_LAYERS is not None
            else None
        ),
        help=(
            "Optional Megatron pipeline split hint. "
            "Leave unset for PP=1."
        ),
    )
    parser.add_argument(
        "--rollout-pt",
        required=True,
        help=(
            "Path to a saved rollout .pt file, or a template containing "
            "`{rollout_id}` (for example: /tmp/rollout_{rollout_id}.pt)."
        ),
    )
    parser.add_argument(
        "--num-rollout",
        type=int,
        default=DEFAULT_NUM_ROLLOUT,
        help=f"Number of rollout ids to consume. Default: {DEFAULT_NUM_ROLLOUT}.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help=f"Number of visible GPUs to use. Default: {DEFAULT_NUM_GPUS}.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to store train dump .pt files. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--save-tag",
        type=str,
        default="ptm_on",
        help="Filename prefix for saved debug train data. Default: ptm_on.",
    )
    parser.add_argument(
        "--load-debug-rollout-data-subsample",
        type=float,
        default=None,
        help="Optional subsample ratio for loaded rollout debug data.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip model/dataset prepare steps if the environment is already ready.",
    )
    parser.add_argument(
        "--ci-test",
        action="store_true",
        help="Enable CI-only sanity assertions in the training pipeline. Disabled by default.",
    )
    return parser


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


def prepare(model_cfg: dict[str, str | None]) -> None:
    model_path = str(model_cfg["model_path"])
    model_name = str(model_cfg["model_name"])
    download_model_id = model_cfg["download_model_id"]

    U.exec_command(f"mkdir -p {MODEL_ROOT} /root/datasets")
    model_path_obj = Path(model_path)
    if model_path_obj.exists() and any(model_path_obj.iterdir()):
        print(f"Skip model download since model path already exists: {model_path}")
    elif download_model_id is not None:
        U.exec_command(f"hf download {download_model_id} --local-dir {model_path}")
    else:
        raise RuntimeError(
            f"Model path does not exist and preset {model_name} has no default download source. "
            "Please provide --model-path pointing to an existing local checkpoint or use --skip-prepare."
        )
    U.hf_download_dataset("zhuzilin/gsm8k")


def _validate_runtime_gpus(num_gpus: int, model_path: str) -> None:
    if _DETECTED_CUDA_GPUS <= 0:
        raise RuntimeError(
            "No CUDA GPU detected in current process. "
            "Please run in a GPU environment or set --num-gpus explicitly after checking device visibility."
        )
    if num_gpus > _DETECTED_CUDA_GPUS:
        raise RuntimeError(
            f"Configured num_gpus={num_gpus} is larger than detected CUDA GPUs={_DETECTED_CUDA_GPUS}. "
            "Set --num-gpus to a valid value."
        )
    print(
        f"PTM phase-3 runtime config: detected_cuda_gpus={_DETECTED_CUDA_GPUS}, "
        f"num_gpus={num_gpus}, model_path={model_path}"
    )


def _validate_parallelism(
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    expert_tensor_parallel_size: int,
    max_tokens_per_gpu: int,
    decoder_last_pipeline_num_layers: int | None,
) -> None:
    for name, value in [
        ("tensor_model_parallel_size", tensor_model_parallel_size),
        ("pipeline_model_parallel_size", pipeline_model_parallel_size),
        ("context_parallel_size", context_parallel_size),
        ("expert_model_parallel_size", expert_model_parallel_size),
        ("expert_tensor_parallel_size", expert_tensor_parallel_size),
        ("max_tokens_per_gpu", max_tokens_per_gpu),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")

    if context_parallel_size != 1:
        raise ValueError(
            "PTM forward-only currently supports context_parallel_size=1 only. "
            "TP is supported, but CP/PTM is not implemented yet."
        )

    if pipeline_model_parallel_size == 1 and decoder_last_pipeline_num_layers is not None:
        raise ValueError(
            "--decoder-last-pipeline-num-layers should be unset when --pipeline-model-parallel-size=1."
        )


def _validate_rollout_pt_path(rollout_pt: str, num_rollout: int) -> None:
    if "{rollout_id}" in rollout_pt:
        missing = []
        for rollout_id in range(num_rollout):
            candidate = rollout_pt.format(rollout_id=rollout_id)
            if not Path(candidate).is_file():
                missing.append(candidate)
        if missing:
            raise FileNotFoundError(
                "Missing rollout .pt files for the provided template: " + ", ".join(missing)
            )
        return

    path = Path(rollout_pt)
    if not path.is_file():
        raise FileNotFoundError(f"Rollout .pt file does not exist: {rollout_pt}")
    if num_rollout != 1:
        raise ValueError(
            "--num-rollout must be 1 when --rollout-pt is a single file path without {rollout_id}."
        )


def _runtime_env_vars() -> dict[str, str]:
    return {
        "PYTHONPATH": RUNTIME_PYTHONPATH,
        "LD_LIBRARY_PATH": _build_runtime_ld_library_path(),
        "CUDNN_LOGERR_DBG": "1",
        "CUDNN_LOGDEST_DBG": "stderr",
        "OTEL_SDK_DISABLED": "true",
        "OTEL_METRICS_EXPORTER": "none",
        "OTEL_TRACES_EXPORTER": "none",
        "OTEL_LOGS_EXPORTER": "none",
    }


def _default_ref_load(model_path: str) -> str:
    return f"{model_path.rstrip('/')}_torch_dist"


def _resolve_ref_load(megatron_to_hf_mode: str, model_path: str, ref_load: str | None) -> str:
    if ref_load is not None:
        return ref_load
    if megatron_to_hf_mode == "bridge":
        return model_path
    return os.environ.get("SLIME_PTM_E2E_REF_LOAD", _default_ref_load(model_path))


def _validate_ref_load(megatron_to_hf_mode: str, ref_load: str) -> None:
    if megatron_to_hf_mode != "raw":
        return
    tracker = Path(ref_load) / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        return
    raise FileNotFoundError(
        "Raw mode requires --ref-load to point to a Megatron torch_dist checkpoint directory. "
        f"Missing tracker file: {tracker}. "
        "Convert the HF checkpoint first with `tools/convert_hf_to_torch_dist.py`."
    )


def _common_args(
    num_rollout: int,
    num_gpus: int,
    model_path: str,
    ref_load: str,
    megatron_to_hf_mode: str,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    expert_tensor_parallel_size: int,
    max_tokens_per_gpu: int,
    decoder_last_pipeline_num_layers: int | None,
) -> str:
    ckpt_args = f"--hf-checkpoint {model_path} " f"--ref-load {ref_load} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {tensor_model_parallel_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {pipeline_model_parallel_size} "
        f"--context-parallel-size {context_parallel_size} "
        f"--expert-model-parallel-size {expert_model_parallel_size} "
        f"--expert-tensor-parallel-size {expert_tensor_parallel_size} "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
    )
    if decoder_last_pipeline_num_layers is not None:
        perf_args += f"--decoder-last-pipeline-num-layers {decoder_last_pipeline_num_layers} "

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
        f"--actor-num-gpus-per-node {num_gpus} "
        "--colocate "
        f"--megatron-to-hf-mode {megatron_to_hf_mode} "
        "--seed 42 "
    )

    return f"{ckpt_args} " f"{rollout_args} " f"{optimizer_args} " f"{grpo_args} " f"{perf_args} " f"{misc_args} "


def _ptm_args() -> str:
    args = f"--slime-prefix-tree-merging --slime-prefix-min-group-size {PTM_MIN_GROUP_SIZE} "
    if PTM_PREFIX_MAX_LEN:
        args += f"--slime-prefix-max-len {PTM_PREFIX_MAX_LEN} "
    return args


def execute_phase3_only(
    *,
    rollout_pt: str,
    num_rollout: int,
    num_gpus: int,
    model_path: str,
    ref_load: str,
    model_type: str,
    megatron_to_hf_mode: str,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    expert_tensor_parallel_size: int,
    max_tokens_per_gpu: int,
    decoder_last_pipeline_num_layers: int | None,
    save_dir: str,
    save_tag: str,
    load_debug_rollout_data_subsample: float | None,
    ci_test: bool,
) -> None:
    phase_args = (
        f"{_common_args(num_rollout, num_gpus, model_path, ref_load, megatron_to_hf_mode, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size, expert_model_parallel_size, expert_tensor_parallel_size, max_tokens_per_gpu, decoder_last_pipeline_num_layers)} "
        f"--load-debug-rollout-data {rollout_pt} "
        f"--save-debug-train-data {save_dir}/{save_tag}_train_{{rollout_id}}_{{rank}}.pt "
    )
    if ci_test:
        phase_args += "--ci-test "
    if load_debug_rollout_data_subsample is not None:
        phase_args += f"--load-debug-rollout-data-subsample {load_debug_rollout_data_subsample} "
    phase_args += _ptm_args()

    print("=" * 80)
    print("Phase 3 only: train-only (PTM ON) from pre-generated rollout .pt")
    print(f"Rollout input: {rollout_pt}")
    print(f"Train dump dir: {save_dir}")
    print(f"HF checkpoint: {model_path}")
    print(f"Ref load: {ref_load}")
    print(f"Megatron/HF mode: {megatron_to_hf_mode}")
    print(
        "Parallel config: "
        f"TP={tensor_model_parallel_size}, "
        f"PP={pipeline_model_parallel_size}, "
        f"CP={context_parallel_size}, "
        f"EP={expert_model_parallel_size}, "
        f"ETP={expert_tensor_parallel_size}, "
        f"max_tokens_per_gpu={max_tokens_per_gpu}, "
        f"decoder_last_pipeline_num_layers={decoder_last_pipeline_num_layers}"
    )
    print("=" * 80)
    U.execute_train(
        train_args=phase_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=model_type,
        extra_env_vars=_runtime_env_vars(),
    )


def main() -> None:
    args = build_parser().parse_args()
    model_cfg = _resolve_model_config(
        model=args.model,
        model_name=args.model_name,
        model_type=args.model_type,
        model_path=args.model_path,
    )

    _validate_runtime_gpus(args.num_gpus, model_cfg["model_path"])
    _validate_rollout_pt_path(args.rollout_pt, args.num_rollout)
    _validate_parallelism(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        decoder_last_pipeline_num_layers=args.decoder_last_pipeline_num_layers,
    )
    ref_load = _resolve_ref_load(args.megatron_to_hf_mode, model_cfg["model_path"], args.ref_load)
    _validate_ref_load(args.megatron_to_hf_mode, ref_load)

    if not args.skip_prepare:
        prepare(model_cfg)

    save_dir = args.save_dir or tempfile.mkdtemp(prefix="slime_ptm_phase3_from_pt_")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Using output dir: {save_dir}")

    execute_phase3_only(
        rollout_pt=args.rollout_pt,
        num_rollout=args.num_rollout,
        num_gpus=args.num_gpus,
        model_path=model_cfg["model_path"],
        ref_load=ref_load,
        model_type=model_cfg["model_type"],
        megatron_to_hf_mode=args.megatron_to_hf_mode,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        decoder_last_pipeline_num_layers=args.decoder_last_pipeline_num_layers,
        save_dir=save_dir,
        save_tag=args.save_tag,
        load_debug_rollout_data_subsample=args.load_debug_rollout_data_subsample,
        ci_test=args.ci_test,
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    main()
