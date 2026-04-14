"""
Run a phase-3-only PTM forward speed benchmark using pre-generated rollout `.pt` data.

Compared with `test_ptm_forward_only.py`, this script focuses on throughput rather than
numerical comparison:
1) Skips rollout generation entirely.
2) Loads rollout data from a user-provided `.pt` path or template.
3) Runs train-only / forward-only on the same `.pt` with PTM OFF, ON, or BOTH.
4) Records wall time and simple throughput metrics into a JSON summary.

Examples:
  python3 tests/test_ptm_forward_speed.py \
      --rollout-pt /tmp/rollout_0.pt

  python3 tests/test_ptm_forward_speed.py \
      --rollout-pt /tmp/rollout_0.pt \
      --ptm-mode both \
      --measure-runs 3 \
      --save-dir /tmp/ptm_speed

  python3 tests/test_ptm_forward_speed.py \
      --model glm4.7-flash \
      --rollout-pt /tmp/rollout_0.pt \
      --ptm-mode on \
      --skip-prepare
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

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
}


def _resolve_model_config(model: str, model_name: str | None, model_type: str | None, model_path: str | None) -> dict[str, str | None]:
    preset = MODEL_PRESETS.get(model)
    if preset is None:
        raise ValueError(f"Unsupported --model {model}. Available presets: {sorted(MODEL_PRESETS.keys())}")

    resolved_model_name = model_name or preset["model_name"]
    resolved_model_type = model_type or preset["model_type"]
    resolved_model_path = model_path or os.environ.get("SLIME_PTM_E2E_MODEL_PATH", f"{MODEL_ROOT}/{resolved_model_name}")
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
    parser.add_argument("--model-name", type=str, default=None, help="Optional override for HuggingFace checkpoint directory/model name.")
    parser.add_argument("--model-type", type=str, default=None, help="Optional override for Slime/Megatron model type.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional override for the full local model checkpoint path.")
    parser.add_argument(
        "--rollout-pt",
        required=True,
        help=(
            "Path to a saved rollout .pt file, or a template containing "
            "`{rollout_id}` (for example: /tmp/rollout_{rollout_id}.pt)."
        ),
    )
    parser.add_argument("--num-rollout", type=int, default=DEFAULT_NUM_ROLLOUT, help=f"Number of rollout ids to consume. Default: {DEFAULT_NUM_ROLLOUT}.")
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS, help=f"Number of visible GPUs to use. Default: {DEFAULT_NUM_GPUS}.")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to store benchmark outputs. Defaults to a temporary directory.")
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional full path of the JSON summary. Defaults to <save-dir>/ptm_forward_speed_summary.json.",
    )
    parser.add_argument(
        "--ptm-mode",
        type=str,
        default="both",
        choices=("on", "off", "both"),
        help="Benchmark PTM ON, OFF, or BOTH. Default: both.",
    )
    parser.add_argument("--measure-runs", type=int, default=1, help="Number of measured runs per PTM mode. Default: 1.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Number of warmup runs per PTM mode before measurement. Default: 0.")
    parser.add_argument(
        "--load-debug-rollout-data-subsample",
        type=float,
        default=None,
        help="Optional subsample ratio for loaded rollout debug data.",
    )
    parser.add_argument(
        "--save-debug-train-data",
        action="store_true",
        help="If set, also dump debug train `.pt` files for each measured run. Disabled by default to avoid I/O noise.",
    )
    parser.add_argument(
        "--save-tag-prefix",
        type=str,
        default="ptm_speed",
        help="Filename prefix used when --save-debug-train-data is enabled. Default: ptm_speed.",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Skip model/dataset prepare steps if the environment is already ready.")
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
        f"PTM phase-3 speed runtime config: detected_cuda_gpus={_DETECTED_CUDA_GPUS}, "
        f"num_gpus={num_gpus}, model_path={model_path}"
    )


def _validate_rollout_pt_path(rollout_pt: str, num_rollout: int) -> None:
    if "{rollout_id}" in rollout_pt:
        missing = []
        for rollout_id in range(num_rollout):
            candidate = rollout_pt.format(rollout_id=rollout_id)
            if not Path(candidate).is_file():
                missing.append(candidate)
        if missing:
            raise FileNotFoundError("Missing rollout .pt files for the provided template: " + ", ".join(missing))
        return

    path = Path(rollout_pt)
    if not path.is_file():
        raise FileNotFoundError(f"Rollout .pt file does not exist: {rollout_pt}")
    if num_rollout != 1:
        raise ValueError("--num-rollout must be 1 when --rollout-pt is a single file path without {rollout_id}.")


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


def _common_args(num_rollout: int, num_gpus: int, model_path: str) -> str:
    ckpt_args = f"--hf-checkpoint {model_path}/ " f"--ref-load {model_path}/ "

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
        f"--actor-num-gpus-per-node {num_gpus} "
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


def _apply_subsample(samples: list[dict], ratio: float | None) -> list[dict]:
    if ratio is None:
        return samples
    original_num_rows = len(samples)
    rough_subsample_num_rows = int(original_num_rows * ratio)
    if rough_subsample_num_rows <= 0:
        return []
    half = rough_subsample_num_rows // 2
    return samples[:half] + samples[-half:]


def _iter_rollout_paths(rollout_pt: str, num_rollout: int) -> list[str]:
    if "{rollout_id}" in rollout_pt:
        return [rollout_pt.format(rollout_id=rollout_id) for rollout_id in range(num_rollout)]
    return [rollout_pt]


def _summarize_rollout_pt(rollout_pt: str, num_rollout: int, subsample_ratio: float | None) -> dict[str, float | int | list[str]]:
    paths = _iter_rollout_paths(rollout_pt, num_rollout)
    total_tokens = 0
    response_tokens = 0
    total_lengths: list[int] = []
    response_lengths: list[int] = []
    num_samples = 0

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw_samples = payload["samples"]
        raw_samples = _apply_subsample(raw_samples, subsample_ratio)
        num_samples += len(raw_samples)

        for sample in raw_samples:
            tokens = sample.get("tokens", []) or []
            response_length = int(sample.get("response_length", 0) or 0)
            total_length = len(tokens)
            total_tokens += total_length
            response_tokens += response_length
            total_lengths.append(total_length)
            response_lengths.append(response_length)

    prompt_tokens = total_tokens - response_tokens
    return {
        "rollout_files": paths,
        "num_rollout_files": len(paths),
        "num_samples": num_samples,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "max_total_length": max(total_lengths) if total_lengths else 0,
        "avg_total_length": (sum(total_lengths) / len(total_lengths)) if total_lengths else 0.0,
        "max_response_length": max(response_lengths) if response_lengths else 0,
        "avg_response_length": (sum(response_lengths) / len(response_lengths)) if response_lengths else 0.0,
    }


def _build_run_spec(ptm_mode: str) -> list[tuple[str, bool]]:
    if ptm_mode == "off":
        return [("ptm_off", False)]
    if ptm_mode == "on":
        return [("ptm_on", True)]
    return [("ptm_off", False), ("ptm_on", True)]


def _safe_div(numerator: float | int, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / denominator


def execute_phase3_only(
    *,
    rollout_pt: str,
    num_rollout: int,
    num_gpus: int,
    model_path: str,
    model_type: str,
    ptm_enabled: bool,
    save_dir: str,
    save_tag: str,
    load_debug_rollout_data_subsample: float | None,
    save_debug_train_data: bool,
) -> float:
    phase_args = f"{_common_args(num_rollout, num_gpus, model_path)} " f"--load-debug-rollout-data {rollout_pt} " "--ci-test "
    if save_debug_train_data:
        phase_args += f"--save-debug-train-data {save_dir}/{save_tag}_train_{{rollout_id}}_{{rank}}.pt "
    if load_debug_rollout_data_subsample is not None:
        phase_args += f"--load-debug-rollout-data-subsample {load_debug_rollout_data_subsample} "
    if ptm_enabled:
        phase_args += _ptm_args()

    print("=" * 80)
    print(f"Phase 3 speed run: {'PTM ON' if ptm_enabled else 'PTM OFF'}")
    print(f"Rollout input: {rollout_pt}")
    print(f"Save dir: {save_dir}")
    print("=" * 80)

    start_time = perf_counter()
    U.execute_train(
        train_args=phase_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=model_type,
        extra_env_vars=_runtime_env_vars(),
    )
    return perf_counter() - start_time


def _aggregate_runs(runs: list[dict]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        if run["phase"] != "measure":
            continue
        grouped.setdefault(run["label"], []).append(run)

    aggregates: dict[str, dict[str, float | int | None]] = {}
    for label, label_runs in grouped.items():
        wall_times = [float(run["wall_time_s"]) for run in label_runs]
        sample_tps = [float(run["samples_per_s"]) for run in label_runs if run["samples_per_s"] is not None]
        total_token_tps = [
            float(run["total_tokens_per_s"]) for run in label_runs if run["total_tokens_per_s"] is not None
        ]
        response_token_tps = [
            float(run["response_tokens_per_s"]) for run in label_runs if run["response_tokens_per_s"] is not None
        ]

        aggregates[label] = {
            "count": len(label_runs),
            "wall_time_s_mean": statistics.mean(wall_times),
            "wall_time_s_min": min(wall_times),
            "wall_time_s_max": max(wall_times),
            "samples_per_s_mean": statistics.mean(sample_tps) if sample_tps else None,
            "total_tokens_per_s_mean": statistics.mean(total_token_tps) if total_token_tps else None,
            "response_tokens_per_s_mean": statistics.mean(response_token_tps) if response_token_tps else None,
        }

    if "ptm_off" in aggregates and "ptm_on" in aggregates:
        off_mean = float(aggregates["ptm_off"]["wall_time_s_mean"])
        on_mean = float(aggregates["ptm_on"]["wall_time_s_mean"])
        off_total_tps = aggregates["ptm_off"]["total_tokens_per_s_mean"]
        on_total_tps = aggregates["ptm_on"]["total_tokens_per_s_mean"]
        off_resp_tps = aggregates["ptm_off"]["response_tokens_per_s_mean"]
        on_resp_tps = aggregates["ptm_on"]["response_tokens_per_s_mean"]
        aggregates["comparison"] = {
            "ptm_on_speedup_vs_off_by_wall_time": _safe_div(off_mean, on_mean),
            "ptm_on_total_tokens_per_s_ratio_vs_off": (
                _safe_div(float(on_total_tps), float(off_total_tps))
                if off_total_tps is not None and on_total_tps is not None
                else None
            ),
            "ptm_on_response_tokens_per_s_ratio_vs_off": (
                _safe_div(float(on_resp_tps), float(off_resp_tps))
                if off_resp_tps is not None and on_resp_tps is not None
                else None
            ),
        }

    return aggregates


def run_benchmark(args, model_cfg: dict[str, str | None], save_dir: str) -> dict:
    rollout_stats = _summarize_rollout_pt(
        args.rollout_pt,
        args.num_rollout,
        args.load_debug_rollout_data_subsample,
    )

    runs: list[dict] = []
    for label, ptm_enabled in _build_run_spec(args.ptm_mode):
        for warmup_idx in range(args.warmup_runs):
            run_tag = f"{args.save_tag_prefix}_{label}_warmup{warmup_idx}"
            elapsed = execute_phase3_only(
                rollout_pt=args.rollout_pt,
                num_rollout=args.num_rollout,
                num_gpus=args.num_gpus,
                model_path=str(model_cfg["model_path"]),
                model_type=str(model_cfg["model_type"]),
                ptm_enabled=ptm_enabled,
                save_dir=save_dir,
                save_tag=run_tag,
                load_debug_rollout_data_subsample=args.load_debug_rollout_data_subsample,
                save_debug_train_data=False,
            )
            runs.append(
                {
                    "label": label,
                    "ptm_enabled": ptm_enabled,
                    "phase": "warmup",
                    "run_index": warmup_idx,
                    "wall_time_s": elapsed,
                    "samples_per_s": _safe_div(int(rollout_stats["num_samples"]), elapsed),
                    "total_tokens_per_s": _safe_div(int(rollout_stats["total_tokens"]), elapsed),
                    "response_tokens_per_s": _safe_div(int(rollout_stats["response_tokens"]), elapsed),
                    "save_tag": run_tag,
                }
            )

        for run_idx in range(args.measure_runs):
            run_tag = f"{args.save_tag_prefix}_{label}_run{run_idx}"
            elapsed = execute_phase3_only(
                rollout_pt=args.rollout_pt,
                num_rollout=args.num_rollout,
                num_gpus=args.num_gpus,
                model_path=str(model_cfg["model_path"]),
                model_type=str(model_cfg["model_type"]),
                ptm_enabled=ptm_enabled,
                save_dir=save_dir,
                save_tag=run_tag,
                load_debug_rollout_data_subsample=args.load_debug_rollout_data_subsample,
                save_debug_train_data=args.save_debug_train_data,
            )
            runs.append(
                {
                    "label": label,
                    "ptm_enabled": ptm_enabled,
                    "phase": "measure",
                    "run_index": run_idx,
                    "wall_time_s": elapsed,
                    "samples_per_s": _safe_div(int(rollout_stats["num_samples"]), elapsed),
                    "total_tokens_per_s": _safe_div(int(rollout_stats["total_tokens"]), elapsed),
                    "response_tokens_per_s": _safe_div(int(rollout_stats["response_tokens"]), elapsed),
                    "save_tag": run_tag,
                }
            )

    summary = {
        "benchmark": "ptm_forward_speed",
        "rollout_pt": args.rollout_pt,
        "num_rollout": args.num_rollout,
        "ptm_mode": args.ptm_mode,
        "measure_runs": args.measure_runs,
        "warmup_runs": args.warmup_runs,
        "num_gpus": args.num_gpus,
        "model": {
            "preset": args.model,
            "model_name": model_cfg["model_name"],
            "model_type": model_cfg["model_type"],
            "model_path": model_cfg["model_path"],
        },
        "rollout_stats": rollout_stats,
        "runs": runs,
        "aggregates": _aggregate_runs(runs),
    }
    return summary


def _summary_path(args, save_dir: str) -> Path:
    if args.summary_path is not None:
        return Path(args.summary_path)
    return Path(save_dir) / "ptm_forward_speed_summary.json"


def _print_summary(summary: dict, summary_path: Path) -> None:
    print("=" * 80)
    print("PTM forward speed benchmark finished")
    print(f"Summary path: {summary_path}")
    print(f"Rollout stats: {summary['rollout_stats']}")
    for label, stats in summary["aggregates"].items():
        print(f"{label}: {stats}")
    print("=" * 80)


def main() -> None:
    args = build_parser().parse_args()
    if args.measure_runs <= 0:
        raise ValueError("--measure-runs must be >= 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")

    model_cfg = _resolve_model_config(
        model=args.model,
        model_name=args.model_name,
        model_type=args.model_type,
        model_path=args.model_path,
    )

    _validate_runtime_gpus(args.num_gpus, str(model_cfg["model_path"]))
    _validate_rollout_pt_path(args.rollout_pt, args.num_rollout)

    if not args.skip_prepare:
        prepare(model_cfg)

    save_dir = args.save_dir or tempfile.mkdtemp(prefix="slime_ptm_phase3_speed_")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Using output dir: {save_dir}")

    summary = run_benchmark(args, model_cfg, save_dir)
    summary_path = _summary_path(args, save_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _print_summary(summary, summary_path)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    main()
