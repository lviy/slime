#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ACTOR_PERF_RE = re.compile(r"train_metric_utils\.py:\d+\s+-\s+perf\s+\d+:\s+(\{.*\})")
ROLLOUT_PERF_RE = re.compile(r"\[PTMDebug\]\s+rollout perf\s+\d+:\s+(\{.*\})")
DRIVER_PERF_RE = re.compile(r"\[PTMDebug\]\s+driver perf\s+\d+:\s+(\{.*\})")
SUMMARY_RE = re.compile(r"^(ptm_off|ptm_on|comparison):\s+(\{.*\})\s*$", re.MULTILINE)
BLOCKSIZE_IN_LOG_RE = re.compile(r"slime-prefix-runtime-block-size\s+(\d+)")
RUNTIME_BLOCKSIZE_RE = re.compile(r"runtime_block_size=(\d+)")
BLOCKSIZE_IN_PATH_RE = re.compile(r"(?:block(?:size)?|bs)[=_-]?(\d+)", re.IGNORECASE)


DEFAULT_STAGE_METRICS = [
    "perf/rollout_ptm_metadata_build_time",
    "perf/rollout_ptm_schedule_context_build_time",
    "perf/driver_generate_time",
    "perf/train_wait_time",
    "perf/log_probs_ptm_time",
    "perf/log_probs_kernel_time",
    "perf/log_probs_time",
    "perf/actor_train_time",
    "perf/train_time",
]


@dataclass
class ParsedLog:
    block_size: int
    path: Path
    actor_perf: dict[str, float]
    rollout_perf: dict[str, float]
    driver_perf: dict[str, float]
    summary: dict[str, dict[str, Any]]


def _safe_literal_eval_dict(raw: str) -> dict[str, Any]:
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dict literal, got: {type(parsed).__name__}")
    return parsed


def _coerce_float_dict(data: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[str(key)] = float(value)
    return out


def _extract_last_dicts(
    text: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, dict[str, Any]]]:
    actor_perf_matches = ACTOR_PERF_RE.findall(text)
    rollout_perf_matches = ROLLOUT_PERF_RE.findall(text)
    driver_matches = DRIVER_PERF_RE.findall(text)
    summary_matches = SUMMARY_RE.findall(text)

    actor_perf = _coerce_float_dict(_safe_literal_eval_dict(actor_perf_matches[-1])) if actor_perf_matches else {}
    rollout_perf = _coerce_float_dict(_safe_literal_eval_dict(rollout_perf_matches[-1])) if rollout_perf_matches else {}
    driver_perf = _coerce_float_dict(_safe_literal_eval_dict(driver_matches[-1])) if driver_matches else {}

    summary: dict[str, dict[str, Any]] = {}
    for label, raw_dict in summary_matches:
        summary[label] = _safe_literal_eval_dict(raw_dict)

    return actor_perf, rollout_perf, driver_perf, summary


def _infer_block_size_from_text(text: str, path: Path) -> int:
    path_match = BLOCKSIZE_IN_PATH_RE.search(str(path))
    if path_match:
        return int(path_match.group(1))

    cli_match = BLOCKSIZE_IN_LOG_RE.search(text)
    if cli_match:
        return int(cli_match.group(1))

    runtime_matches = RUNTIME_BLOCKSIZE_RE.findall(text)
    if runtime_matches:
        # Prefer the largest seen runtime block size to avoid accidental "1" from unrelated logs.
        return max(int(v) for v in runtime_matches)

    # Backward-compatible default: token-level exact PTM.
    return 1


def parse_one_log(path: Path, explicit_block_size: int | None = None) -> ParsedLog:
    text = path.read_text(encoding="utf-8", errors="replace")
    actor_perf, rollout_perf, driver_perf, summary = _extract_last_dicts(text)
    block_size = explicit_block_size if explicit_block_size is not None else _infer_block_size_from_text(text, path)
    return ParsedLog(
        block_size=block_size,
        path=path,
        actor_perf=actor_perf,
        rollout_perf=rollout_perf,
        driver_perf=driver_perf,
        summary=summary,
    )


def _build_combined_metric_row(parsed: ParsedLog, metrics: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "block_size": parsed.block_size,
        "log_path": str(parsed.path),
    }

    for metric in metrics:
        value = parsed.actor_perf.get(metric)
        if value is None:
            value = parsed.rollout_perf.get(metric)
        if value is None:
            value = parsed.driver_perf.get(metric)
        row[metric] = value

    ptm_on_summary = parsed.summary.get("ptm_on", {})
    row["ptm_on_wall_time_s_mean"] = _maybe_float(ptm_on_summary.get("wall_time_s_mean"))
    row["ptm_on_samples_per_s_mean"] = _maybe_float(ptm_on_summary.get("samples_per_s_mean"))
    row["ptm_on_total_tokens_per_s_mean"] = _maybe_float(ptm_on_summary.get("total_tokens_per_s_mean"))
    row["ptm_on_response_tokens_per_s_mean"] = _maybe_float(ptm_on_summary.get("response_tokens_per_s_mean"))
    return row


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _save_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_json(rows: list[dict[str, Any]], json_path: Path) -> None:
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _label_for_metric(metric: str) -> str:
    if metric.startswith("perf/") and metric.endswith("_time"):
        metric = metric[len("perf/") : -len("_time")]
    return metric.replace("_", "\n")


def plot_stage_bars(rows: list[dict[str, Any]], metrics: list[str], output_path: Path) -> None:
    block_sizes = [row["block_size"] for row in rows]
    x = np.arange(len(block_sizes))
    width = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots(figsize=(max(12, len(block_sizes) * 1.7), 7))
    cmap = plt.get_cmap("tab20")

    for idx, metric in enumerate(metrics):
        values = [row.get(metric) if row.get(metric) is not None else 0.0 for row in rows]
        offsets = x + (idx - (len(metrics) - 1) / 2) * width
        ax.bar(offsets, values, width=width, label=_label_for_metric(metric), color=cmap(idx % 20))

    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in block_sizes])
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Time (s)")
    ax.set_title("PTM Stage Time vs Block Size")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_kernel_and_total(rows: list[dict[str, Any]], output_path: Path) -> None:
    block_sizes = [row["block_size"] for row in rows]
    x = np.arange(len(block_sizes))

    kernel = [row.get("perf/log_probs_kernel_time") if row.get("perf/log_probs_kernel_time") is not None else np.nan for row in rows]
    total = [
        row.get("ptm_on_wall_time_s_mean")
        if row.get("ptm_on_wall_time_s_mean") is not None
        else row.get("perf/step_time")
        if row.get("perf/step_time") is not None
        else np.nan
        for row in rows
    ]

    fig, ax1 = plt.subplots(figsize=(max(10, len(block_sizes) * 1.5), 6))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, kernel, width=0.55, color="#2a6f97", label="log_probs_kernel_time")
    line = ax2.plot(x, total, color="#c1121f", marker="o", linewidth=2.0, label="ptm_on_wall_time_s_mean / step_time")

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(bs) for bs in block_sizes])
    ax1.set_xlabel("Block Size")
    ax1.set_ylabel("Kernel Time (s)", color="#2a6f97")
    ax2.set_ylabel("Total Time (s)", color="#c1121f")
    ax1.set_title("Kernel Time and Total Time vs Block Size")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    handles = [bars, line[0]]
    labels = ["log_probs_kernel_time", "ptm_on_wall_time_s_mean / step_time"]
    ax1.legend(handles, labels, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_total_breakdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    block_sizes = [row["block_size"] for row in rows]
    x = np.arange(len(block_sizes))

    stacked_metrics = [
        "perf/rollout_ptm_metadata_build_time",
        "perf/rollout_ptm_schedule_context_build_time",
        "perf/driver_generate_time",
        "perf/train_wait_time",
        "perf/log_probs_ptm_time",
        "perf/log_probs_kernel_time",
        "perf/actor_train_time",
    ]

    fig, ax = plt.subplots(figsize=(max(12, len(block_sizes) * 1.7), 7))
    bottom = np.zeros(len(block_sizes), dtype=float)
    cmap = plt.get_cmap("Set2")

    for idx, metric in enumerate(stacked_metrics):
        values = np.array([row.get(metric) or 0.0 for row in rows], dtype=float)
        ax.bar(x, values, bottom=bottom, label=_label_for_metric(metric), color=cmap(idx % 8))
        bottom += values

    total_line = np.array(
        [
            row.get("ptm_on_wall_time_s_mean")
            if row.get("ptm_on_wall_time_s_mean") is not None
            else row.get("perf/step_time") or np.nan
            for row in rows
        ],
        dtype=float,
    )
    ax.plot(x, total_line, color="black", marker="o", linewidth=2.0, label="total")

    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in block_sizes])
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Time (s)")
    ax.set_title("PTM Time Breakdown vs Block Size")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _parse_log_arg(raw: str) -> tuple[int | None, Path]:
    if "=" in raw:
        left, right = raw.split("=", 1)
        left = left.strip()
        right = right.strip()
        if left.isdigit():
            return int(left), Path(right)
    return None, Path(raw)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse multiple PTM speed-test logs for different block sizes and plot PTM-related stage times, "
            "kernel time, and total time differences."
        )
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help=(
            "Log paths, optionally with explicit block size as '256=/path/to/log.txt'. "
            "If omitted, block size is inferred from filename or log contents."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="slime/ptm_blocksize_plots",
        help="Directory to write plots and summary tables.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_STAGE_METRICS,
        help="PTM-related metrics to show in the grouped stage-time plot.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_logs: list[ParsedLog] = []
    for raw in args.logs:
        explicit_block_size, path = _parse_log_arg(raw)
        parsed_logs.append(parse_one_log(path, explicit_block_size=explicit_block_size))

    parsed_logs.sort(key=lambda item: item.block_size)
    rows = [_build_combined_metric_row(item, args.metrics) for item in parsed_logs]

    _save_csv(rows, output_dir / "ptm_blocksize_metrics.csv")
    _save_json(rows, output_dir / "ptm_blocksize_metrics.json")
    plot_stage_bars(rows, args.metrics, output_dir / "ptm_stage_times_by_blocksize.png")
    plot_kernel_and_total(rows, output_dir / "ptm_kernel_and_total_by_blocksize.png")
    plot_total_breakdown(rows, output_dir / "ptm_total_breakdown_by_blocksize.png")

    print(f"Wrote plots and tables to: {output_dir}")
    print(f"Parsed block sizes: {[item.block_size for item in parsed_logs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
