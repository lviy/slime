#!/usr/bin/env python3
"""Print prompt/response pairs from a Slime rollout debug dump `.pt` file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print prompt/response pairs from a Slime rollout debug dump `.pt` file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python slime/tools/print_rollout_prompt_response.py /path/to/rollout_0.pt\n"
            "  python slime/tools/print_rollout_prompt_response.py /path/to/rollout_0.pt --limit 5\n"
            "  python slime/tools/print_rollout_prompt_response.py /path/to/rollout_0.pt --show-metadata"
        ),
    )
    parser.add_argument("pt_path", type=Path, help="Path to the rollout debug `.pt` file")
    parser.add_argument("--limit", type=int, help="Print at most this many samples")
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Also print `metadata` for each sample",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        help="Truncate each rendered prompt/response/metadata field to this many characters",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if args.max_chars is not None and args.max_chars <= 0:
        parser.error("--max-chars must be positive.")

    return args


def sample_to_dict(sample: Any) -> dict[str, Any]:
    if hasattr(sample, "to_dict"):
        sample = sample.to_dict()
    if isinstance(sample, dict):
        return sample

    result = {}
    for key in ("prompt", "response", "metadata", "index", "group_index", "tokens", "response_length"):
        if hasattr(sample, key):
            result[key] = getattr(sample, key)
    return result


def format_value(value: Any, max_chars: int | None) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        rendered = json.dumps(value, ensure_ascii=False, indent=2)

    if max_chars is not None and len(rendered) > max_chars:
        return f"{rendered[:max_chars]}...<truncated:{len(rendered)}>"
    return rendered


def load_samples(pt_path: Path) -> tuple[Any, list[Any]]:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected top-level payload to be dict, got {type(payload)!r}")

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise KeyError("Expected top-level payload to contain a `samples` list.")

    return payload.get("rollout_id"), samples


def main() -> int:
    args = parse_args()
    rollout_id, raw_samples = load_samples(args.pt_path)

    samples = [sample_to_dict(sample) for sample in raw_samples]
    if args.limit is not None:
        samples = samples[: args.limit]

    print(f"file: {args.pt_path}")
    print(f"rollout_id: {rollout_id}")
    print(f"num_samples: {len(raw_samples)}")
    print(f"printed_samples: {len(samples)}")

    for idx, sample in enumerate(samples):
        prompt = format_value(sample.get("prompt", ""), args.max_chars)
        response = format_value(sample.get("response", ""), args.max_chars)

        print()
        print("=" * 80)
        print(f"sample[{idx}]")
        if sample.get("index") is not None:
            print(f"sample_index: {sample['index']}")
        if sample.get("group_index") is not None:
            print(f"group_index: {sample['group_index']}")
        print("-" * 80)
        print("PROMPT:")
        print(prompt)
        print("-" * 80)
        print("RESPONSE:")
        print(response)

        if args.show_metadata:
            print("-" * 80)
            print("METADATA:")
            print(format_value(sample.get("metadata", {}), args.max_chars))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.stderr.close()
        raise SystemExit(1)
