import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PrefixTreeMergingContext:
    """Summary context for Slime-side Prefix Tree Merging wiring."""

    group_ids: list[int]
    prefix_lens: list[int]
    mergeable_groups: dict[int, list[int]]
    num_samples: int
    num_groups: int
    num_mergeable_groups: int
    num_mergeable_samples: int
    max_group_size: int
    avg_group_size: float

    @property
    def enabled(self) -> bool:
        return self.num_mergeable_groups > 0


def _as_int_list(tokens: Any) -> list[int]:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return [int(x) for x in tokens]


def build_prefix_group_metadata(
    tokens: Sequence[Sequence[int] | Any],
    response_lengths: Sequence[int],
    prefix_max_len: int | None = None,
    min_group_size: int = 2,
) -> dict[str, Any]:
    """Build per-sample prefix-group metadata from tokenized rollout samples.

    The prefix is defined as the prompt segment (total_len - response_len),
    optionally clipped by ``prefix_max_len``.
    """

    if len(tokens) != len(response_lengths):
        raise ValueError(
            f"tokens and response_lengths length mismatch: {len(tokens)} vs {len(response_lengths)}"
        )

    n = len(tokens)
    if n == 0:
        return {
            "ptm_group_ids": [],
            "ptm_prefix_lens": [],
            "ptm_group_sizes": [],
            "ptm_num_groups": 0,
            "ptm_num_mergeable_groups": 0,
            "ptm_num_mergeable_samples": 0,
            "ptm_max_group_size": 0,
            "ptm_avg_group_size": 0.0,
        }

    groups_by_key: dict[tuple[int, ...], list[int]] = defaultdict(list)
    prefix_lens: list[int] = [0] * n

    for i, (tok, response_len) in enumerate(zip(tokens, response_lengths, strict=True)):
        tok_list = _as_int_list(tok)
        prompt_len = max(len(tok_list) - int(response_len), 0)
        prefix_len = prompt_len if prefix_max_len is None else min(prompt_len, prefix_max_len)
        prefix_lens[i] = prefix_len
        key = tuple(tok_list[:prefix_len]) if prefix_len > 0 else tuple()
        groups_by_key[key].append(i)

    group_ids: list[int] = [-1] * n
    group_sizes: list[int] = [1] * n
    mergeable_group_id = 0
    max_group_size = 0
    mergeable_samples = 0

    for key, indices in groups_by_key.items():
        group_size = len(indices)
        for idx in indices:
            group_sizes[idx] = group_size

        max_group_size = max(max_group_size, group_size)
        if len(key) == 0 or group_size < min_group_size:
            continue

        for idx in indices:
            group_ids[idx] = mergeable_group_id
        mergeable_group_id += 1
        mergeable_samples += group_size

    num_groups = len(groups_by_key)
    num_mergeable_groups = mergeable_group_id
    avg_group_size = (mergeable_samples / num_mergeable_groups) if num_mergeable_groups > 0 else 0.0

    return {
        "ptm_group_ids": group_ids,
        "ptm_prefix_lens": prefix_lens,
        "ptm_group_sizes": group_sizes,
        "ptm_num_groups": num_groups,
        "ptm_num_mergeable_groups": num_mergeable_groups,
        "ptm_num_mergeable_samples": mergeable_samples,
        "ptm_max_group_size": max_group_size,
        "ptm_avg_group_size": avg_group_size,
    }


def build_prefix_tree_context_from_rollout_data(
    rollout_data: dict[str, Any],
    min_group_size: int = 2,
) -> PrefixTreeMergingContext | None:
    """Build local-rank PTM context from rollout_data.

    Prefers precomputed metadata in rollout_data. If absent, it computes from
    local tokens and response lengths as fallback.
    """

    group_ids = rollout_data.get("ptm_group_ids")
    prefix_lens = rollout_data.get("ptm_prefix_lens")

    if group_ids is None or prefix_lens is None:
        tokens = rollout_data.get("tokens")
        response_lengths = rollout_data.get("response_lengths")
        if tokens is None or response_lengths is None:
            return None
        fallback = build_prefix_group_metadata(tokens, response_lengths, min_group_size=min_group_size)
        group_ids = fallback["ptm_group_ids"]
        prefix_lens = fallback["ptm_prefix_lens"]

    if len(group_ids) != len(prefix_lens):
        logger.warning(
            "[PTM] inconsistent metadata lengths: group_ids=%s, prefix_lens=%s",
            len(group_ids),
            len(prefix_lens),
        )
        return None

    local_groups: dict[int, list[int]] = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        if gid is None or gid < 0:
            continue
        local_groups[int(gid)].append(idx)

    mergeable_groups = {
        gid: indices
        for gid, indices in local_groups.items()
        if len(indices) >= min_group_size and int(prefix_lens[indices[0]]) > 0
    }

    num_mergeable_groups = len(mergeable_groups)
    num_mergeable_samples = sum(len(v) for v in mergeable_groups.values())
    max_group_size = max((len(v) for v in mergeable_groups.values()), default=0)
    avg_group_size = (num_mergeable_samples / num_mergeable_groups) if num_mergeable_groups > 0 else 0.0

    return PrefixTreeMergingContext(
        group_ids=[int(x) for x in group_ids],
        prefix_lens=[int(x) for x in prefix_lens],
        mergeable_groups=mergeable_groups,
        num_samples=len(group_ids),
        num_groups=len(local_groups),
        num_mergeable_groups=num_mergeable_groups,
        num_mergeable_samples=num_mergeable_samples,
        max_group_size=max_group_size,
        avg_group_size=avg_group_size,
    )


def log_prefix_tree_context(
    stage: str,
    context: PrefixTreeMergingContext,
    extra: dict[str, Any] | None = None,
) -> None:
    extra_info = ""
    if extra:
        extra_info = ", " + ", ".join(f"{k}={v}" for k, v in sorted(extra.items()))
    logger.info(
        "[PTM] stage=%s, enabled=%s, num_samples=%d, num_groups=%d, "
        "num_mergeable_groups=%d, num_mergeable_samples=%d, max_group_size=%d, avg_group_size=%.2f%s",
        stage,
        context.enabled,
        context.num_samples,
        context.num_groups,
        context.num_mergeable_groups,
        context.num_mergeable_samples,
        context.max_group_size,
        context.avg_group_size,
        extra_info,
    )
