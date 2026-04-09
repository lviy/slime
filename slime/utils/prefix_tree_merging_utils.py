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


@dataclass
class _TrieNode:
    count: int
    children: dict[int, "_TrieNode"]

    @staticmethod
    def new() -> "_TrieNode":
        return _TrieNode(count=0, children={})


def build_prefix_group_metadata(
    tokens: Sequence[Sequence[int] | Any],
    effective_lengths: Sequence[int] | None = None,
    response_lengths: Sequence[int] | None = None,
    prefix_max_len: int | None = None,
    min_group_size: int = 2,
) -> dict[str, Any]:
    """Build per-sample PTM metadata from full-sequence prefix-tree matching.

    Each sample traverses a trie built from effective token sequences (without
    right-padding). Its reusable prefix length is the deepest depth whose trie
    node support count is at least ``min_group_size``.
    """

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

    if effective_lengths is None and response_lengths is not None:
        # Backward-compatible fallback.
        effective_lengths = []
        for tok, response_len in zip(tokens, response_lengths, strict=True):
            tok_list = _as_int_list(tok)
            effective_lengths.append(max(len(tok_list) - int(response_len), 0))

    if effective_lengths is None:
        effective_lengths = [len(_as_int_list(tok)) for tok in tokens]

    if len(tokens) != len(effective_lengths):
        raise ValueError(
            f"tokens and effective_lengths length mismatch: {len(tokens)} vs {len(effective_lengths)}"
        )

    clipped_sequences: list[list[int]] = []
    for tok, effective_len in zip(tokens, effective_lengths, strict=True):
        tok_list = _as_int_list(tok)
        capped_effective_len = max(min(int(effective_len), len(tok_list)), 0)
        cap_len = capped_effective_len if prefix_max_len is None else min(capped_effective_len, int(prefix_max_len))
        clipped_sequences.append(tok_list[:cap_len])

    root = _TrieNode.new()
    for seq in clipped_sequences:
        node = root
        for token in seq:
            child = node.children.get(token)
            if child is None:
                child = _TrieNode.new()
                node.children[token] = child
            child.count += 1
            node = child

    prefix_lens: list[int] = [0] * n
    groups_by_key: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, seq in enumerate(clipped_sequences):
        node = root
        best_depth = 0
        for depth, token in enumerate(seq, start=1):
            child = node.children.get(token)
            if child is None:
                break
            if child.count >= min_group_size:
                best_depth = depth
            node = child
        prefix_lens[i] = best_depth
        if best_depth > 0:
            groups_by_key[tuple(seq[:best_depth])].append(i)

    group_ids: list[int] = [-1] * n
    group_sizes: list[int] = [1] * n
    mergeable_group_id_by_key: dict[tuple[int, ...], int] = {}
    max_group_size = 0
    mergeable_samples = 0

    for key, indices in groups_by_key.items():
        group_size = len(indices)
        max_group_size = max(max_group_size, group_size)
        if group_size < min_group_size:
            continue

        gid = len(mergeable_group_id_by_key)
        mergeable_group_id_by_key[key] = gid
        for idx in indices:
            group_ids[idx] = gid
            group_sizes[idx] = group_size
        mergeable_samples += group_size

    max_group_size = max(max_group_size, 1 if n > 0 else 0)
    num_mergeable_groups = len(mergeable_group_id_by_key)
    num_groups = num_mergeable_groups + (n - mergeable_samples)
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
    local tokens and effective lengths as fallback.
    """

    group_ids = rollout_data.get("ptm_group_ids")
    prefix_lens = rollout_data.get("ptm_prefix_lens")

    if group_ids is None or prefix_lens is None:
        tokens = rollout_data.get("tokens")
        effective_lengths = rollout_data.get("total_lengths")
        if tokens is None:
            return None
        fallback = build_prefix_group_metadata(tokens, effective_lengths=effective_lengths, min_group_size=min_group_size)
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
