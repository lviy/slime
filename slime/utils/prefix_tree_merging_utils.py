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


@dataclass
class _ScheduleTrieNode:
    children: dict[int, "_ScheduleTrieNode"]
    terminal_sample_ids: list[int]

    @staticmethod
    def root() -> "_ScheduleTrieNode":
        return _ScheduleTrieNode(children={}, terminal_sample_ids=[])


@dataclass
class _RuntimeTrieNode:
    token: int | None
    parent: "_RuntimeTrieNode | None"
    depth: int
    children: dict[int, "_RuntimeTrieNode"]
    index: int = -1

    @staticmethod
    def root() -> "_RuntimeTrieNode":
        return _RuntimeTrieNode(token=None, parent=None, depth=0, children={})


@dataclass
class PrefixTreeBatchPlan:
    merged_tokens: list[int]
    merged_position_ids: list[int]
    q_ranges: list[list[int]]
    k_ranges: list[list[int]]
    attn_type_map: list[int]
    unmerge_index: list[int]
    num_input_tokens: int
    num_merged_tokens: int


@dataclass
class PrefixTreeScheduleContext:
    sample_lengths: dict[int, int]
    sample_ranks: dict[int, int]
    rank_to_sample: list[int]
    adjacent_lcps: list[int]
    lcp_sparse_table: list[list[int]]
    lcp_log2: list[int]

    @property
    def num_samples(self) -> int:
        return len(self.rank_to_sample)

    def get_length(self, sample_idx: int) -> int:
        return int(self.sample_lengths[int(sample_idx)])

    def get_rank(self, sample_idx: int) -> int:
        return int(self.sample_ranks[int(sample_idx)])

    def get_lcp(self, sample_idx_a: int, sample_idx_b: int) -> int:
        return self.get_lcp_by_rank(self.get_rank(sample_idx_a), self.get_rank(sample_idx_b))

    def get_lcp_by_rank(self, rank_a: int, rank_b: int) -> int:
        rank_a = int(rank_a)
        rank_b = int(rank_b)
        if rank_a == rank_b:
            return self.get_length(self.rank_to_sample[rank_a])

        left = min(rank_a, rank_b)
        right = max(rank_a, rank_b) - 1
        if not self.adjacent_lcps:
            return 0

        width = right - left + 1
        level = self.lcp_log2[width]
        return min(
            self.lcp_sparse_table[level][left],
            self.lcp_sparse_table[level][right - (1 << level) + 1],
        )


def _merge_contiguous_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge tangent intervals into maximal contiguous segments."""

    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        start = int(start)
        end = int(end)
        if not merged or merged[-1][1] != start:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], end)
    return merged


def _longest_common_prefix(seq_a: Sequence[int], seq_b: Sequence[int]) -> int:
    matched = 0
    for token_a, token_b in zip(seq_a, seq_b, strict=False):
        if int(token_a) != int(token_b):
            break
        matched += 1
    return matched


def build_prefix_tree_schedule_context(
    tokens: Sequence[Sequence[int] | Any],
    sample_ids: Sequence[int] | None = None,
) -> PrefixTreeScheduleContext:
    """Build an exact PTM schedule context for one local rollout step.

    The context is used by the PTM-aware dynamic batching planner. It constructs
    a single global trie for the step, emits samples in trie-derived lexicographic
    order, and precomputes RMQ over adjacent LCPs so bucket insertion costs can
    be updated exactly from predecessor/successor relationships.
    """

    if sample_ids is None:
        sample_ids = list(range(len(tokens)))
    else:
        sample_ids = [int(sample_idx) for sample_idx in sample_ids]

    if len(tokens) != len(sample_ids):
        raise ValueError(f"tokens/sample_ids length mismatch: {len(tokens)} vs {len(sample_ids)}")

    root = _ScheduleTrieNode.root()
    sequences_by_sample: dict[int, list[int]] = {}
    sample_lengths: dict[int, int] = {}

    for sample_idx, seq_like in zip(sample_ids, tokens, strict=True):
        seq = _as_int_list(seq_like)
        sequences_by_sample[sample_idx] = seq
        sample_lengths[sample_idx] = len(seq)

        node = root
        for token in seq:
            child = node.children.get(token)
            if child is None:
                child = _ScheduleTrieNode(children={}, terminal_sample_ids=[])
                node.children[int(token)] = child
            node = child
        node.terminal_sample_ids.append(sample_idx)

    rank_to_sample: list[int] = []
    stack: list[_ScheduleTrieNode] = [root]
    while stack:
        node = stack.pop()
        if node.terminal_sample_ids:
            rank_to_sample.extend(node.terminal_sample_ids)
        if node.children:
            for token in sorted(node.children.keys(), reverse=True):
                stack.append(node.children[token])

    sample_ranks = {sample_idx: rank for rank, sample_idx in enumerate(rank_to_sample)}

    adjacent_lcps: list[int] = []
    for prev_sample_idx, sample_idx in zip(rank_to_sample, rank_to_sample[1:], strict=True):
        adjacent_lcps.append(
            _longest_common_prefix(
                sequences_by_sample[prev_sample_idx],
                sequences_by_sample[sample_idx],
            )
        )

    lcp_log2 = [0] * (len(adjacent_lcps) + 1)
    for width in range(2, len(lcp_log2)):
        lcp_log2[width] = lcp_log2[width // 2] + 1

    lcp_sparse_table: list[list[int]] = [adjacent_lcps]
    level = 1
    while (1 << level) <= len(adjacent_lcps):
        prev = lcp_sparse_table[level - 1]
        half_width = 1 << (level - 1)
        row = [
            min(prev[offset], prev[offset + half_width])
            for offset in range(len(adjacent_lcps) - (1 << level) + 1)
        ]
        lcp_sparse_table.append(row)
        level += 1

    return PrefixTreeScheduleContext(
        sample_lengths=sample_lengths,
        sample_ranks=sample_ranks,
        rank_to_sample=rank_to_sample,
        adjacent_lcps=adjacent_lcps,
        lcp_sparse_table=lcp_sparse_table,
        lcp_log2=lcp_log2,
    )


def summarize_prefix_tree_batch_plan(plan: PrefixTreeBatchPlan) -> dict[str, int | float]:
    """Summarize PTM arbitrary-range metadata size for lightweight diagnostics."""

    num_queries = int(plan.num_merged_tokens)
    num_q_ranges = len(plan.q_ranges)
    num_k_ranges = len(plan.k_ranges)
    ranges_per_query = [0] * num_queries
    total_q_range_tokens = 0
    total_k_range_tokens = 0
    max_q_range_width = 0
    max_k_range_width = 0
    q_range_pairs: list[tuple[int, int]] = []

    for q_range, k_range in zip(plan.q_ranges, plan.k_ranges, strict=True):
        q_start, q_end = int(q_range[0]), int(q_range[1])
        k_start, k_end = int(k_range[0]), int(k_range[1])
        q_width = max(q_end - q_start, 0)
        k_width = max(k_end - k_start, 0)
        q_range_pairs.append((q_start, q_end))

        total_q_range_tokens += q_width
        total_k_range_tokens += k_width
        max_q_range_width = max(max_q_range_width, q_width)
        max_k_range_width = max(max_k_range_width, k_width)

        upper = min(q_end, num_queries)
        for q_idx in range(max(q_start, 0), upper):
            ranges_per_query[q_idx] += 1

    queries_with_multiple_ranges = sum(1 for count in ranges_per_query if count > 1)
    max_ranges_per_query = max(ranges_per_query, default=0)
    unique_q_ranges = len(set(q_range_pairs))
    duplicated_q_ranges = num_q_ranges - unique_q_ranges
    q_ranges_non_overlapped = 1
    sorted_q_range_pairs = sorted(q_range_pairs)
    prev_end: int | None = None
    for q_start, q_end in sorted_q_range_pairs:
        if prev_end is not None and q_start < prev_end:
            q_ranges_non_overlapped = 0
            break
        prev_end = q_end

    return {
        "num_q_ranges": num_q_ranges,
        "num_k_ranges": num_k_ranges,
        "num_unique_q_ranges": unique_q_ranges,
        "num_duplicated_q_ranges": duplicated_q_ranges,
        "q_ranges_non_overlapped": q_ranges_non_overlapped,
        "avg_ranges_per_query": (num_q_ranges / num_queries) if num_queries > 0 else 0.0,
        "max_ranges_per_query": max_ranges_per_query,
        "queries_with_multiple_ranges": queries_with_multiple_ranges,
        "total_q_range_tokens": total_q_range_tokens,
        "total_k_range_tokens": total_k_range_tokens,
        "avg_q_range_width": (total_q_range_tokens / num_q_ranges) if num_q_ranges > 0 else 0.0,
        "avg_k_range_width": (total_k_range_tokens / num_k_ranges) if num_k_ranges > 0 else 0.0,
        "avg_attended_tokens_per_query": (total_k_range_tokens / num_queries) if num_queries > 0 else 0.0,
        "max_q_range_width": max_q_range_width,
        "max_k_range_width": max_k_range_width,
    }


def get_prefix_tree_runtime_skip_reason(
    tokens: Sequence[Sequence[int] | Any],
    group_ids: Sequence[int | None] | None = None,
) -> str | None:
    """Return a cheap PTM runtime skip reason before building the full batch plan.

    This only uses micro-batch cardinality and local PTM group overlap metadata.
    It intentionally avoids any trie construction so the check stays cheap.
    """

    if len(tokens) <= 1:
        return "single_sample_microbatch"

    if group_ids is None:
        return None

    mergeable_group_ids: list[int] = []
    for gid in group_ids:
        if gid is None:
            continue
        gid = int(gid)
        if gid >= 0:
            mergeable_group_ids.append(gid)

    if len(mergeable_group_ids) < 2:
        return "no_mergeable_group_overlap"

    if len(set(mergeable_group_ids)) == len(mergeable_group_ids):
        return "no_mergeable_group_overlap"

    return None


def build_prefix_tree_batch_plan(tokens: Sequence[Sequence[int] | Any]) -> PrefixTreeBatchPlan:
    sequences: list[list[int]] = [_as_int_list(t) for t in tokens]
    if len(sequences) == 0:
        return PrefixTreeBatchPlan(
            merged_tokens=[],
            merged_position_ids=[],
            q_ranges=[],
            k_ranges=[],
            attn_type_map=[],
            unmerge_index=[],
            num_input_tokens=0,
            num_merged_tokens=0,
        )

    local_root = _RuntimeTrieNode.root()
    sample_paths: list[list[_RuntimeTrieNode]] = []
    for seq in sequences:
        node = local_root
        path: list[_RuntimeTrieNode] = []
        for token in seq:
            token = int(token)
            child = node.children.get(token)
            if child is None:
                child = _RuntimeTrieNode(token=token, parent=node, depth=node.depth + 1, children={})
                node.children[token] = child
            node = child
            path.append(node)
        sample_paths.append(path)

    merged_tokens: list[int] = []
    merged_position_ids: list[int] = []
    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    attn_type_map: list[int] = []

    local_root.index = -1
    stack: list[tuple[_RuntimeTrieNode, tuple[tuple[int, int], ...]]] = [
        (child, ()) for child in reversed(list(local_root.children.values()))
    ]
    while stack:
        start_node, ancestor_chain_intervals = stack.pop()

        chain_nodes: list[_RuntimeTrieNode] = [start_node]
        chain_tail = start_node
        while len(chain_tail.children) == 1:
            chain_tail = next(iter(chain_tail.children.values()))
            chain_nodes.append(chain_tail)

        chain_start = len(merged_tokens)
        for node in chain_nodes:
            node.index = len(merged_tokens)
            merged_tokens.append(int(node.token))
            merged_position_ids.append(max(int(node.depth) - 1, 0))
        chain_end = len(merged_tokens)

        chain_interval = (chain_start, chain_end)
        merged_path_segments = _merge_contiguous_intervals([*ancestor_chain_intervals, chain_interval])
        for segment_start, segment_end in merged_path_segments[:-1]:
            q_ranges.append([chain_start, chain_end])
            k_ranges.append([segment_start, segment_end])
            attn_type_map.append(0)

        causal_start, causal_end = merged_path_segments[-1]
        q_ranges.append([chain_start, chain_end])
        k_ranges.append([causal_start, causal_end])
        attn_type_map.append(1)

        child_ancestor_intervals = (*ancestor_chain_intervals, chain_interval)
        if chain_tail.children:
            for child in reversed(list(chain_tail.children.values())):
                stack.append((child, child_ancestor_intervals))

    unmerge_index: list[int] = []
    for path in sample_paths:
        indices = [node.index for node in path]
        unmerge_index.extend(indices)

    num_input_tokens = sum(len(seq) for seq in sequences)
    return PrefixTreeBatchPlan(
        merged_tokens=merged_tokens,
        merged_position_ids=merged_position_ids,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        unmerge_index=unmerge_index,
        num_input_tokens=num_input_tokens,
        num_merged_tokens=len(merged_tokens),
    )


def estimate_prefix_tree_merged_token_count(tokens: Sequence[Sequence[int] | Any]) -> int:
    merged_count = 0
    root = _TrieNode.new()
    for seq_like in tokens:
        node = root
        for token in _as_int_list(seq_like):
            child = node.children.get(token)
            if child is None:
                child = _TrieNode.new()
                node.children[token] = child
                merged_count += 1
            node = child
    return merged_count


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
