import logging
import os
from bisect import bisect_left
from argparse import Namespace
from collections.abc import Sequence
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from slime.utils import train_metric_utils
from slime.utils.data import get_minimum_num_micro_batch_size
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step
from slime.utils.prefix_tree_merging_utils import (
    PrefixTreeScheduleContext,
    build_prefix_tree_batch_plan,
    estimate_prefix_tree_runtime_batch_tokens_from_schedule_context,
    get_prefix_tree_runtime_skip_reason,
    is_ptm_debug_enabled,
    summarize_prefix_tree_batch_plan,
)
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.timer import Timer
from slime.utils.types import RolloutBatch

from ...utils import logging_utils
from .cp_utils import get_sum_of_sample_mean, slice_with_cp

logger = logging.getLogger(__name__)


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    qkv_format: str = "thd",
    allgather_cp: bool = False,
    enable_prefix_tree_merging: bool = False,
    profile_timer_prefix: str | None = None,
    prefix_tree_block_size: int = 1,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    """
    Generate a CP-ready micro-batch with packed sequence parameters.

    Steps:
    - Fetch raw fields via iterator.
    - Save original token tensors under "unconcat_tokens".
    - Slice tokens into two chunks for Context Parallelism (CP), concatenate, and pad to a configurable multiple.
    - Optionally run PTM trie merge (experimental): build merged token stream and TreeMask metadata.
    - Build cu_seqlens and `PackedSeqParams` with T-H-D layout (T: sequence length, H: attention heads, D: head dimension).

    Args:
        data_iterator: Iterator providing micro-batch data.
        keys: List of keys to fetch from the iterator.
        pad_multiplier: Multiplier for padding size calculation (default: 128).
        enable_prefix_tree_merging: Whether to enable PTM token merge + TreeMask metadata path.

    Returns a dict including:
    - "tokens": torch.LongTensor of shape [1, T_padded] on the current CUDA device
    - "position_ids": optional torch.LongTensor of shape [1, T_padded] when runtime PTM keeps
      original per-token positions for RoPE / absolute position embeddings
    - "unconcat_tokens": list[torch.LongTensor] for the micro-batch before CP slicing/concat
    - "packed_seq_params": PackedSeqParams with T-H-D settings (cu_seqlens on CUDA, dtype=int)
    Plus any other requested keys forwarded from the iterator.
    """

    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    if "dynamic_global_batch_size" in data_iterator.rollout_data:
        batch["dynamic_global_batch_size"] = data_iterator.rollout_data["dynamic_global_batch_size"]

    tokens = batch["tokens"]
    local_sample_indices = batch.get("local_sample_indices")
    # use 0 as the pad token id should be fine?
    pad_token_id = 0
    tp_size = mpu.get_tensor_model_parallel_world_size()
    pad_size = tp_size * pad_multiplier
    prefix_tree_block_size = max(int(prefix_tree_block_size), 1)

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens
    batch["position_ids"] = None

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    ptm_applied = False
    ptm_pad = 0
    pad = 0
    ptm_runtime_stats: dict[str, object] = {
        "requested": bool(enable_prefix_tree_merging),
        "applied": False,
        "skip_reason": "disabled",
        "runtime_block_size": prefix_tree_block_size,
    }

    if qkv_format == "bshd":
        if enable_prefix_tree_merging:
            ptm_runtime_stats["skip_reason"] = "unsupported_qkv_format_bshd"
        max_seqlen = batch["max_seq_lens"][0]
        assert max([t.size(0) for t in tokens]) <= max_seqlen
        tokens = [slice_with_cp(t, pad_token_id, qkv_format, max_seqlen) for t in tokens]
        tokens = torch.stack(tokens)
        packed_seq_params = None

    elif qkv_format == "thd":
        if enable_prefix_tree_merging:
            if allgather_cp:
                ptm_runtime_stats["skip_reason"] = "allgather_cp_enabled"
            elif cp_size != 1:
                ptm_runtime_stats["skip_reason"] = f"context_parallel_size_{cp_size}"
            else:
                ptm_runtime_stats["skip_reason"] = "no_runtime_token_reduction"
        enable_ptm_now = enable_prefix_tree_merging and not allgather_cp and cp_size == 1
        ptm_profile_enabled = profile_timer_prefix is not None and is_ptm_debug_enabled()
        if ptm_profile_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
            ptm_profile_start = perf_counter()
        else:
            ptm_profile_start = None
        if enable_ptm_now:
            cheap_skip_reason = get_prefix_tree_runtime_skip_reason(tokens)
            if cheap_skip_reason is not None:
                ptm_runtime_stats["skip_reason"] = cheap_skip_reason
            else:
                original_input_tokens = sum(int(t.size(0)) for t in tokens)
                original_pad = (pad_size - original_input_tokens % pad_size) % pad_size
                original_forward_tokens = original_input_tokens + original_pad
                ptm_runtime_stats.update(
                    {
                        "original_input_tokens": original_input_tokens,
                        "original_forward_tokens": original_forward_tokens,
                        "original_padded_tokens": original_pad,
                    }
                )

                exact_runtime_estimate = None
                schedule_contexts = data_iterator.rollout_data.get("ptm_schedule_contexts")
                if (
                    schedule_contexts is not None
                    and local_sample_indices is not None
                    and len(local_sample_indices) > 0
                    and len(schedule_contexts) > 0
                ):
                    num_local_samples = len(data_iterator.rollout_data["total_lengths"])
                    if num_local_samples % len(schedule_contexts) == 0:
                        num_local_gbs = num_local_samples // len(schedule_contexts)
                        if num_local_gbs > 0:
                            step_idx = local_sample_indices[0] // num_local_gbs
                            if (
                                all(sample_idx // num_local_gbs == step_idx for sample_idx in local_sample_indices)
                                and step_idx < len(schedule_contexts)
                            ):
                                exact_runtime_estimate = (
                                    estimate_prefix_tree_runtime_batch_tokens_from_schedule_context(
                                        schedule_contexts[step_idx],
                                        local_sample_indices,
                                        pad_size=pad_size if tp_size > 1 else 1,
                                        block_size=prefix_tree_block_size,
                                    )
                                )

                if exact_runtime_estimate is not None:
                    ptm_runtime_stats.update(
                        {
                            "cheap_gate_input_tokens": exact_runtime_estimate.num_input_tokens,
                            "cheap_gate_merged_tokens": exact_runtime_estimate.num_merged_tokens,
                            "cheap_gate_forward_tokens": exact_runtime_estimate.num_forward_tokens,
                            "cheap_gate_padded_tokens": exact_runtime_estimate.num_padded_tokens,
                            "cheap_gate_passed": False,
                        }
                    )
                    if exact_runtime_estimate.num_merged_tokens >= exact_runtime_estimate.num_input_tokens:
                        ptm_runtime_stats.update(
                            {
                                "skip_reason": "cheap_gate_no_runtime_token_reduction",
                                "num_input_tokens": exact_runtime_estimate.num_input_tokens,
                                "num_merged_tokens": exact_runtime_estimate.num_merged_tokens,
                            }
                        )
                    elif exact_runtime_estimate.num_forward_tokens >= original_forward_tokens:
                        ptm_runtime_stats.update(
                            {
                                "skip_reason": "cheap_gate_no_forward_token_reduction",
                                "num_input_tokens": exact_runtime_estimate.num_input_tokens,
                                "num_merged_tokens": exact_runtime_estimate.num_merged_tokens,
                            }
                        )
                    else:
                        ptm_runtime_stats["cheap_gate_passed"] = True

                if ptm_runtime_stats.get("skip_reason") not in {
                    "cheap_gate_no_runtime_token_reduction",
                    "cheap_gate_no_forward_token_reduction",
                }:
                    ptm_plan = build_prefix_tree_batch_plan(tokens, block_size=prefix_tree_block_size)
                    ptm_runtime_stats.update(
                        {
                            "num_input_tokens": ptm_plan.num_input_tokens,
                            "num_merged_tokens": ptm_plan.num_merged_tokens,
                            "runtime_block_size": ptm_plan.runtime_block_size,
                            "num_input_blocks": ptm_plan.num_input_blocks,
                            "num_merged_blocks": ptm_plan.num_merged_blocks,
                            "num_block_suffix_tokens": ptm_plan.num_block_suffix_tokens,
                        }
                    )
                    ptm_runtime_stats.update(summarize_prefix_tree_batch_plan(ptm_plan))
                    if 0 < ptm_plan.num_merged_tokens < ptm_plan.num_input_tokens:
                        device = tokens[0].device
                        dtype = tokens[0].dtype
                        merged_tokens = torch.tensor(ptm_plan.merged_tokens, dtype=dtype, device=device)
                        merged_position_ids = torch.tensor(
                            ptm_plan.merged_position_ids, dtype=torch.long, device=device
                        )
                        # PTM still runs in THD/CP=1 mode, but TP/sequence parallel may require
                        # the token stream length to be padded to a TP-aligned multiple.
                        ptm_pad = (pad_size - merged_tokens.size(0) % pad_size) % pad_size if tp_size > 1 else 0
                        if ptm_pad != 0:
                            merged_tokens = F.pad(merged_tokens, (0, ptm_pad), value=pad_token_id)
                            merged_position_ids = F.pad(merged_position_ids, (0, ptm_pad), value=0)
                        cu_seqlens = torch.tensor(
                            [0, merged_tokens.size(0)], dtype=torch.int, device=torch.cuda.current_device()
                        )
                        max_seqlen = merged_tokens.size(0)
                        packed_seq_params = PackedSeqParams(
                            cu_seqlens_q=cu_seqlens,
                            cu_seqlens_kv=cu_seqlens,
                            max_seqlen_q=max_seqlen,
                            max_seqlen_kv=max_seqlen,
                            qkv_format="thd",
                        )
                        # Keep the constructor ABI compatible with older Megatron runtimes used
                        # in cloud jobs, then attach PTM metadata dynamically.
                        packed_seq_params.ptm_q_ranges = torch.tensor(
                            ptm_plan.q_ranges, dtype=torch.int32, device=torch.cuda.current_device()
                        )
                        packed_seq_params.ptm_k_ranges = torch.tensor(
                            ptm_plan.k_ranges, dtype=torch.int32, device=torch.cuda.current_device()
                        )
                        packed_seq_params.ptm_attn_type_map = torch.tensor(
                            ptm_plan.attn_type_map, dtype=torch.int32, device=torch.cuda.current_device()
                        )
                        packed_seq_params.ptm_q_ranges_non_overlapped = bool(
                            ptm_runtime_stats.get("q_ranges_non_overlapped", 0)
                        )
                        packed_seq_params.explicit_position_ids = True
                        tokens = merged_tokens.unsqueeze(0)
                        batch["position_ids"] = merged_position_ids.unsqueeze(0)
                        batch["ptm_unmerge_index"] = torch.tensor(
                            ptm_plan.unmerge_index, dtype=torch.long, device=torch.cuda.current_device()
                        )
                        ptm_runtime_stats.update(
                            {
                                "applied": True,
                                "skip_reason": "applied",
                                "max_position_id": max(ptm_plan.merged_position_ids, default=-1),
                                "num_unique_position_ids": len(set(ptm_plan.merged_position_ids)),
                            }
                        )
                        ptm_applied = True
        if ptm_profile_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_s = perf_counter() - ptm_profile_start
            Timer().add(f"{profile_timer_prefix}_ptm", elapsed_s)

        if ptm_applied:
            pass
        elif allgather_cp:
            # DSA mode: concatenate all sequences first, then slice once with CP.
            # We also pad the *global* concatenated stream to make per-rank chunks equal.
            cu_seqlens_list: list[int] = [0]
            for t in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

            tokens = torch.cat(tokens, dim=0)

            # Pad global stream so (1) divisible by cp_size (equal chunks),
            # (2) divisible by pad_size (reduce fragmentation).
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

            cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=torch.cuda.current_device())
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        else:
            tokens = [slice_with_cp(t, pad_token_id, qkv_format) for t in tokens]

            cu_seqlens = [0]
            for t in tokens:
                cu_seqlens.append(cu_seqlens[-1] + t.size(0))

            tokens = torch.cat(tokens)

            # Always pad to reduce memory fragmentation and maybe make the computation faster
            pad = (pad_size - tokens.size(0) % pad_size) % pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens.append(cu_seqlens[-1] + pad)

            # thd requires the cu_seqlens to be of the origin length
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size

        if not ptm_applied:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
                qkv_format="thd",
            )
            tokens = tokens.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params
    if ptm_runtime_stats["requested"]:
        ptm_runtime_stats["num_forward_tokens"] = int(tokens.numel())
        ptm_runtime_stats["num_padded_tokens"] = int(ptm_pad if ptm_applied else pad)
    batch["ptm_runtime_stats"] = ptm_runtime_stats

    # loss masks
    if ptm_applied:
        # PTM no-grad path currently does not consume token-level loss masks.
        # Keep padded tail tokens masked out so TP alignment padding remains inert.
        loss_masks = torch.ones_like(tokens, dtype=torch.float32)
        if ptm_pad != 0:
            loss_masks[:, -ptm_pad:] = 0
    else:
        loss_masks = []
        for loss_mask, total_length, response_length in zip(
            batch["loss_masks"],
            batch["total_lengths"],
            batch["response_lengths"],
            strict=True,
        ):
            prompt_length = total_length - response_length
            # Align mask to token stream positions (prompt_length-1 left pad, 1 right pad)
            loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
            if allgather_cp:
                loss_masks.append(loss_mask)
                continue
            loss_mask = slice_with_cp(loss_mask, 0, qkv_format, max_seqlen)
            loss_masks.append(loss_mask)

        if qkv_format == "bshd":
            loss_masks = torch.stack(loss_masks)
        elif qkv_format == "thd" and allgather_cp:
            # DSA: concatenate first (same as tokens), pad globally (same pad as above), then slice once.
            loss_masks = torch.cat(loss_masks, dim=0)
            if pad != 0:
                loss_masks = F.pad(loss_masks, (0, pad), value=0)
            loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
        elif qkv_format == "thd":
            loss_masks = torch.cat(loss_masks)
            loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}  # key -> concatenated tensor
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
        batch["multimodal_train_inputs"] = multimodal_data

    return batch


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, float],
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce by mean on the DP source rank, and log.

    Expects `log_dict` to contain plain scalars. The DP source rank prints and
    optionally logs to WandB/TensorBoard with a step derived from `rollout_id` and
    batch sizes. Returns the reduced dict on the DP source rank; returns None on others.
    """

    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)

        gathered_log_dict = [None] * dp_size
        # Not sure if this will be a performance bottleneck.
        dist.gather_object(
            log_dict,
            gathered_log_dict,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )

        reduced_log_dict = {
            f"{metric_name}/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
        }
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        logging_utils.log(args, reduced_log_dict, step_key="rollout/step")

        return reduced_log_dict
    else:
        dist.gather_object(
            log_dict,
            None,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )
        return None


class DataIterator:
    """Micro-batch iterator over rollout dicts.

    Supports either fixed contiguous micro-batches or an explicit per-step
    index schedule (for dynamic batch sizing / sequence-length balancing).
    """

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_size: int | None = None,
        micro_batch_indices: list[list[int]] | None = None,
    ) -> None:
        """Initialize an iterator over `rollout_data`.

        Args:
            rollout_data: Dict of per-sample fields for the local step.
            micro_batch_size: Fixed contiguous slice size when not using dynamic scheduling.
            micro_batch_indices: Explicit indices per micro-batch when using dynamic balancing.
                Must be mutually exclusive with `micro_batch_size`.
        """
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        assert micro_batch_size is None or micro_batch_indices is None
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        - If `micro_batch_indices` is provided, selects rows according to the current
          index list for each requested key.
        - Otherwise, slices a contiguous window of size `micro_batch_size` starting
          at the current offset.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        if self.micro_batch_indices is not None:
            local_sample_indices = [int(idx) for idx in self.micro_batch_indices[self.offset]]
        else:
            batch_size = int(self.micro_batch_size)
            local_sample_indices = list(range(self.offset, self.offset + batch_size))
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                if self.micro_batch_indices is not None:
                    batch[key] = [vals[i] for i in local_sample_indices]
                else:
                    assert self.offset + self.micro_batch_size <= len(
                        vals
                    ), f"offset: {self.offset}, micro_batch_size: {self.micro_batch_size}, len(vals): {len(vals)}"
                    batch[key] = vals[self.offset : self.offset + self.micro_batch_size]

        batch["local_sample_indices"] = local_sample_indices

        if self.micro_batch_indices is not None:
            self.offset += 1
        else:
            self.offset += self.micro_batch_size
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def get_data_iterator(
    args: Namespace,
    model: torch.nn.Module | Sequence[torch.nn.Module],
    rollout_data: RolloutBatch,
    force_single_sample_microbatch: bool = False,
    enable_ptm_aware_dynamic_batching: bool | None = None,
    timer_prefix: str | None = None,
) -> tuple[list[DataIterator], list[int]]:
    """
    Create iterators and a micro-batch schedule for a rollout step.

    - If `force_single_sample_microbatch` is True, each sample is assigned to its own
      micro-batch (primarily for no-grad logprob debugging/prototyping).
    - `enable_ptm_aware_dynamic_batching` controls whether dynamic batching may use
      PTM-compressed token estimates instead of original `total_lengths`. When unset,
      the existing PTM capability checks decide automatically.
    - Else if `use_dynamic_batch_size` is False, splits into fixed-size contiguous
      micro-batches of `micro_batch_size`.
    - If True, computes the number of micro-batches per local step based on
      `max_tokens_per_gpu` and per-sample lengths, all-reduces to a DP-wide
      maximum, optionally enforces divisibility for Virtual Pipeline Parallelism (VPP), and builds a balanced
      index schedule to equalize token counts across micro-batches.

    Returns `(data_iterators, num_microbatches)` where:
    - `data_iterators`: list of `DataIterator`, one per VPP stage (size 1 if VPP disabled)
    - `num_microbatches`: list[int], one per local step in the rollout (length = steps)
    """
    iterator_start_time = perf_counter()
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dp_group = mpu.get_data_parallel_group()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1
    if vpp_size > 1:
        from megatron.core.utils import get_model_config

        config = get_model_config(model[0])
        microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
    cp_size = mpu.get_context_parallel_world_size()

    num_local_samples = len(rollout_data["total_lengths"])
    global_batch_size = rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
    num_local_gbs = global_batch_size // dp_size
    num_steps_per_rollout = num_local_samples // num_local_gbs

    if global_batch_size != args.global_batch_size:
        logger.info(
            f"Using dynamic global_batch_size={global_batch_size} (original={args.global_batch_size}), "
            f"num_local_samples={num_local_samples}, num_steps_per_rollout={num_steps_per_rollout}"
        )

    ptm_debug_enabled = is_ptm_debug_enabled()
    track_iterator_timing = timer_prefix is not None and (
        timer_prefix != "actor_logprob_data_iterator" or ptm_debug_enabled
    )

    def _add_timing(metric_name: str, elapsed_s: float) -> None:
        if track_iterator_timing:
            Timer().add(metric_name, elapsed_s)

    def _should_log_schedule_details() -> bool:
        return ptm_debug_enabled and (not dist.is_initialized() or dist.get_rank() == 0)

    def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
        data_iterator = []
        for _ in range(vpp_size):
            data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
        return data_iterator

    pad_size = mpu.get_tensor_model_parallel_world_size() * args.data_pad_size_multiplier
    ptm_runtime_block_size = max(int(getattr(args, "slime_prefix_runtime_block_size", 1)), 1)

    def _pad_ptm_sched_tokens(merged_tokens: int) -> int:
        if merged_tokens <= 0:
            return 0
        return merged_tokens + ((pad_size - merged_tokens % pad_size) % pad_size)

    ptm_sched_metrics: dict[str, int | float] = {
        "build_calls": 0,
        "candidate_evals": 0,
        "estimate_calls": 0,
        "estimate_cache_hits": 0,
        "estimate_cache_misses": 0,
        "estimate_time_s": 0.0,
        "reused_first_pass_steps": 0,
    }

    class _ScheduleBucket:
        __slots__ = ("sample_indices", "sorted_ranks", "merged_tokens", "padded_tokens")

        def __init__(self, sample_idx: int, schedule_ctx: PrefixTreeScheduleContext):
            rank = schedule_ctx.get_rank(sample_idx)
            merged_tokens = schedule_ctx.get_length(sample_idx)
            self.sample_indices = [int(sample_idx)]
            self.sorted_ranks = [rank]
            self.merged_tokens = merged_tokens
            self.padded_tokens = _pad_ptm_sched_tokens(merged_tokens)

        def evaluate_insert(
            self,
            sample_idx: int,
            schedule_ctx: PrefixTreeScheduleContext,
        ) -> tuple[int, int, int]:
            rank = schedule_ctx.get_rank(sample_idx)
            insert_at = bisect_left(self.sorted_ranks, rank)

            prev_rank = self.sorted_ranks[insert_at - 1] if insert_at > 0 else None
            next_rank = self.sorted_ranks[insert_at] if insert_at < len(self.sorted_ranks) else None

            merged_tokens = self.merged_tokens + schedule_ctx.get_length(sample_idx)
            if prev_rank is not None:
                merged_tokens -= (
                    schedule_ctx.get_lcp_by_rank(prev_rank, rank) // ptm_runtime_block_size
                ) * ptm_runtime_block_size
            if next_rank is not None:
                merged_tokens -= (
                    schedule_ctx.get_lcp_by_rank(rank, next_rank) // ptm_runtime_block_size
                ) * ptm_runtime_block_size
            if prev_rank is not None and next_rank is not None:
                merged_tokens += (
                    schedule_ctx.get_lcp_by_rank(prev_rank, next_rank) // ptm_runtime_block_size
                ) * ptm_runtime_block_size

            padded_tokens = _pad_ptm_sched_tokens(merged_tokens)
            return padded_tokens, merged_tokens, insert_at

        def commit_insert(
            self,
            sample_idx: int,
            schedule_ctx: PrefixTreeScheduleContext,
            padded_tokens: int | None = None,
            merged_tokens: int | None = None,
            insert_at: int | None = None,
        ) -> None:
            if padded_tokens is None or merged_tokens is None or insert_at is None:
                padded_tokens, merged_tokens, insert_at = self.evaluate_insert(sample_idx, schedule_ctx)
            self.sample_indices.append(int(sample_idx))
            self.sample_indices.sort()
            self.sorted_ranks.insert(int(insert_at), schedule_ctx.get_rank(sample_idx))
            self.merged_tokens = int(merged_tokens)
            self.padded_tokens = int(padded_tokens)

    def _build_ptm_aware_microbatches(
        step_indices: list[int],
        token_budget: int,
        target_num_mbs: int,
        require_exact_count: bool = False,
        schedule_ctx: PrefixTreeScheduleContext | None = None,
    ) -> list[list[int]]:
        if not step_indices:
            return []

        assert target_num_mbs >= 1, f"target_num_mbs must be >= 1, got {target_num_mbs}"
        assert target_num_mbs <= len(step_indices), (
            f"target_num_mbs {target_num_mbs} exceeds sample count {len(step_indices)}"
        )

        total_lengths = rollout_data["total_lengths"]
        group_sizes = rollout_data.get("ptm_group_sizes")
        prefix_lens = rollout_data.get("ptm_prefix_lens")
        if schedule_ctx is None:
            raise ValueError("schedule_ctx is required for PTM-aware dynamic batching")
        if ptm_debug_enabled:
            ptm_sched_metrics["build_calls"] += 1

        def _prefix_len(sample_idx: int) -> int:
            if prefix_lens is None:
                return 0
            return int(prefix_lens[sample_idx])

        def _group_size(sample_idx: int) -> int:
            if group_sizes is None:
                return 1
            return int(group_sizes[sample_idx])

        def _suffix_len(sample_idx: int) -> int:
            return max(int(total_lengths[sample_idx]) - _prefix_len(sample_idx), 0)

        prioritized = sorted(
            step_indices,
            key=lambda idx: (
                -_prefix_len(idx),
                -_group_size(idx),
                _suffix_len(idx),
                -int(total_lengths[idx]),
                idx,
            ),
        )

        buckets: list[_ScheduleBucket] = []

        for position, sample_idx in enumerate(prioritized):
            remaining_samples = len(prioritized) - position
            remaining_buckets_to_open = target_num_mbs - len(buckets)
            force_open_new_bucket = (
                require_exact_count
                and len(buckets) < target_num_mbs
                and remaining_samples == remaining_buckets_to_open
            )

            if force_open_new_bucket:
                buckets.append(_ScheduleBucket(sample_idx, schedule_ctx))
                continue

            feasible_candidates: list[tuple[int, int, int, int, int, int, int]] = []
            overflow_candidates: list[tuple[int, int, int, int, int, int, int]] = []

            for slot_idx, bucket in enumerate(buckets):
                if ptm_debug_enabled:
                    ptm_sched_metrics["candidate_evals"] += 1
                    estimate_start_time = perf_counter()
                candidate_cost, merged_tokens, insert_at = bucket.evaluate_insert(sample_idx, schedule_ctx)
                if ptm_debug_enabled:
                    ptm_sched_metrics["estimate_time_s"] += perf_counter() - estimate_start_time
                    ptm_sched_metrics["estimate_calls"] += 1
                    ptm_sched_metrics["estimate_cache_misses"] += 1
                marginal_cost = candidate_cost - bucket.padded_tokens
                slack = token_budget - candidate_cost
                candidate_info = (
                    slot_idx,
                    candidate_cost,
                    marginal_cost,
                    slack,
                    len(bucket.sample_indices),
                    merged_tokens,
                    insert_at,
                )
                if candidate_cost <= token_budget:
                    feasible_candidates.append(candidate_info)
                else:
                    overflow_candidates.append(candidate_info)

            if feasible_candidates:
                best_slot, best_cost, _, _, _, best_merged_tokens, best_insert_at = min(
                    feasible_candidates,
                    key=lambda item: (
                        item[2],
                        -item[4],
                        item[3],
                        item[0],
                    ),
                )
                buckets[best_slot].commit_insert(
                    sample_idx,
                    schedule_ctx,
                    padded_tokens=best_cost,
                    merged_tokens=best_merged_tokens,
                    insert_at=best_insert_at,
                )
                continue

            if len(buckets) < target_num_mbs:
                buckets.append(_ScheduleBucket(sample_idx, schedule_ctx))
                continue

            if not overflow_candidates:
                raise RuntimeError("PTM scheduler could not place sample into any existing bucket.")

            best_slot, best_cost, _, _, _, best_merged_tokens, best_insert_at = min(
                overflow_candidates,
                key=lambda item: (
                    item[1] - token_budget,
                    item[2],
                    -item[4],
                    item[0],
                ),
            )
            buckets[best_slot].commit_insert(
                sample_idx,
                schedule_ctx,
                padded_tokens=best_cost,
                merged_tokens=best_merged_tokens,
                insert_at=best_insert_at,
            )

        partitions = [bucket.sample_indices.copy() for bucket in buckets if bucket.sample_indices]
        if require_exact_count:
            assert len(partitions) == target_num_mbs, f"{len(partitions)} != {target_num_mbs}"
        return partitions

    if force_single_sample_microbatch:
        if vpp_size > 1:
            raise ValueError("force_single_sample_microbatch does not support virtual pipeline parallelism yet.")
        num_microbatches = [num_local_gbs for _ in range(num_steps_per_rollout)]
        micro_batch_indices = []
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            micro_batch_indices.extend([[sample_idx] for sample_idx in range(start, end)])
        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)
    elif not args.use_dynamic_batch_size:
        num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
        data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    else:
        assert args.max_tokens_per_gpu is not None
        ptm_sched_supported = bool(
            args.slime_prefix_magi_attention
            and args.slime_prefix_tree_merging
            and args.qkv_format == "thd"
            and cp_size == 1
            and not args.allgather_cp
        )
        if enable_ptm_aware_dynamic_batching is None:
            use_ptm_sched = ptm_sched_supported
        else:
            use_ptm_sched = bool(enable_ptm_aware_dynamic_batching and ptm_sched_supported)
        samples = rollout_data["total_lengths"]
        assert len(samples) == num_local_samples
        token_budget = args.max_tokens_per_gpu * cp_size
        num_microbatches = []
        original_num_microbatches = []
        ptm_sched_first_pass_elapsed = 0.0
        ptm_sched_second_pass_elapsed = 0.0
        first_pass_partitions: list[list[list[int]] | None] = [None] * num_steps_per_rollout
        step_schedule_contexts = rollout_data.get("ptm_schedule_contexts")
        first_pass_start_time = perf_counter() if ptm_debug_enabled else None
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            original_num_microbatches.append(get_minimum_num_micro_batch_size(samples[start:end], token_budget))
            if use_ptm_sched:
                step_indices = list(range(start, end))
                if step_schedule_contexts is None or i >= len(step_schedule_contexts):
                    raise ValueError(f"Missing ptm_schedule_contexts for step {i}")
                step_partitions = _build_ptm_aware_microbatches(
                    step_indices,
                    token_budget,
                    target_num_mbs=len(step_indices),
                    require_exact_count=False,
                    schedule_ctx=step_schedule_contexts[i],
                )
                first_pass_partitions[i] = step_partitions
                num_microbatches.append(len(step_partitions))
            else:
                num_microbatches.append(original_num_microbatches[-1])
        if ptm_debug_enabled and first_pass_start_time is not None:
            ptm_sched_first_pass_elapsed = perf_counter() - first_pass_start_time

        sync_start_time = perf_counter()
        original_num_microbatches = torch.tensor(
            original_num_microbatches, dtype=torch.int, device=torch.cuda.current_device()
        )
        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        dist.all_reduce(original_num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)

        if vpp_size > 1:
            # vpp requies the number of microbatches to be divisible by vpp_size
            original_num_microbatches = torch.clamp(
                original_num_microbatches // microbatch_group_size_per_vp_stage * microbatch_group_size_per_vp_stage,
                min=1,
            )
            num_microbatches = torch.clamp(
                num_microbatches // microbatch_group_size_per_vp_stage * microbatch_group_size_per_vp_stage,
                min=1,
            )

        original_num_microbatches = original_num_microbatches.tolist()
        num_microbatches = num_microbatches.tolist()
        sync_elapsed = perf_counter() - sync_start_time

        micro_batch_indices = []
        second_pass_start_time = perf_counter() if ptm_debug_enabled else None
        if use_ptm_sched:
            for i, target_num_mbs in enumerate(num_microbatches):
                start, end = i * num_local_gbs, (i + 1) * num_local_gbs
                step_indices = list(range(start, end))
                local_first_pass_partitions = first_pass_partitions[i]
                if local_first_pass_partitions is not None and len(local_first_pass_partitions) == target_num_mbs:
                    partitions = [partition.copy() for partition in local_first_pass_partitions]
                    if ptm_debug_enabled:
                        ptm_sched_metrics["reused_first_pass_steps"] += 1
                else:
                    if step_schedule_contexts is None or i >= len(step_schedule_contexts):
                        raise ValueError(f"Missing ptm_schedule_contexts for step {i}")
                    partitions = _build_ptm_aware_microbatches(
                        step_indices,
                        token_budget,
                        target_num_mbs=target_num_mbs,
                        require_exact_count=True,
                        schedule_ctx=step_schedule_contexts[i],
                    )
                micro_batch_indices.extend(partitions)
        else:
            samples = rollout_data["total_lengths"]
            for i, num_mbs in enumerate(num_microbatches):
                start, end = i * num_local_gbs, (i + 1) * num_local_gbs
                samples = rollout_data["total_lengths"][start:end]
                partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
                for j in range(num_mbs):
                    for k in range(len(partitions[j])):
                        partitions[j][k] += start
                micro_batch_indices.extend(partitions)
        if ptm_debug_enabled and second_pass_start_time is not None:
            ptm_sched_second_pass_elapsed = perf_counter() - second_pass_start_time

        assert len(set(sum(micro_batch_indices, []))) == num_local_samples

        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)

        total_original_microbatches = int(sum(original_num_microbatches))
        total_ptm_microbatches = int(sum(num_microbatches))
        batch_reduction_ratio = (
            total_original_microbatches / total_ptm_microbatches if total_ptm_microbatches > 0 else 0.0
        )
        _add_timing(f"{timer_prefix}_original_microbatches", float(total_original_microbatches))
        _add_timing(f"{timer_prefix}_ptm_microbatches", float(total_ptm_microbatches))
        _add_timing(f"{timer_prefix}_batch_reduction_ratio", batch_reduction_ratio)

        if use_ptm_sched and ptm_debug_enabled:
            if _should_log_schedule_details():
                logger.info(
                    "[PTMScheduler] stage=%s steps=%d token_budget=%d "
                    "runtime_block_size=%d "
                    "first_pass_time_s=%.3f second_pass_time_s=%.3f sync_time_s=%.3f "
                    "estimate_time_s=%.3f estimate_calls=%d estimate_cache_hits=%d "
                    "estimate_cache_misses=%d candidate_evals=%d build_calls=%d "
                    "reused_first_pass_steps=%d/%d original_num_microbatches=%s "
                    "ptm_num_microbatches=%s total_original_microbatches=%d "
                    "total_ptm_microbatches=%d batch_reduction_ratio=%.3f",
                    timer_prefix or "data_iterator",
                    num_steps_per_rollout,
                    token_budget,
                    ptm_runtime_block_size,
                    ptm_sched_first_pass_elapsed,
                    ptm_sched_second_pass_elapsed,
                    sync_elapsed,
                    float(ptm_sched_metrics["estimate_time_s"]),
                    int(ptm_sched_metrics["estimate_calls"]),
                    int(ptm_sched_metrics["estimate_cache_hits"]),
                    int(ptm_sched_metrics["estimate_cache_misses"]),
                    int(ptm_sched_metrics["candidate_evals"]),
                    int(ptm_sched_metrics["build_calls"]),
                    int(ptm_sched_metrics["reused_first_pass_steps"]),
                    num_steps_per_rollout,
                    original_num_microbatches,
                    num_microbatches,
                    total_original_microbatches,
                    total_ptm_microbatches,
                    batch_reduction_ratio,
                )
        _add_timing(f"{timer_prefix}_microbatch_sync", sync_elapsed)

    total_iterator_elapsed = perf_counter() - iterator_start_time
    _add_timing(timer_prefix or "", total_iterator_elapsed)
    return (
        data_iterator,
        num_microbatches,
    )


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """
    Summarize rollout fields and log reduced metrics on PP last stage, TP rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Numeric non-tensor lists are averaged elementwise.
    - Opaque helper/raw-data lists (for example cached CPU tokens or scheduler
      contexts) are skipped instead of being treated as metrics.
    - Scalars are converted to Python numbers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        max_seq_lens = rollout_data.get("max_seq_lens", None)

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_routed_experts",
                "max_seq_lens",
                "dynamic_global_batch_size",
                "tokens_cpu",
                "ptm_schedule_contexts",
            ]:
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, (list, tuple)):
                if len(val) == 0:
                    continue
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    if key in [
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "teacher_log_probs",
                        "opd_reverse_kl",
                    ]:
                        val = torch.cat(val).clone().detach()
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                        )
                        val = cp_size * sum_of_sample_mean(val) / len(loss_masks)
                    else:
                        val = torch.cat(val).clone().detach()
                        val = val.mean() * cp_size
                else:
                    if all(isinstance(item, (int, float, bool, np.integer, np.floating)) for item in val):
                        val = sum(float(item) for item in val) / len(val)
                    else:
                        logger.debug(
                            "Skip non-numeric rollout field %s from logging (element type: %s)",
                            key,
                            type(val[0]).__name__,
                        )
                        continue
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            elif isinstance(val, (int, float, np.integer, np.floating)):
                val = float(val)
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        if args.ci_test and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # TODO: figure out why there is a small numerical difference in log_probs and ref_log_probs in CI test, and whether it's expected or not.
                # assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
            if "rollout/log_probs" in reduced_log_dict:
                assert -0.5 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.5

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)

    if args.log_correct_samples:
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
            cp_size = mpu.get_context_parallel_world_size()
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            def quantile(total_value, n_quantiles, data) -> dict:
                import math

                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                percentile = {f"p{min(math.ceil(q*100),100)}": p for q, p in zip(quantiles, percentile, strict=True)}
                return percentile

            raw_rewards = rollout_data["raw_reward"]
            # Additional metrics for correct cases are calculated separately below.
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    correct_entropy.append(-rollout_data["log_probs"][i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Log multi-turn auxiliary metrics such as raw/observed response lengths and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Compute pass@k metrics from `raw_reward` groups and log the results.

    `raw_reward` is reshaped to `[group_number, group_size]`, then pass@k is
    estimated per problem and averaged.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data(rollout_id: int, args: Namespace) -> None:
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.is_pipeline_last_stage()
            and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
    )


def sync_actor_critic_data(
    args: Namespace,
    rollout_data: RolloutBatch | None = None,
    group: dist.ProcessGroup | None = None,
) -> None:
    """
    Broadcast `values` (from critic) and optionally `log_probs`/`ref_log_probs`
    (from actor) across PP ranks to align data dependencies.

    - Values are broadcast from src=1.
    - Log-probs and ref-log-probs are broadcast from src=0 when KL is used.
    Updates `rollout_data` in place with the synchronized tensors.
    """
    log_probs_key = "log_probs" if not args.use_rollout_logprobs else "rollout_log_probs"
    values, log_probs, ref_log_probs = map(rollout_data.get, ("values", log_probs_key, "ref_log_probs"))

    # return when not the pp last stage
    if not values and not log_probs:
        return

    handles = []

    if not values:
        values = [torch.empty_like(log_prob) for log_prob in log_probs]
    for value in values:
        handles.append(dist.broadcast(value, src=1, group=group, async_op=True))

    if args.kl_coef != 0 or args.use_kl_loss:
        if not log_probs:
            log_probs = [torch.empty_like(value) for value in values]
        if not ref_log_probs:
            ref_log_probs = [torch.empty_like(value) for value in values]
        for ref_log_prob, log_prob in zip(ref_log_probs, log_probs, strict=False):
            handles.append(dist.broadcast(log_prob, src=0, group=group, async_op=True))
            handles.append(dist.broadcast(ref_log_prob, src=0, group=group, async_op=True))

    for handle in handles:
        handle.wait()

    rollout_data.update(
        {
            k: v
            for k, v in {
                "values": values,
                log_probs_key: log_probs,
                "ref_log_probs": ref_log_probs,
            }.items()
            if v is not None
        }
    )
