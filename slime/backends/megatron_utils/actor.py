import logging
import os
import random
import socket
from argparse import Namespace
from contextlib import nullcontext
from time import perf_counter

import numpy as np
import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_cuda_memory_stats, print_memory
from slime.utils.prefix_tree_merging_utils import (
    build_prefix_tree_context_from_rollout_data,
    format_ptm_debug_timing_metrics,
    is_ptm_debug_enabled,
    log_prefix_tree_context,
)
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp, slice_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data, sync_actor_critic_data
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        }
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep(reason="after_init")
            return start_rollout_id

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size (which may be smaller, e.g. GPT-OSS).
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep(reason="after_init")

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    def _log_ptm_debug_offload_transition(
        self,
        transition: str,
        timings: dict[str, float],
        total_time: float,
        reason: str,
        extra_metrics: dict[str, object] | None = None,
    ) -> None:
        if not is_ptm_debug_enabled():
            return

        role = getattr(self, "role", None)
        rank = getattr(self.args, "rank", getattr(self, "_rank", None))
        metrics: dict[str, object] = {
            "transition": transition,
            "reason": reason,
            "rank": int(rank) if rank is not None else None,
            "role": role,
            "call_idx": int(getattr(self, f"_{transition}_call_idx", 0)),
            f"perf/{transition}_time": total_time,
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        logger.info(
            f"[PTMDebug] {transition} breakdown: "
            f"{format_ptm_debug_timing_metrics(timings, extra_metrics=metrics)}"
        )

    @timer
    def sleep(self, reason: str = "unspecified") -> None:
        assert self.args.offload_train

        ptm_debug_enabled = is_ptm_debug_enabled()
        if ptm_debug_enabled:
            self._sleep_call_idx = getattr(self, "_sleep_call_idx", 0) + 1
            sleep_total_start = perf_counter()
            sleep_timings: dict[str, float] = {}
            pre_clear_cuda_stats = print_cuda_memory_stats(f"before {reason} clear_memory")
        else:
            sleep_total_start = None
            sleep_timings = {}
            pre_clear_cuda_stats = None

        clear_memory_start = perf_counter() if ptm_debug_enabled else None
        clear_memory(clear_host_memory=True)
        if clear_memory_start is not None:
            elapsed = perf_counter() - clear_memory_start
            Timer().add("sleep_clear_memory", elapsed)
            sleep_timings["sleep_clear_memory"] = elapsed
        print_memory("before offload model")
        if ptm_debug_enabled:
            pre_pause_cuda_stats = print_cuda_memory_stats(f"before {reason} torch_memory_saver.pause")
        else:
            pre_pause_cuda_stats = None

        destroy_pg_start = perf_counter() if ptm_debug_enabled else None
        destroy_process_groups()
        if destroy_pg_start is not None:
            elapsed = perf_counter() - destroy_pg_start
            Timer().add("sleep_destroy_process_groups", elapsed)
            sleep_timings["sleep_destroy_process_groups"] = elapsed

        pause_start = perf_counter() if ptm_debug_enabled else None
        torch_memory_saver.pause()
        if pause_start is not None:
            elapsed = perf_counter() - pause_start
            Timer().add("sleep_torch_memory_saver_pause", elapsed)
            sleep_timings["sleep_torch_memory_saver_pause"] = elapsed

        print_memory("after offload model")
        if ptm_debug_enabled:
            post_pause_cuda_stats = print_cuda_memory_stats(f"after {reason} torch_memory_saver.pause")
        else:
            post_pause_cuda_stats = None
        if sleep_total_start is not None:
            self._log_ptm_debug_offload_transition(
                transition="sleep",
                timings=sleep_timings,
                total_time=perf_counter() - sleep_total_start,
                reason=reason,
                extra_metrics={
                    "pre_clear_reserved_GB": None if pre_clear_cuda_stats is None else pre_clear_cuda_stats["reserved_GB"],
                    "pre_clear_active_GB": None if pre_clear_cuda_stats is None else pre_clear_cuda_stats["active_GB"],
                    "pre_clear_inactive_split_GB": (
                        None if pre_clear_cuda_stats is None else pre_clear_cuda_stats["inactive_split_GB"]
                    ),
                    "pre_clear_alloc_retries": (
                        None if pre_clear_cuda_stats is None else pre_clear_cuda_stats["alloc_retries"]
                    ),
                    "pre_clear_ooms": None if pre_clear_cuda_stats is None else pre_clear_cuda_stats["ooms"],
                    "pre_pause_reserved_GB": None if pre_pause_cuda_stats is None else pre_pause_cuda_stats["reserved_GB"],
                    "pre_pause_active_GB": None if pre_pause_cuda_stats is None else pre_pause_cuda_stats["active_GB"],
                    "pre_pause_inactive_split_GB": (
                        None if pre_pause_cuda_stats is None else pre_pause_cuda_stats["inactive_split_GB"]
                    ),
                    "pre_pause_alloc_retries": (
                        None if pre_pause_cuda_stats is None else pre_pause_cuda_stats["alloc_retries"]
                    ),
                    "pre_pause_ooms": None if pre_pause_cuda_stats is None else pre_pause_cuda_stats["ooms"],
                    "post_pause_reserved_GB": (
                        None if post_pause_cuda_stats is None else post_pause_cuda_stats["reserved_GB"]
                    ),
                    "post_pause_active_GB": None if post_pause_cuda_stats is None else post_pause_cuda_stats["active_GB"],
                    "post_pause_inactive_split_GB": (
                        None if post_pause_cuda_stats is None else post_pause_cuda_stats["inactive_split_GB"]
                    ),
                    "post_pause_alloc_retries": (
                        None if post_pause_cuda_stats is None else post_pause_cuda_stats["alloc_retries"]
                    ),
                    "post_pause_ooms": None if post_pause_cuda_stats is None else post_pause_cuda_stats["ooms"],
                },
            )

    @timer
    def wake_up(self, reason: str = "unspecified") -> None:
        assert self.args.offload_train
        ptm_debug_enabled = is_ptm_debug_enabled()
        if ptm_debug_enabled:
            self._wake_up_call_idx = getattr(self, "_wake_up_call_idx", 0) + 1
            wake_total_start = perf_counter()
            wake_timings: dict[str, float] = {}
            pre_resume_cuda_stats = print_cuda_memory_stats(f"before {reason} torch_memory_saver.resume")
        else:
            wake_total_start = None
            wake_timings = {}
            pre_resume_cuda_stats = None
        print_memory("before wake_up model")

        resume_start = perf_counter() if ptm_debug_enabled else None
        torch_memory_saver.resume()
        if resume_start is not None:
            elapsed = perf_counter() - resume_start
            Timer().add("wake_up_torch_memory_saver_resume", elapsed)
            wake_timings["wake_up_torch_memory_saver_resume"] = elapsed

        clear_memory_start = perf_counter() if ptm_debug_enabled else None
        clear_memory()
        if clear_memory_start is not None:
            elapsed = perf_counter() - clear_memory_start
            Timer().add("wake_up_clear_memory", elapsed)
            wake_timings["wake_up_clear_memory"] = elapsed

        reload_pg_start = perf_counter() if ptm_debug_enabled else None
        reload_process_groups()
        if reload_pg_start is not None:
            elapsed = perf_counter() - reload_pg_start
            Timer().add("wake_up_reload_process_groups", elapsed)
            wake_timings["wake_up_reload_process_groups"] = elapsed
        print_memory("after wake_up model")
        if ptm_debug_enabled:
            post_resume_cuda_stats = print_cuda_memory_stats(f"after {reason} torch_memory_saver.resume")
        else:
            post_resume_cuda_stats = None
        if wake_total_start is not None:
            self._log_ptm_debug_offload_transition(
                transition="wake_up",
                timings=wake_timings,
                total_time=perf_counter() - wake_total_start,
                reason=reason,
                extra_metrics={
                    "pre_resume_reserved_GB": (
                        None if pre_resume_cuda_stats is None else pre_resume_cuda_stats["reserved_GB"]
                    ),
                    "pre_resume_active_GB": None if pre_resume_cuda_stats is None else pre_resume_cuda_stats["active_GB"],
                    "pre_resume_inactive_split_GB": (
                        None if pre_resume_cuda_stats is None else pre_resume_cuda_stats["inactive_split_GB"]
                    ),
                    "pre_resume_alloc_retries": (
                        None if pre_resume_cuda_stats is None else pre_resume_cuda_stats["alloc_retries"]
                    ),
                    "pre_resume_ooms": None if pre_resume_cuda_stats is None else pre_resume_cuda_stats["ooms"],
                    "post_resume_reserved_GB": (
                        None if post_resume_cuda_stats is None else post_resume_cuda_stats["reserved_GB"]
                    ),
                    "post_resume_active_GB": (
                        None if post_resume_cuda_stats is None else post_resume_cuda_stats["active_GB"]
                    ),
                    "post_resume_inactive_split_GB": (
                        None if post_resume_cuda_stats is None else post_resume_cuda_stats["inactive_split_GB"]
                    ),
                    "post_resume_alloc_retries": (
                        None if post_resume_cuda_stats is None else post_resume_cuda_stats["alloc_retries"]
                    ),
                    "post_resume_ooms": None if post_resume_cuda_stats is None else post_resume_cuda_stats["ooms"],
                },
            )

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: (
                            torch.from_numpy(v.copy()).to(device=torch.cuda.current_device())
                            if isinstance(v, np.ndarray)
                            else v.to(device=torch.cuda.current_device())
                        )
                        for key, v in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        if self.args.qkv_format == "bshd":
            # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
            max_seq_len = max(rollout_data["total_lengths"])

            # pad to reduce memory fragmentation and maybe make the computation faster
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

            rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        log_prob,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
        prefix_tree_context=None,
        prefix_tree_stage: str | None = None,
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                prefix_tree_context=prefix_tree_context,
                prefix_tree_stage=prefix_tree_stage,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        if self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            self.wake_up(reason="before_train")

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(
            self.args,
            self.model,
            rollout_data,
            enable_ptm_aware_dynamic_batching=False,
            timer_prefix="critic_data_iterator",
        )
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps and not self.args.critic_train_only:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)

        prefix_tree_context = None
        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            prefix_tree_context=prefix_tree_context,
            prefix_tree_stage="critic-train",
        )

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for training.
        data_iterator, num_microbatches = get_data_iterator(
            self.args,
            self.model,
            rollout_data,
            enable_ptm_aware_dynamic_batching=False,
            timer_prefix="actor_train_data_iterator",
        )
        # By default, logprob uses the same schedule as training.
        logprob_data_iterator, logprob_num_microbatches = data_iterator, num_microbatches
        if self.args.slime_prefix_single_sample_logprob:
            logprob_data_iterator, logprob_num_microbatches = get_data_iterator(
                self.args,
                self.model,
                rollout_data,
                force_single_sample_microbatch=True,
                timer_prefix="actor_logprob_data_iterator",
            )
            if is_megatron_main_rank():
                logger.info(
                    "slime_prefix_single_sample_logprob enabled: "
                    "using one-sample-per-micro-batch for no-grad logprob forward."
                )
        elif self.args.slime_prefix_tree_merging and self.args.slime_prefix_magi_attention:
            # PTM runtime currently only applies to the no-grad logprob forward path.
            # Keep training on original-length budgeting so PTM-compressed scheduling
            # does not overpack actor_train and trigger CE OOMs.
            logprob_data_iterator, logprob_num_microbatches = get_data_iterator(
                self.args,
                self.model,
                rollout_data,
                enable_ptm_aware_dynamic_batching=True,
                timer_prefix="actor_logprob_data_iterator",
            )
            if is_megatron_main_rank():
                logger.info(
                    "Using PTM-aware dynamic batching for no-grad logprob only; "
                    "actor_train keeps original-length dynamic batching."
                )

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            prefix_tree_context = None
            if self.args.slime_prefix_tree_merging:
                prefix_tree_context = build_prefix_tree_context_from_rollout_data(
                    rollout_data,
                    min_group_size=self.args.slime_prefix_min_group_size,
                    block_size=self.args.slime_prefix_runtime_block_size,
                )
                if prefix_tree_context is not None and is_megatron_main_rank() and is_ptm_debug_enabled():
                    log_prefix_tree_context("train_actor_build", prefix_tree_context)

            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            logprob_data_iterator,
                            logprob_num_microbatches,
                            store_prefix="ref_",
                            prefix_tree_context=prefix_tree_context,
                            prefix_tree_stage="ref-logprobs",
                        )
                    )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            logprob_data_iterator,
                            logprob_num_microbatches,
                            store_prefix="teacher_",
                            prefix_tree_context=prefix_tree_context,
                            prefix_tree_stage="teacher-logprobs",
                        )
                    )

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            logprob_data_iterator,
                            logprob_num_microbatches,
                            store_prefix="",
                            prefix_tree_context=prefix_tree_context,
                            prefix_tree_stage="actor-logprobs",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    prefix_tree_context=prefix_tree_context,
                    prefix_tree_stage="actor-train",
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from slime.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

        if self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
