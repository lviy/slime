from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Ensure local repo package is imported instead of an installed site-package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.utils.prefix_tree_merging_utils import (
    build_prefix_group_metadata,
    build_prefix_tree_batch_plan,
    build_prefix_tree_context_from_rollout_data,
    estimate_prefix_tree_merged_token_count,
    get_prefix_tree_runtime_skip_reason,
)


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {dtype_name}")


def _build_ptm_eval_prompts() -> list[str]:
    shared_prefix = (
        "System: You are a rigorous math assistant.\n"
        "Tools: calculator, python.\n"
        "Rules: show concise steps.\n"
        "User: "
    )
    return [
        shared_prefix + "Compute 123 + 456 and give the final integer.",
        shared_prefix + "Compute 123 + 789 and give the final integer.",
        shared_prefix + "Compute 987 - 654 and give the final integer.",
        shared_prefix + "Compute 44 * 12 and give the final integer.",
    ]


def _install_forward_only_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install minimal stubs so megatron_utils.model can be imported in unit tests."""

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    mpu_stub = types.SimpleNamespace(is_pipeline_last_stage=lambda: True)
    core_mod.mpu = mpu_stub

    distributed_mod = types.ModuleType("megatron.core.distributed")
    distributed_mod.DistributedDataParallel = object
    distributed_mod.finalize_model_grads = lambda *args, **kwargs: None

    enums_mod = types.ModuleType("megatron.core.enums")
    enums_mod.ModelType = types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")

    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_mod.GPTModel = object

    optimizer_mod = types.ModuleType("megatron.core.optimizer")
    optimizer_mod.OptimizerConfig = object
    optimizer_mod.get_megatron_optimizer = lambda *args, **kwargs: None

    optimizer_obj_mod = types.ModuleType("megatron.core.optimizer.optimizer")
    optimizer_obj_mod.MegatronOptimizer = object

    optimizer_sched_mod = types.ModuleType("megatron.core.optimizer_param_scheduler")
    optimizer_sched_mod.OptimizerParamScheduler = object

    pp_mod = types.ModuleType("megatron.core.pipeline_parallel")
    pp_mod.get_forward_backward_func = lambda: None

    utils_mod = types.ModuleType("megatron.core.utils")
    utils_mod.get_model_config = lambda _model: SimpleNamespace(timers=None)

    training_mod = types.ModuleType("megatron.training")
    global_vars_mod = types.ModuleType("megatron.training.global_vars")
    global_vars_mod.get_args = lambda: SimpleNamespace()

    training_training_mod = types.ModuleType("megatron.training.training")
    training_training_mod.get_model = lambda *args, **kwargs: []

    checkpointing_mod = types.ModuleType("megatron.training.checkpointing")
    checkpointing_mod.load_checkpoint = lambda *args, **kwargs: (0, 0)
    checkpointing_mod.save_checkpoint = lambda *args, **kwargs: None

    data_mod = types.ModuleType("slime.backends.megatron_utils.data")
    data_mod.DataIterator = object
    data_mod.get_batch = lambda *args, **kwargs: {}

    checkpoint_mod = types.ModuleType("slime.backends.megatron_utils.checkpoint")
    checkpoint_mod.load_checkpoint = lambda *args, **kwargs: (0, 0)
    checkpoint_mod.save_checkpoint = lambda *args, **kwargs: None

    loss_mod = types.ModuleType("slime.backends.megatron_utils.loss")
    loss_mod.loss_function = lambda *args, **kwargs: (torch.tensor(0.0), {})

    model_provider_mod = types.ModuleType("slime.backends.megatron_utils.model_provider")
    model_provider_mod.get_model_provider_func = lambda *args, **kwargs: None
    model_provider_mod.wrap_model_provider_with_freeze = lambda fn, _args: fn

    module_map = {
        "megatron": megatron_mod,
        "megatron.core": core_mod,
        "megatron.core.distributed": distributed_mod,
        "megatron.core.enums": enums_mod,
        "megatron.core.models.gpt": gpt_mod,
        "megatron.core.optimizer": optimizer_mod,
        "megatron.core.optimizer.optimizer": optimizer_obj_mod,
        "megatron.core.optimizer_param_scheduler": optimizer_sched_mod,
        "megatron.core.pipeline_parallel": pp_mod,
        "megatron.core.utils": utils_mod,
        "megatron.training": training_mod,
        "megatron.training.global_vars": global_vars_mod,
        "megatron.training.training": training_training_mod,
        "megatron.training.checkpointing": checkpointing_mod,
        "slime.backends.megatron_utils.data": data_mod,
        "slime.backends.megatron_utils.checkpoint": checkpoint_mod,
        "slime.backends.megatron_utils.loss": loss_mod,
        "slime.backends.megatron_utils.model_provider": model_provider_mod,
    }
    for key, value in module_map.items():
        monkeypatch.setitem(sys.modules, key, value)


class _DummyModel:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.train_calls = 0
        self.forward_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1

    def train(self) -> None:
        self.train_calls += 1

    def __call__(self, **kwargs) -> torch.Tensor:
        self.forward_calls += 1
        tokens = kwargs["input_ids"]
        # Match forward_only expectation: [B, T, V]
        return torch.ones((tokens.size(0), tokens.size(1), 2), dtype=torch.float32)


class _DummyIterator:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> "_DummyIterator":
        self.reset_calls += 1
        return self


@pytest.mark.unit
def test_forward_only_with_ptm_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_forward_only_import_stubs(monkeypatch)
    sys.modules.pop("slime.backends.megatron_utils.model", None)
    model_mod = importlib.import_module("slime.backends.megatron_utils.model")

    dummy_tokens = [[101, 11, 12, 13], [101, 11, 12, 99], [7, 8]]
    dummy_rollout_data = {"tokens": dummy_tokens, "total_lengths": [4, 4, 2]}
    ptm_meta = build_prefix_group_metadata(tokens=dummy_tokens, effective_lengths=dummy_rollout_data["total_lengths"])
    dummy_rollout_data.update(ptm_meta)
    ptm_ctx = build_prefix_tree_context_from_rollout_data(dummy_rollout_data)
    assert ptm_ctx is not None
    assert ptm_ctx.enabled
    assert ptm_meta["ptm_num_mergeable_groups"] == 1
    assert ptm_meta["ptm_num_mergeable_samples"] == 2

    def fake_get_batch(*_args, **_kwargs):
        tokens = torch.tensor([[101, 11, 12, 13, 101, 11, 12, 99]], dtype=torch.long)
        return {
            "unconcat_tokens": [torch.tensor(dummy_tokens[0]), torch.tensor(dummy_tokens[1])],
            "tokens": tokens,
            "packed_seq_params": None,
            "total_lengths": [4, 4],
            "response_lengths": [1, 1],
            "full_loss_masks": torch.ones_like(tokens, dtype=torch.float32),
            "multimodal_train_inputs": None,
            "max_seq_lens": [4, 4],
        }

    def fake_forward_backward_func():
        def _run(
            *,
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            seq_length,
            micro_batch_size,
            forward_only,
        ):
            assert seq_length == 16
            assert micro_batch_size == 1
            assert forward_only is True
            out = []
            for _ in range(num_microbatches):
                output_tensor, collector = forward_step_func(data_iterator[0], model[0], return_schedule_plan=False)
                out.append(collector(output_tensor))
            return out

        return _run

    ptm_log_calls: list[tuple[str, object, dict | None]] = []

    def fake_log_prefix_tree_context(stage, context, extra=None):
        ptm_log_calls.append((stage, context, extra))

    def collect_outputs(
        logits: torch.Tensor,
        *,
        args,
        unconcat_tokens,
        total_lengths,
        response_lengths,
        with_entropy,
        max_seq_lens,
    ):
        assert with_entropy is True
        assert len(unconcat_tokens) == 2
        assert total_lengths == [4, 4]
        assert response_lengths == [1, 1]
        assert max_seq_lens == [4, 4]
        return {"scores": [logits.mean()]}

    monkeypatch.setattr(model_mod, "get_batch", fake_get_batch)
    monkeypatch.setattr(model_mod, "get_forward_backward_func", fake_forward_backward_func)
    monkeypatch.setattr(model_mod, "log_prefix_tree_context", fake_log_prefix_tree_context)
    monkeypatch.setattr(model_mod.mpu, "is_pipeline_last_stage", lambda: True)

    args = SimpleNamespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        use_rollout_entropy=True,
        custom_megatron_before_log_prob_hook_path=None,
        seq_length=16,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
    )
    model = _DummyModel()
    iterator = _DummyIterator()
    res = model_mod.forward_only(
        collect_outputs,
        args,
        [model],
        [iterator],
        [2],
        store_prefix="actor_",
        prefix_tree_context=ptm_ctx,
        prefix_tree_stage="actor-logprobs",
    )

    assert "actor_scores" in res
    assert len(res["actor_scores"]) == 2
    assert iterator.reset_calls == 1
    assert model.eval_calls == 1
    assert model.train_calls == 1
    assert model.forward_calls == 2

    assert len(ptm_log_calls) == 1
    stage, context, extra = ptm_log_calls[0]
    assert stage == "actor-logprobs"
    assert context is ptm_ctx
    assert extra is not None
    assert extra["micro_batch_samples"] == 2
    assert extra["qkv_format"] == "thd"
    assert extra["tokens_shape"] == (1, 8)


@pytest.mark.unit
def test_prefix_tree_batch_plan_merges_shared_prefixes() -> None:
    token_lists = [
        [101, 11, 12, 13],
        [101, 11, 12, 99],
        [7, 8],
    ]

    plan = build_prefix_tree_batch_plan(token_lists)

    assert plan.num_input_tokens == 10
    assert plan.num_merged_tokens == 7
    assert estimate_prefix_tree_merged_token_count(token_lists) == 7
    assert len(plan.unmerge_index) == plan.num_input_tokens
    assert len(plan.merged_tokens) == plan.num_merged_tokens


@pytest.mark.unit
def test_prefix_tree_runtime_skip_reason_single_sample_microbatch() -> None:
    reason = get_prefix_tree_runtime_skip_reason(
        [torch.tensor([101, 11, 12, 13], dtype=torch.long)],
        group_ids=[0],
    )

    assert reason == "single_sample_microbatch"


@pytest.mark.unit
def test_prefix_tree_runtime_skip_reason_no_mergeable_group_overlap() -> None:
    reason = get_prefix_tree_runtime_skip_reason(
        [
            torch.tensor([101, 11, 12, 13], dtype=torch.long),
            torch.tensor([101, 11, 12, 99], dtype=torch.long),
        ],
        group_ids=[0, 1],
    )

    assert reason == "no_mergeable_group_overlap"


@pytest.mark.unit
def test_prefix_tree_runtime_skip_reason_allows_repeated_group() -> None:
    reason = get_prefix_tree_runtime_skip_reason(
        [
            torch.tensor([101, 11, 12, 13], dtype=torch.long),
            torch.tensor([101, 11, 12, 99], dtype=torch.long),
        ],
        group_ids=[0, 0],
    )

    assert reason is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
