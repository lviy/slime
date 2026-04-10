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

from slime.utils.prefix_tree_merging_utils import build_prefix_group_metadata, build_prefix_tree_context_from_rollout_data


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


@pytest.mark.integration
def test_ptm_logits_distribution_matches_full_forward() -> None:
    """Accuracy test: compare logits from PTM-like cached forward vs full forward.

    This test intentionally runs with a real HF checkpoint instead of Megatron
    distributed runtime. It validates the key PTM invariant: reusing prefix KV
    and forwarding only suffix should preserve logits distribution.
    """

    transformers = pytest.importorskip("transformers")
    hf_ckpt = os.getenv("SLIME_PTM_HF_CHECKPOINT")
    if not hf_ckpt:
        pytest.skip("Set SLIME_PTM_HF_CHECKPOINT to run PTM logits accuracy integration test.")
    if not torch.cuda.is_available():
        pytest.skip("PTM logits integration test expects CUDA.")

    dtype = _resolve_torch_dtype(os.getenv("SLIME_PTM_DTYPE", "bf16"))
    atol = float(os.getenv("SLIME_PTM_ATOL", "5e-2"))
    rtol = float(os.getenv("SLIME_PTM_RTOL", "1e-2"))
    min_group_size = int(os.getenv("SLIME_PTM_MIN_GROUP_SIZE", "2"))
    prefix_max_len_env = os.getenv("SLIME_PTM_PREFIX_MAX_LEN")
    prefix_max_len = int(prefix_max_len_env) if prefix_max_len_env else None

    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            hf_ckpt,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        .cuda()
        .eval()
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_ckpt, trust_remote_code=True)

    prompts = _build_ptm_eval_prompts()
    token_tensors = [tokenizer(p, return_tensors="pt").input_ids[0].to(torch.long) for p in prompts]
    token_lists = [t.tolist() for t in token_tensors]
    total_lengths = [len(t) for t in token_lists]

    rollout_data = {"tokens": token_lists, "total_lengths": total_lengths}
    ptm_meta = build_prefix_group_metadata(
        tokens=token_lists,
        effective_lengths=total_lengths,
        prefix_max_len=prefix_max_len,
        min_group_size=min_group_size,
    )
    rollout_data.update(ptm_meta)
    ptm_ctx = build_prefix_tree_context_from_rollout_data(rollout_data, min_group_size=min_group_size)
    assert ptm_ctx is not None
    assert ptm_ctx.num_samples == len(prompts)
    assert ptm_ctx.num_mergeable_groups > 0, "Prompts should share reusable prefix for this integration test."

    max_abs_diffs: list[float] = []
    mean_abs_diffs: list[float] = []

    with torch.no_grad():
        for sample_idx, input_ids_cpu in enumerate(token_tensors):
            input_ids = input_ids_cpu.unsqueeze(0).cuda()
            full_logits = model(input_ids=input_ids, use_cache=False).logits.squeeze(0).float().cpu()

            prefix_len = ptm_meta["ptm_prefix_lens"][sample_idx]
            seq_len = total_lengths[sample_idx]
            if prefix_len <= 0 or prefix_len >= seq_len:
                # No reusable prefix for this sample; skip equivalence check.
                continue

            prefix_ids = input_ids[:, :prefix_len]
            suffix_ids = input_ids[:, prefix_len:seq_len]

            prefill = model(input_ids=prefix_ids, use_cache=True)
            suffix_logits = (
                model(
                    input_ids=suffix_ids,
                    past_key_values=prefill.past_key_values,
                    use_cache=False,
                )
                .logits.squeeze(0)
                .float()
                .cpu()
            )
            ref_suffix_logits = full_logits[prefix_len:seq_len]

            assert suffix_logits.shape == ref_suffix_logits.shape
            abs_diff = (suffix_logits - ref_suffix_logits).abs()
            max_abs_diffs.append(abs_diff.max().item())
            mean_abs_diffs.append(abs_diff.mean().item())
            assert torch.allclose(suffix_logits, ref_suffix_logits, rtol=rtol, atol=atol)

    assert max_abs_diffs, "No sample had reusable prefix; cannot validate PTM logits equivalence."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
