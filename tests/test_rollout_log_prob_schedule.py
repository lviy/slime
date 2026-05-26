import pytest

from slime.ray.rollout import RolloutRayActor
from types import SimpleNamespace


@pytest.mark.unit
def test_split_train_data_by_dp_keeps_default_path_when_log_prob_cap_matches_train_cap(monkeypatch):
    actor = RolloutRayActor.__new__(RolloutRayActor)
    actor.args = SimpleNamespace(
        global_batch_size=4,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=16,
        log_probs_max_tokens_per_gpu=16,
    )
    actor.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }

    monkeypatch.setattr("slime.ray.rollout.ray.put", lambda x: x)

    data = {
        "tokens": [[1] * 8 for _ in range(4)],
        "response_lengths": [4] * 4,
        "rewards": [1.0] * 4,
        "truncated": [0] * 4,
        "loss_masks": [[1] * 4 for _ in range(4)],
        "sample_indices": list(range(4)),
        "rollout_ids": list(range(4)),
        "rollout_mask_sums": [4] * 4,
    }

    refs = actor._split_train_data_by_dp(data)
    rank0 = refs[0].inner

    assert "log_prob_micro_batch_indices" not in rank0
    assert "log_prob_num_microbatches" not in rank0


@pytest.mark.unit
def test_split_train_data_by_dp_adds_log_prob_schedule_when_explicit_override_is_set(monkeypatch):
    actor = RolloutRayActor.__new__(RolloutRayActor)
    actor.args = SimpleNamespace(
        global_batch_size=4,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=16,
        log_probs_max_tokens_per_gpu=8,
    )
    actor.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }

    monkeypatch.setattr("slime.ray.rollout.ray.put", lambda x: x)

    data = {
        "tokens": [[1] * 8 for _ in range(4)],
        "response_lengths": [4] * 4,
        "rewards": [1.0] * 4,
        "truncated": [0] * 4,
        "loss_masks": [[1] * 4 for _ in range(4)],
        "sample_indices": list(range(4)),
        "rollout_ids": list(range(4)),
        "rollout_mask_sums": [4] * 4,
    }

    refs = actor._split_train_data_by_dp(data)
    rank0 = refs[0].inner

    assert "log_prob_micro_batch_indices" in rank0
    assert "log_prob_num_microbatches" in rank0
    assert rank0["log_prob_num_microbatches"] != rank0["num_microbatches"]
