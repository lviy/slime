SLIME_PTM_E2E_COMPARE_KEYS=log_probs \
SLIME_PTM_HF_CHECKPOINT=/gfs/platform/public/infra/Moonlight-16B-A3B-Instruct \
SLIME_PTM_DTYPE=bf16 \
SLIME_PTM_ATOL=5e-2 \
SLIME_PTM_RTOL=1e-2 \
python3 -m pytest -q -rs tests/test_ptm_forward_only.py::test_ptm_logits_distribution_matches_full_forward