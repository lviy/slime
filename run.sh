SLIME_PTM_E2E_MODEL_PATH=/gfs/platform/public/infra/lxr/models/Qwen2.5-0.5B-Instruct \
SLIME_PTM_E2E_SGLANG_ROOT=/gfs/platform/public/infra/lxr/sglang \
SLIME_PTM_E2E_MEGATRON_PATH=/gfs/platform/public/infra/lxr/Megatron-LM \
SLIME_PTM_E2E_NUM_GPUS=2 \
./scripts/run_ptm_e2e_accuracy.sh


bash scripts/run_ptm_forward_only.sh \
    --rollout-pt /gfs/platform/public/infra/lxr/dumpdata/rollout_0.pt \
    --save-dir /tmp/ptm_forward_only \
    --skip-prepare

##可以直接按magi?
python3 -m pip install --no-deps /gfs/platform/public/infra/lxr/wheels/magi_attention-1.0.5-cp312-cp312-linux_x86_64.whl
# Moonlight
SLIME_PTM_E2E_MEGATRON_PATH=/gfs/platform/public/infra/lxr/Megatron-LM \
SLIME_PTM_E2E_PYTHONPATH=/gfs/platform/public/infra/lxr/sglang/python:/gfs/platform/public/infra/lxr/Megatron-LM \
SLIME_PTM_E2E_LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
bash scripts/run_ptm_forward_speed.sh \
    --model qwen3-4B \
    --model-name Moonlight-16B-A3B-Instruct \
    --model-type moonlight \
    --model-path /gfs/platform/public/infra/Moonlight-16B-A3B-Instruct \
    --ref-load /gfs/platform/public/infra/Moonlight-16B-A3B-Instruct_torch_dist \
    --megatron-to-hf-mode raw \
    --rollout-pt /tmp/moonlight16b_a3b_ptm_rollout/rollout_0.pt \
    --num-gpus 8 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 4 \
    --expert-tensor-parallel-size 1 \
    --max-tokens-per-gpu 10240 \
    --ptm-mode both \
    --save-dir /tmp/moonlight16b_speed

# qwen 8b
SLIME_PTM_E2E_MEGATRON_PATH=/gfs/platform/public/infra/lxr/Megatron-LM \
SLIME_PTM_E2E_PYTHONPATH=/gfs/platform/public/infra/lxr/sglang/python:/gfs/platform/public/infra/lxr/Megatron-LM \
SLIME_PTM_E2E_LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
 bash scripts/run_ptm_forward_speed.sh \
    --model qwen3-8B \
    --model-path /gfs/platform/public/infra/Qwen3-8B \
    --megatron-to-hf-mode bridge \
    --rollout-pt /tmp/qwen3_8b_ptm_rollout/rollout_0.pt \
    --num-gpus 4 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --context-parallel-size 1 \
    --warmup-runs 1 \
    --measure-runs 2 \
    --max-tokens-per-gpu 10240 \
    --ptm-mode both \
    --save-dir /tmp/qwen3_8b_speed \
    --skip-prepare > /gfs/platform/public/infra/lxr/logs/qwen3_8b_ptm_forward_speed.log 2>&1