import os

import slime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))
TIGHT_HOST_MEMORY = bool(int(os.environ.get("SLIME_TEST_TIGHT_HOST_MEMORY", "1")))

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 8
TRAIN_MAX_TOKENS_PER_GPU = 8192
LOGPROB_MAX_TOKENS_PER_GPU = TRAIN_MAX_TOKENS_PER_GPU * 3


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")

    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def _build_train_args(*, calculate_per_token_loss: bool) -> str:
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 2 "
        "--rollout-num-gpus 8 "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-cuda-graph-max-bs 32 "
    )

    ci_args = "--ci-test "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--colocate "
    )

    per_token_loss_args = "--calculate-per-token-loss " if calculate_per_token_loss else ""

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{per_token_loss_args}"
    )


def _run_parallel_load_sweep(
    train_args: str,
    *,
    data_path: str,
    grad_norm_path: str,
    logprob_max_tokens_per_gpu: int | None,
):
    logprob_arg = (
        f"--log-probs-max-tokens-per-gpu {logprob_max_tokens_per_gpu} " if logprob_max_tokens_per_gpu is not None else ""
    )

    for num_gpus in [8, 4, 2]:
        remaining_gpus = num_gpus
        for tp_size in [1, 2, 4, 8]:
            remaining_gpus /= tp_size
            for pp_size in [1, 2, 4]:
                if remaining_gpus < pp_size:
                    continue
                remaining_gpus /= pp_size
                for cp_size in [1, 2, 4, 8]:
                    if remaining_gpus < cp_size:
                        continue
                    args = train_args + (
                        f"--load-debug-rollout-data {data_path} "
                        f"--ci-load-grad-norm {grad_norm_path} "
                        f"--context-parallel-size {cp_size} "
                        f"--tensor-model-parallel-size {tp_size} "
                        f"--pipeline-model-parallel-size {pp_size} "
                        "--sequence-parallel "
                        f"--actor-num-gpus-per-node {num_gpus} "
                        "--use-dynamic-batch-size "
                        f"--max-tokens-per-gpu {TRAIN_MAX_TOKENS_PER_GPU} "
                        f"{logprob_arg}"
                    )

                    U.execute_train(
                        train_args=args,
                        num_gpus_per_node=num_gpus,
                        megatron_model_type=MODEL_TYPE,
                    )


def execute():
    for pass_id, calculate_per_token_loss in enumerate([False, True]):
        train_args = _build_train_args(calculate_per_token_loss=calculate_per_token_loss)

        U.execute_train(
            train_args=train_args
            + (
                f"--save-debug-rollout-data data-{pass_id}.pt "
                f"--ci-save-grad-norm grad_norms-{pass_id}.pt "
                f"--actor-num-gpus-per-node {NUM_GPUS} "
            ),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )

        _run_parallel_load_sweep(
            train_args,
            data_path=f"data-{pass_id}.pt",
            grad_norm_path=f"grad_norms-{pass_id}.pt",
            logprob_max_tokens_per_gpu=None,
        )
        _run_parallel_load_sweep(
            train_args,
            data_path=f"data-{pass_id}.pt",
            grad_norm_path=f"grad_norms-{pass_id}.pt",
            logprob_max_tokens_per_gpu=LOGPROB_MAX_TOKENS_PER_GPU,
        )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
