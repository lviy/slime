import logging
from time import perf_counter

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action
from slime.utils.prefix_tree_merging_utils import (
    add_ptm_debug_timing,
    format_ptm_debug_timing_metrics,
    is_ptm_debug_enabled,
    start_ptm_debug_timer,
)

logger = logging.getLogger(__name__)


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload(reason="post_train_offload")
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    actor_model.offload(reason="post_train_offload")
            else:
                actor_model.offload(reason="post_train_offload")
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        ptm_debug_enabled = is_ptm_debug_enabled()
        driver_timings: dict[str, float] | None = {} if ptm_debug_enabled else None
        rollout_step_start = perf_counter() if ptm_debug_enabled else None

        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        generate_start = start_ptm_debug_timer()
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        add_ptm_debug_timing(driver_timings, "driver_generate", generate_start)

        if args.offload_rollout:
            rollout_offload_start = start_ptm_debug_timer()
            ray.get(rollout_manager.offload.remote())
            add_ptm_debug_timing(driver_timings, "driver_rollout_offload", rollout_offload_start)

        train_wait_total_start = start_ptm_debug_timer()
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                actor_train_start = start_ptm_debug_timer()
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
                add_ptm_debug_timing(driver_timings, "driver_actor_train_wait", actor_train_start)
            critic_join_start = start_ptm_debug_timer()
            ray.get(critic_train_handle)
            add_ptm_debug_timing(driver_timings, "driver_critic_train_join", critic_join_start)
        else:
            actor_train_start = start_ptm_debug_timer()
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            add_ptm_debug_timing(driver_timings, "driver_actor_train_wait", actor_train_start)
        add_ptm_debug_timing(driver_timings, "driver_train_wait_total", train_wait_total_start)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        train_offload_start = start_ptm_debug_timer()
        offload_train(rollout_id)
        add_ptm_debug_timing(driver_timings, "driver_train_offload_or_clear", train_offload_start)
        if args.offload_rollout:
            onload_weights_start = start_ptm_debug_timer()
            ray.get(rollout_manager.onload_weights.remote())
            add_ptm_debug_timing(driver_timings, "driver_rollout_onload_weights", onload_weights_start)
        if not args.critic_train_only:
            update_weights_start = start_ptm_debug_timer()
            actor_model.update_weights()
            add_ptm_debug_timing(driver_timings, "driver_actor_update_weights", update_weights_start)
        if args.offload_rollout:
            onload_kv_start = start_ptm_debug_timer()
            ray.get(rollout_manager.onload_kv.remote())
            add_ptm_debug_timing(driver_timings, "driver_rollout_onload_kv", onload_kv_start)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

        if ptm_debug_enabled and driver_timings is not None and rollout_step_start is not None:
            total_driver_time = perf_counter() - rollout_step_start
            driver_metrics = format_ptm_debug_timing_metrics(
                driver_timings,
                extra_metrics={
                    "perf/driver_rollout_total_time": total_driver_time,
                    "perf/driver_rollout_non_train_time": total_driver_time
                    - driver_timings.get("driver_train_wait_total", 0.0),
                },
            )
            logger.info(f"[PTMDebug] driver perf {rollout_id}: {driver_metrics}")

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
