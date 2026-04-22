import gc
import logging

import psutil
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    vm = psutil.virtual_memory()
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
        "host_total_GB": _byte_to_gb(vm.total),
        "host_available_GB": _byte_to_gb(vm.available),
        "host_used_GB": _byte_to_gb(vm.used),
        "host_free_GB": _byte_to_gb(vm.free),
    }


def cuda_memory_stats_summary() -> dict[str, float | int | str]:
    device = torch.cuda.current_device()
    stats = torch.cuda.memory_stats(device)
    return {
        "gpu": str(device),
        "reserved_GB": _byte_to_gb(int(stats.get("reserved_bytes.all.current", 0))),
        "active_GB": _byte_to_gb(int(stats.get("active_bytes.all.current", 0))),
        "inactive_split_GB": _byte_to_gb(int(stats.get("inactive_split_bytes.all.current", 0))),
        "alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "ooms": int(stats.get("num_ooms", 0)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info


def print_cuda_memory_stats(msg: str):
    stats = cuda_memory_stats_summary()
    logger.info(f"[Rank {dist.get_rank()}] CUDA memory stats {msg}: {stats}")
    return stats
