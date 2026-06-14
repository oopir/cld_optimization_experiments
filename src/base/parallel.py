import multiprocessing as mp
from typing import List, Mapping, Optional, Sequence

import torch


def maybe_configure_spawn_for_cuda(device: str) -> None:
    """Use spawn for CUDA worker processes, matching PyTorch multiprocessing guidance."""
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass


def resolve_worker_gpu_ids(base_device: str, gpu_ids: Optional[Sequence[int]] = None) -> List[Optional[int]]:
    """
    Resolve worker GPU IDs for simple round-robin process pools.

    CPU workers are represented by `[None]`; CUDA workers are integer device IDs.
    """
    if gpu_ids is not None:
        resolved = [int(idx) for idx in gpu_ids]
        maybe_configure_spawn_for_cuda(base_device)
        return resolved

    if isinstance(base_device, str) and base_device.startswith("cuda") and torch.cuda.is_available():
        maybe_configure_spawn_for_cuda(base_device)
        if ":" in base_device:
            return [int(base_device.split(":", 1)[1])]
        num_gpus = torch.cuda.device_count()
        return list(range(num_gpus)) if num_gpus > 0 else [0]

    return [None]


def worker_device(base_device: str, gpu_ids: Sequence[Optional[int]], worker_index: int) -> str:
    """Return the device string for a worker index under round-robin GPU assignment."""
    if not gpu_ids or gpu_ids[0] is None:
        return base_device
    return f"cuda:{gpu_ids[worker_index % len(gpu_ids)]}"


def round_robin_device_names(
    per_device_counts: Mapping[int, int],
    ordered_devices: Optional[Sequence[int]] = None,
) -> List[str]:
    """Spread CUDA device names before adding second workers on the same GPU."""
    ordered = list(ordered_devices) if ordered_devices is not None else sorted(per_device_counts)
    max_slots = max((int(per_device_counts.get(idx, 0)) for idx in ordered), default=0)
    devices = []
    for slot_index in range(max_slots):
        for idx in ordered:
            if int(per_device_counts.get(idx, 0)) > slot_index:
                devices.append(f"cuda:{idx}")
    return devices
