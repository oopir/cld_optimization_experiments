from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str
    gpu_indices: list[int] | None = None

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def init_distributed(device_preference: str = "cuda", gpu_indices: list[int] | None = None) -> DistContext:
    """Initialize torch.distributed when launched by torchrun."""
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    wants_cuda = str(device_preference).startswith("cuda")
    use_cuda = wants_cuda and torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if use_cuda:
        visible_device_count = torch.cuda.device_count()
        if gpu_indices is None:
            cuda_index = local_rank
        else:
            if len(gpu_indices) != env_world_size:
                raise ValueError(
                    f"gpu_indices length ({len(gpu_indices)}) must equal "
                    f"torchrun WORLD_SIZE ({env_world_size})."
                )
            if local_rank >= len(gpu_indices):
                raise ValueError(
                    f"LOCAL_RANK={local_rank} has no matching gpu_indices entry "
                    f"(gpu_indices={gpu_indices})."
                )
            cuda_index = int(gpu_indices[local_rank])

        if cuda_index < 0 or cuda_index >= visible_device_count:
            raise ValueError(
                f"CUDA device index {cuda_index} is unavailable; "
                f"torch sees {visible_device_count} visible CUDA device(s)."
            )

        device = torch.device(f"cuda:{cuda_index}")
        torch.cuda.set_device(device)
    else:
        if gpu_indices is not None:
            raise ValueError("gpu_indices can only be set when device starts with 'cuda'.")
        device = torch.device("cpu")

    if launched and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    return DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
        gpu_indices=gpu_indices,
    )


def distributed_is_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if distributed_is_active():
        if not tensor.is_contiguous():
            reduced = tensor.contiguous()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            tensor.copy_(reduced)
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.clone()
    return all_reduce_sum_(out)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(ctx: DistContext, *args, **kwargs) -> None:
    if ctx.is_rank0:
        print(*args, **kwargs)
