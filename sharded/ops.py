from __future__ import annotations

import torch
import torch.distributed as dist

from .distributed import distributed_is_active


class _AllGatherConcat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local: torch.Tensor, dim: int):
        ctx.dim = dim
        ctx.local_size = local.shape[dim]
        if not distributed_is_active():
            ctx.world_size = 1
            ctx.rank = 0
            return local

        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        chunks = [torch.empty_like(local) for _ in range(ctx.world_size)]
        dist.all_gather(chunks, local.contiguous())
        return torch.cat(chunks, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size == 1:
            return grad_output, None

        chunks = [
            chunk.contiguous()
            for chunk in torch.split(grad_output, ctx.local_size, dim=ctx.dim)
        ]
        if len(chunks) != ctx.world_size:
            raise RuntimeError(
                "AllGatherConcat backward expected one gradient chunk per rank; "
                f"got {len(chunks)} chunk(s) for world_size={ctx.world_size}."
            )

        stacked = torch.stack(chunks, dim=0)
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        grad_local = stacked[ctx.rank].contiguous()
        return grad_local, None


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local: torch.Tensor):
        if not distributed_is_active():
            ctx.world_size = 1
            return local

        ctx.world_size = dist.get_world_size()
        out = local.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size == 1:
            return grad_output

        grad_local = grad_output.contiguous().clone()
        dist.all_reduce(grad_local, op=dist.ReduceOp.SUM)
        return grad_local


def all_gather_cat(local: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return _AllGatherConcat.apply(local, dim)


def all_reduce_sum_autograd(local: torch.Tensor) -> torch.Tensor:
    return _AllReduceSum.apply(local)
