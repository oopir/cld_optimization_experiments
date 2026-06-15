from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
import math
import multiprocessing as mp
import subprocess
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch

from ..base.parallel import round_robin_device_names
from ..base.data import DATASET_METADATA, load_binary_classification_data
from .core import (
    InitScaleProbeConfig,
    _rows_for_trained_initialization,
    sort_probe_rows,
    summarize_init_averaged_data_variability_rows,
    summarize_data_averaged_init_variability_rows,
    summarize_rows,
    write_csv,
)
from .plotting import plot_probe_summaries

MB = 1024 * 1024


@dataclass(frozen=True)
class WorkItem:
    """One scheduled unit: one data/model shape and one chunk of init seeds."""
    n: int
    data_seed: int
    synthetic_anisotropy_power: float
    m: int
    alpha: float
    beta: float
    init_seeds: Tuple[int, ...]
    batch_size: int
    jacobian_batch_size: int

    @property
    def profile_fields(self) -> Tuple[int, int]:
        """Return the key used to share calibration data across alpha/beta/init chunks."""
        return (self.n, self.m)


@dataclass(frozen=True)
class DeviceSlot:
    """A logical worker slot bound to a CPU or CUDA device string."""
    device: str

# -------------------------------------------------------------------------- #
# ------------------------------ public entry ------------------------------ #
# -------------------------------------------------------------------------- #

def run_probe_parallel(config: InitScaleProbeConfig):
    """
    Main function.
    The parent process builds work items, chooses device slots, aggregates
    rows, and writes outputs. Dataset loading and metric computation happen
    inside worker processes.
    """
    # Orchestrate and run workers
    device = _resolve_base_device(config.device)
    items = _build_work_items(config)
    if not items:
        rows = []
    elif device == "cpu":
        slots = _cpu_slots(config, items)
        rows = _run_items(config, items, slots)
    else:
        items = _profile_items(config, items, device)
        slots = _cuda_slots(config, items, device)
        _print_slots("CUDA worker slots", slots)
        rows = _run_items(config, items, slots)

    # Aggregate in a deterministic order so parallel and serial CSVs are comparable.
    rows = sort_probe_rows(rows)
    summary_rows = summarize_rows(rows, config.tracked_metrics or [], report_data_seed=config.report_data_seed)
    data_averaged_init_variability_rows = summarize_data_averaged_init_variability_rows(rows, config.tracked_metrics or [])
    init_averaged_data_variability_rows = summarize_init_averaged_data_variability_rows(rows, config.tracked_metrics or [])

    # File writing and plotting
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": config.output_dir / "_init_scale_rows.csv",
        "summary": config.output_dir / "_init_scale_summary.csv",
    }
    write_csv(paths["rows"], rows)
    write_csv(paths["summary"], summary_rows)
    paths.update(
        plot_probe_summaries(
            summary_rows,
            config,
            config.output_dir,
            data_averaged_init_variability_rows=data_averaged_init_variability_rows,
            init_averaged_data_variability_rows=init_averaged_data_variability_rows,
        )
    )
    return rows, summary_rows, paths

# -------------------------------------------------------------------------- #
# -------------------------- work item construction ------------------------ #
# -------------------------------------------------------------------------- #

def _chunks(values: Sequence[int], chunk_size: int) -> Iterable[Sequence[int]]:
    """Yield consecutive chunks of a seed list without shuffling seed order."""
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]

def _build_work_items(config: InitScaleProbeConfig) -> List[WorkItem]:
    """
    Create init-seed chunks for every point in the configured sweep grid.
    A WorkItem keeps `(n, data_seed, anisotropy, m, alpha, beta)` and groups several init seeds.
    """
    items = []
    init_seeds = list(config.init_seeds or [])
    for n in config.n_values:
        for data_seed in config.data_seeds:
            for anisotropy_power in config.synthetic_anisotropy_powers or [config.synthetic_anisotropy_power]:
                for m in config.m_values:
                    for alpha in config.alpha_values:
                        for beta in config.beta_values:
                            for seed_chunk in _chunks(init_seeds, config.init_chunk_size):
                                items.append(
                                    WorkItem(
                                        n=n,
                                        data_seed=data_seed,
                                        synthetic_anisotropy_power=float(anisotropy_power),
                                        m=m,
                                        alpha=alpha,
                                        beta=beta,
                                        init_seeds=tuple(seed_chunk),
                                        batch_size=config.batch_size,
                                        jacobian_batch_size=config.jacobian_batch_size,
                                    )
                                )
    return items

# -------------------------------------------------------------------------- #
# ----------------------------- device slots ------------------------------- #
# -------------------------------------------------------------------------- #

def _resolve_base_device(device: str) -> str:
    "Decide whether to use the CPU path or CUDA path."
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device!r}, but CUDA is unavailable.")
    return "cuda" if device.startswith("cuda") else "cpu"

def _cpu_slots(config: InitScaleProbeConfig, items: Sequence[WorkItem]) -> List[DeviceSlot]:
    # Reuse max_workers_per_gpu as the CPU worker cap; there is no CPU-specific knob yet.
    count = min(len(items), max(1, config.max_workers_per_gpu))
    return [DeviceSlot("cpu") for _ in range(max(1, count))]

def _cuda_slots(config: InitScaleProbeConfig, items: Sequence[WorkItem], device: str) -> List[DeviceSlot]:
    """
    Create CUDA worker slots from selected GPUs and estimated memory budget.
    With adaptive packing, each GPU gets roughly
    `usable_free_memory / largest_estimated_worker_memory` slots, capped by
    `max_workers_per_gpu`. The estimate is deliberately conservative; it is a
    packing heuristic, not an exact CUDA allocator prediction.
    """
    gpu_ids = _gpu_ids(config, device)
    if not gpu_ids:
        raise RuntimeError("No CUDA GPUs were selected for the parallel probe.")

    if not config.adaptive_gpu_packing:
        per_gpu = {idx: config.max_workers_per_gpu for idx in gpu_ids}
    else:
        peak_mb = max(_static_memory_estimate_mb(config, item) for item in items)
        per_gpu = {}
        for idx in gpu_ids:
            budget = _gpu_memory_budget_mb(config, idx)
            candidate_num_workers = int(budget // max(peak_mb, 1.0))
            per_gpu[idx] = max(1, min(config.max_workers_per_gpu, candidate_num_workers))

    slots = _round_robin_cuda_slots(per_gpu, gpu_ids=gpu_ids)
    return slots or [DeviceSlot(f"cuda:{gpu_ids[0]}")]

def _round_robin_cuda_slots(
    per_gpu: Mapping[int, int],
    gpu_ids: Optional[Sequence[int]] = None,
) -> List[DeviceSlot]:
    """
    Build CUDA slots by spreading across GPUs before adding second workers.

    For example, counts `{0: 2, 1: 2, 2: 1}` produce:
    `cuda:0, cuda:1, cuda:2, cuda:0, cuda:1`.
    """
    return [
        DeviceSlot(device)
        for device in round_robin_device_names(per_gpu, ordered_devices=gpu_ids)
    ]

def _gpu_ids(config: InitScaleProbeConfig, device: str) -> List[int]:
    if config.gpu_ids is not None:
        return list(config.gpu_ids)
    if ":" in config.device:
        return [int(config.device.split(":", 1)[1])]
    return list(range(torch.cuda.device_count()))

# -------------------------------------------------------------------------- #
# -------------------------- memory and profiling -------------------------- #
# -------------------------------------------------------------------------- #

def _profile_items(config: InitScaleProbeConfig, items: Sequence[WorkItem], device: str) -> List[WorkItem]:
    "Calibrate one representative per shape to see which batch sizes actually run ok."
    if not config.adaptive_gpu_packing or not items:
        return list(items)

    start_time = time.monotonic()
    print(f"Profiling starts...")
    representatives: Dict[Tuple[int, int], WorkItem] = {}
    for item in items:
        if item.profile_fields not in representatives:
            representatives[item.profile_fields] = replace(item, init_seeds=(item.init_seeds[0],))

    profile_items = list(representatives.values())
    profile_slots = _cuda_slots(config, profile_items, device)

    profile_step_values = [0] if max(config.training_step_values) <= 0 else [0, 1]
    profile_config = replace(config, training_step_values=profile_step_values)
    profile_batches: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for item, result in _run_profile_items(profile_config, profile_items, profile_slots):
        # `_run_item_with_retries` returns a result dict containing `ok`, `rows`,
        # `peak_mb`, `batch_size`, and `jacobian_batch_size`. For ok=True, the
        # returned sizes are the quantities that actually worked.
        if not result["ok"]:
            raise RuntimeError(f"GPU calibration failed: {result['error']}")
        profile_batches[item.profile_fields] = (result["batch_size"], result["jacobian_batch_size"])

    print(f"Profiling ends after {_format_duration(time.monotonic() - start_time)}")
    return [
        replace(
            item,
            batch_size=profile_batches[item.profile_fields][0],
            jacobian_batch_size=profile_batches[item.profile_fields][1],
        )
        for item in items
    ]

def _gpu_memory_budget_mb(config: InitScaleProbeConfig, gpu_idx: int) -> float:
    "Estimate usable free GPU memory after safety and reservation margins."
    free_mb = None

    # try obtaining the amount of free memory by running nvidia-smi
    out = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    if out is not None:
        for line in out.strip().splitlines():
            idx_str, free_str = [part.strip() for part in line.split(",", 1)]
            if int(idx_str) == gpu_idx:
                free_mb = float(free_str)
                break

    # if using nvidia-smi failed, try obtaining the info via cuda
    if free_mb is None:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
            free_mb = free_bytes / MB
        except Exception:
            free_mb = 0.0

    return max(0.0, free_mb * config.gpu_memory_safety_fraction - config.gpu_reserved_memory_mb)

def _print_slots(label: str, slots: Sequence[DeviceSlot]) -> None:
    """Print a compact diagnostic showing effective spread-first slot order."""
    slot_names = [slot.device for slot in slots]
    counts = Counter(slot_names)
    count_parts = [
        f"{name}: {counts[name]}"
        for name in sorted(counts, key=_device_sort_key)
    ]
    print(
        f"{label}: total={len(slot_names)}, "
        f"counts={{{', '.join(count_parts)}}}, "
        f"order_preview={_preview_sequence(slot_names)}",
        flush=True,
    )

def _static_memory_estimate_mb(config: InitScaleProbeConfig, item: WorkItem) -> float:
    """
    Conservatively approximate one worker process's peak tensor memory.
    This estimates the live tensors for one work item on one device: the binary
    training data, the scalar-output two-layer model, the empirical-gradient
    accumulator, and the largest batch's temporary activation tensors.
    It does not estimate Python object overhead in detail or PyTorch/CUDA
    context and allocator overhead; the fixed 512 MiB term is a coarse buffer
    for those effects.
    """
    if config.dataset in {"synthetic_isotropic", "synthetic_anisotropic"}:
        d = int(config.synthetic_d_in)
    else:
        d = int(DATASET_METADATA[config.dataset]["d_in"])
    dtype_bytes = 4
    batch = max(item.batch_size, item.jacobian_batch_size)

    data_bytes = item.n * d * dtype_bytes
    model_bytes = item.m * (d + 1) * dtype_bytes
    act_bytes = batch * item.m * dtype_bytes

    # Conservative counts for live tensors with model-size and activation-size order.
    return (data_bytes + 3 * model_bytes + 8 * act_bytes) / MB + 512.0

# -------------------------------------------------------------------------- #
# -------------------------- scheduling & tracking ------------------------- #
# -------------------------------------------------------------------------- #

def _run_items(
    config: InitScaleProbeConfig,
    items: Sequence[WorkItem],
    slots: Sequence[DeviceSlot],
) -> List[Dict[str, Any]]:
    """Run work items over slots and collect metric rows."""
    rows = []
    for item, result in _run_scheduled_items(config, items, slots, phase="work"):
        rows.extend(_checked_worker_result(result, item))
    return rows

def _run_profile_items(
    config: InitScaleProbeConfig,
    items: Sequence[WorkItem],
    slots: Sequence[DeviceSlot],
) -> List[Tuple[WorkItem, Mapping[str, Any]]]:
    """
    Run profiling items over fixed device slots and return each raw result.

    Unlike `_run_items`, this keeps the worker metadata (`batch_size` and
    `jacobian_batch_size`) instead of returning only metric rows.
    """
    return list(_run_scheduled_items(config, items, slots, phase="profiling"))

def _run_scheduled_items(
    config: InitScaleProbeConfig,
    items: Sequence[WorkItem],
    slots: Sequence[DeviceSlot],
    phase: str,
) -> Iterator[Tuple[WorkItem, Mapping[str, Any]]]:
    """
    Run items over fixed device slots and yield completed worker results.

    Each slot holds at most one running item. When that item completes, the
    scheduler yields its result and then refills the freed slot if more work is
    pending. Keeping this as a generator lets callers fail fast on bad worker
    results without waiting for the whole queue.
    """
    if not items:
        return

    log_phase = phase != "profiling"
    progress = None
    if log_phase and config.progress_detail != "grid":
        progress = _ProgressPrinter(phase, total=len(items), interval_seconds=config.progress_interval_seconds)
    tracker = _RunTracker(
        phase,
        items,
        enabled=log_phase and config.progress_detail == "grid",
        interval_seconds=config.progress_interval_seconds,
    )
    if len(slots) == 1:
        if progress is not None:
            progress.print(0, force=True)
        for completed, item in enumerate(items, start=1):
            device = slots[0].device
            tracker.on_submit(item, device)
            result = _run_item_with_retries(config, item, device)
            tracker.on_done(item, completed=completed)
            if progress is not None:
                progress.print(completed, force=completed == len(items))
            yield item, result
        return

    if progress is not None:
        progress.print(0, force=True)
    ctx = mp.get_context("spawn")
    pending = list(items)
    futures = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=len(slots), mp_context=ctx) as pool:
        for slot in slots:
            if pending:
                _submit_to_pool(pool, futures, config, pending.pop(0), slot, tracker)

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                item, slot = futures.pop(fut)
                result = fut.result()
                completed += 1
                tracker.on_done(item, completed=completed)
                if progress is not None:
                    progress.print(completed, force=completed == len(items))
                yield item, result
                if pending:
                    _submit_to_pool(pool, futures, config, pending.pop(0), slot, tracker)

def _submit_to_pool(
    pool: ProcessPoolExecutor,
    futures: Dict[Any, Tuple[WorkItem, DeviceSlot]],
    config: InitScaleProbeConfig,
    item: WorkItem,
    slot: DeviceSlot,
    tracker: "_RunTracker",
) -> None:
    """Submit one item to one slot and register it with the progress tracker."""
    tracker.on_submit(item, slot.device)
    futures[pool.submit(_run_item_with_retries, config, item, slot.device)] = (item, slot)


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS for stable long-run logs."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _preview_sequence(values: Sequence[str], limit: int = 16) -> str:
    """Return a compact preview of a potentially long sequence."""
    if len(values) <= limit:
        return "[" + ", ".join(values) + "]"
    preview = ", ".join(values[:limit])
    return f"[{preview}, ...]"


def _device_sort_key(device: str) -> Tuple[str, int, str]:
    """Sort cuda devices numerically and other device labels lexically."""
    if device.startswith("cuda:"):
        try:
            return ("cuda", int(device.split(":", 1)[1]), device)
        except ValueError:
            pass
    return (device, -1, device)


class _RunTracker:
    """Print tuple-level starts/completions and optional active-slot snapshots."""

    def __init__(
        self,
        phase: str,
        items: Sequence[WorkItem],
        enabled: bool,
        interval_seconds: Optional[float] = None,
    ):
        self.phase = phase
        self.enabled = bool(enabled)
        self.interval_seconds = None if interval_seconds is None else float(interval_seconds)
        self.total_by_grid = Counter(_grid_key(item) for item in items)
        self.total_chunks = sum(self.total_by_grid.values())
        self.num_grids = len(self.total_by_grid)
        self.grid_index = {
            key: idx
            for idx, key in enumerate(sorted(self.total_by_grid, key=_sort_grid_key), start=1)
        }
        self.done_by_grid = Counter()
        self.grid_started_at = {}
        self.active = {}
        self.start_time = time.monotonic()
        self.last_snapshot_time = self.start_time

    def on_submit(self, item: WorkItem, device: str) -> None:
        """Record a submitted item and print a grid-start line if this grid is new."""
        if not self.enabled:
            return
        now = time.monotonic()
        key = _grid_key(item)
        self.active[_item_key(item)] = (item, device, now)
        if key not in self.grid_started_at:
            self.grid_started_at[key] = now
            print(
                f"{self.phase} tuple start: {_format_grid_key(key)} "
                f"tuple={self.grid_index[key]:>3}/{self.num_grids:<3} "
                f"chunks={self.total_by_grid[key]:<2}",
                flush=True,
            )
        self.maybe_snapshot(now)

    def on_done(self, item: WorkItem, completed: int) -> None:
        """Record completion and print a grid-done line when a grid finishes."""
        if not self.enabled:
            return
        now = time.monotonic()
        key = _grid_key(item)
        self.done_by_grid[key] += 1
        _, _, submitted_at = self.active.pop(_item_key(item), (item, "", now))
        grid_done = self.done_by_grid[key]
        grid_total = self.total_by_grid[key]
        if grid_done == grid_total:
            print(
                f"{self.phase} tuple done:  {_format_grid_key(key)} "
                f"tuple={self.grid_index[key]:>3}/{self.num_grids:<3} "
                f"chunks={grid_done:>2}/{grid_total:<2} total_done={completed}/{self.total_chunks} "
                f"tuple_elapsed={_format_duration(now - self.grid_started_at.get(key, now))} ",
                # f"last_chunk={_format_duration(now - submitted_at)} "
                # f"elapsed={_format_duration(now - self.start_time)}",
                flush=True,
            )
        self.maybe_snapshot(now)

    def maybe_snapshot(self, now: Optional[float] = None) -> None:
        """Print active slot details at the configured interval."""
        if not self.enabled or self.interval_seconds is None or self.interval_seconds <= 0:
            return
        now = time.monotonic() if now is None else now
        if now - self.last_snapshot_time < self.interval_seconds:
            return
        print(
            f"{self.phase} active: active={len(self.active)} "
            f"elapsed={_format_duration(now - self.start_time)}",
            flush=True,
        )
        for _, (item, device, submitted_at) in sorted(
            self.active.items(),
            key=lambda pair: (_device_sort_key(pair[1][1]), _sort_grid_key(_grid_key(pair[1][0])), pair[0]),
        ):
            print(
                f"  {device} {_format_item(item)} "
                f"running={_format_duration(now - submitted_at)}",
                flush=True,
            )
        self.last_snapshot_time = now


def _grid_key(item: WorkItem) -> Tuple[int, float, int, float, float]:
    """Return the tuple used for progress counts."""
    return (
        int(item.n),
        float(item.synthetic_anisotropy_power),
        int(item.m),
        float(item.alpha),
        float(item.beta),
    )


def _item_key(item: WorkItem) -> Tuple[int, float, int, float, float, int, Tuple[int, ...]]:
    """Return a unique key for a scheduled work/profiling item."""
    return (*_grid_key(item), int(item.data_seed), tuple(int(seed) for seed in item.init_seeds))


def _sort_grid_key(key: Tuple[int, float, int, float, float]) -> Tuple[int, float, int, float, float]:
    """Sort progress keys by the visible `(n, m, alpha, beta)` tuple."""
    n, anisotropy_power, m, alpha, beta = key
    return (n, anisotropy_power, m, alpha, beta)


def _format_grid_key(key: Tuple[int, float, int, float, float]) -> str:
    """Format the progress tuple without seed/data-seed detail."""
    n, anisotropy_power, m, alpha, beta = key
    return (
        f"n={n:>5} anisotropy={_format_value(anisotropy_power):>3} "
        f"m={m:>5} alpha={_format_value(alpha):>3} beta={_format_value(beta):>4}"
    )


def _format_item(item: WorkItem) -> str:
    """Return a concise description of one active work/profiling item."""
    text = _format_grid_key(_grid_key(item))
    return f"{text} data_seed={item.data_seed} init_seeds={_format_seed_chunk(item.init_seeds)}"


def _format_seed_chunk(seeds: Sequence[int]) -> str:
    """Format one init-seed chunk as a compact range or short list."""
    if not seeds:
        return "[]"
    if len(seeds) == 1:
        return str(seeds[0])
    step = seeds[1] - seeds[0]
    if all(curr - prev == step for prev, curr in zip(seeds, seeds[1:])):
        if step == 1:
            return f"{seeds[0]}..{seeds[-1]}"
        return f"{seeds[0]}..{seeds[-1]} step={step}"
    return "[" + ", ".join(str(seed) for seed in seeds) + "]"


def _format_value(value: Any) -> str:
    """Format scalar values compactly for log lines."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class _ProgressPrinter:
    """Parent-process progress printer shared by profiling and work schedulers."""

    def __init__(self, label: str, total: int, interval_seconds: Optional[float] = None):
        self.label = label
        self.total = int(total)
        self.interval_seconds = None if interval_seconds is None else float(interval_seconds)
        self.start_time = time.monotonic()
        self.last_print_time = self.start_time

    def print(self, completed: int, force: bool = False) -> None:
        """Print progress when forced, completed, or the interval has elapsed."""
        now = time.monotonic()
        completed = int(completed)
        if not force and (self.interval_seconds is None or self.interval_seconds <= 0):
            return
        if (
            not force
            and completed < self.total
            and now - self.last_print_time < self.interval_seconds
        ):
            return

        elapsed = max(0.0, now - self.start_time)
        pct = 100.0 if self.total == 0 else 100.0 * completed / self.total
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = "unknown"
        if completed > 0 and completed < self.total and rate > 0:
            eta = _format_duration((self.total - completed) / rate)

        print(
            f"{self.label} progress: {completed}/{self.total} "
            f"({pct:.1f}%), elapsed={_format_duration(elapsed)}, "
            f"eta={eta}, rate={rate:.3f} chunks/s",
            flush=True,
        )
        self.last_print_time = now

# -------------------------------------------------------------------------- #
# --------------------------- worker execution ----------------------------- #
# -------------------------------------------------------------------------- #

def _run_item_with_retries(config: InitScaleProbeConfig, item: WorkItem, device: str) -> Dict[str, Any]:
    "Run one item, shrinking batch sizes after CUDA OOM when enabled."
    current = item
    while True:
        result = _run_one_item(config, current, device)
        if result["ok"] or not result.get("oom") or not config.retry_on_oom:
            return result

        retry = _shrink_after_oom(current, config)
        if retry == current: # if we are already at the minimum size
            return result
        current = retry

def _run_one_item(config: InitScaleProbeConfig, item: WorkItem, device: str) -> Dict[str, Any]:
    """
    Load data on one device and evaluate every init seed in the work item.
    data_seed is used here; init_seeds each produce one trajectory over the
    configured training-step checkpoints.
    """
    try:
        if device.startswith("cuda"):
            torch.cuda.set_device(int(device.split(":", 1)[1]))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        worker_config = \
            replace(
                config,
                synthetic_anisotropy_power=item.synthetic_anisotropy_power,
                batch_size=item.batch_size,
                jacobian_batch_size=item.jacobian_batch_size,
                parallel=False,
            )

        binary_data = load_binary_classification_data(
            dataset=worker_config.dataset,
            n=item.n,
            negative_classes=worker_config.negative_classes,
            positive_classes=worker_config.positive_classes,
            random_labels=worker_config.random_labels,
            device=device,
            seed=item.data_seed,
            reserve_last=worker_config.reserve_last,
            synthetic_d_in=worker_config.synthetic_d_in,
            synthetic_test_size=worker_config.synthetic_test_size,
            synthetic_projection_fraction=worker_config.synthetic_projection_fraction,
            synthetic_anisotropy_power=worker_config.synthetic_anisotropy_power,
        )

        rows = []
        for init_seed in item.init_seeds:
            rows.extend(
                _rows_for_trained_initialization(
                    worker_config,
                    binary_data,
                    n=item.n,
                    m=item.m,
                    alpha=item.alpha,
                    beta=item.beta,
                    data_seed=item.data_seed,
                    init_seed=init_seed,
                    device=device,
                )
            )

        peak_mb = torch.cuda.max_memory_reserved() / MB if device.startswith("cuda") else None
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return {
            "ok": True,
            "rows": rows,
            "peak_mb": peak_mb,
            "batch_size": item.batch_size,
            "jacobian_batch_size": item.jacobian_batch_size,
        }
    except RuntimeError as exc:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        msg = str(exc).lower()
        oom = "cuda" in msg and "out of memory" in msg
        return {"ok": False, "oom": oom, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "oom": False, "error": str(exc)}

def _shrink_after_oom(item: WorkItem, config: InitScaleProbeConfig) -> WorkItem:
    new_jac = _shrunk(item.jacobian_batch_size, config)
    if new_jac < item.jacobian_batch_size:
        return replace(item, jacobian_batch_size=new_jac)
    new_batch = _shrunk(item.batch_size, config)
    if new_batch < item.batch_size:
        return replace(item, batch_size=new_batch)
    return item

def _shrunk(value: int, config: InitScaleProbeConfig) -> int:
    """Shrink a batch size by the configured OOM factor without crossing minimum."""
    if value <= config.min_batch_size:
        return value
    return max(config.min_batch_size, int(math.floor(value * config.oom_shrink_factor)))

def _checked_worker_result(result: Mapping[str, Any], item: WorkItem) -> List[Dict[str, Any]]:
    if result.get("ok"):
        return list(result["rows"])
    raise RuntimeError(
        "Probe worker failed for "
        f"n={item.n}, data_seed={item.data_seed}, m={item.m}, "
        f"anisotropy={item.synthetic_anisotropy_power}, alpha={item.alpha}, "
        f"beta={item.beta}, init_seeds={list(item.init_seeds)}: "
        f"{result.get('error', 'unknown error')}"
    )
