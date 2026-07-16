from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .sweep import (
    CORE_METRICS,
    metric_names_present_in_rows,
    normalize_common_sweep_config,
    normalize_init_seeds,
    prepare_output_dir,
    read_csv,
    resolve_common_metric_fields,
    resolve_run_device,
    run_sweep_rows,
    sort_rows,
    summarize_training_stats_rows,
    validate_common_data_and_sweep_fields,
    validate_common_execution_fields,
    validate_plot_metrics_in_rows,
    write_rows_output,
)


# -------------------------------------------------------------------------- #
# --------------------------------- config --------------------------------- #
# -------------------------------------------------------------------------- #

@dataclass
class TrainingStatsConfig:
    """Public config for training-time metric sweeps over checkpoint steps."""

    # data
    dataset: str = "digits"
    random_labels: bool = False
    reserve_last: int = 1000
    negative_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    positive_classes: List[int] = field(default_factory=lambda: [5, 6, 7, 8, 9])
    synthetic_d_in: int = 784
    synthetic_test_size: int = 0
    synthetic_projection_fraction: float = 0.25
    synthetic_anisotropy_power: float = 1.0
    synthetic_anisotropy_powers: Optional[List[float]] = None
    # sweep
    n_values: List[int] = field(default_factory=lambda: [10])
    m_values: List[int] = field(default_factory=lambda: [10])
    alpha_values: List[float] = field(default_factory=lambda: [1.0])
    beta_values: List[float] = field(default_factory=lambda: [float("inf")])
    training_step_values: Optional[List[int]] = None
    eta: float = 0.001
    init_type: str = "alpha"
    # randomness
    data_seed: int = 0
    data_seeds: List[int] = field(init=False) # Internal canonical single-data-seed list.
    num_data_seeds: Optional[int] = None # Rejected legacy field.
    init_seeds: Optional[List[int]] = None # Effective __post_init__ default is init_seed_start + k.
    num_inits: int = 2
    init_seed_start: int = 10000
    report_data_seed: Optional[int] = None # Rejected legacy field.
    # execution
    device: str = "cpu"
    batch_size: int = 1024
    jacobian_batch_size: int = 256
    parallel: bool = False
    gpu_ids: Optional[List[int]] = None
    init_chunk_size: int = 4
    adaptive_gpu_packing: bool = True
    gpu_memory_safety_fraction: float = 0.75
    gpu_reserved_memory_mb: int = 1000
    max_workers_per_gpu: int = 8
    retry_on_oom: bool = True
    oom_shrink_factor: float = 0.5
    min_batch_size: int = 1
    # logging
    progress_interval_seconds: Optional[float] = None
    progress_detail: str = "summary"
    # plotting
    tracked_metrics: Optional[List[str]] = None
    plot_metrics: List[str] = field(default_factory=lambda: list(CORE_METRICS))
    ntk_label_energy_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 50])
    plot_format: str = "combined"
    plot_heatmaps: bool = True
    # output
    output_dir: Path = Path("plots/training_stats/debug")

    def __post_init__(self):
        """Normalize and validate the public training-stats config surface."""
        normalize_common_sweep_config(self)
        self._normalize_training_steps()
        self._normalize_data_seed()
        normalize_init_seeds(self)
        validate_common_data_and_sweep_fields(self)
        self._validate_training_stats_fields()
        validate_common_execution_fields(self)
        resolve_common_metric_fields(self)

    def _normalize_training_steps(self) -> None:
        """Normalize the required multi-step training sweep axis."""
        if self.training_step_values is None:
            raise ValueError("training_stats requires training_step_values with at least two steps.")
        self.training_step_values = sorted({int(x) for x in self.training_step_values})

    def _normalize_data_seed(self) -> None:
        """Normalize the intentionally single data-seed interface."""
        self.data_seed = int(self.data_seed)
        if self.num_data_seeds is not None:
            raise ValueError("training_stats uses data_seed; num_data_seeds is not supported.")
        if self.report_data_seed is not None:
            raise ValueError("training_stats uses data_seed directly; report_data_seed is not supported.")
        self.data_seeds = [self.data_seed]

    def _validate_training_stats_fields(self) -> None:
        """Validate fields whose semantics are specific to training_stats."""
        if len(self.synthetic_anisotropy_powers) > 1:
            raise ValueError("training_stats does not support synthetic_anisotropy_powers sweeps.")
        if len(self.training_step_values) < 2:
            raise ValueError("training_stats requires at least two distinct training_step_values.")
        if any(step < 0 for step in self.training_step_values):
            raise ValueError("training_step_values must be non-negative.")


# -------------------------------------------------------------------------- #
# ----------------------------- public entrypoints ------------------------- #
# -------------------------------------------------------------------------- #

def run_experiment(config: TrainingStatsConfig):
    """Run a training-stats experiment and write rows plus plots."""
    if config.parallel:
        from .parallel import run_parallel
        return run_parallel(config)

    device = resolve_run_device(config)

    output_dir = config.output_dir
    rows = run_sweep_rows(
        config,
        device=device,
        data_seeds=config.data_seeds,
        synthetic_anisotropy_powers=config.synthetic_anisotropy_powers or [config.synthetic_anisotropy_power],
    )
    summary_rows = summarize_training_stats_rows(rows, config.tracked_metrics or [])

    paths = write_rows_output(output_dir, "_training_stats_rows.csv", rows)
    from .plotting import plot_training_summaries

    paths.update(plot_training_summaries(summary_rows, config, output_dir))
    return rows, summary_rows, paths


def plot_from_rows(
    config: TrainingStatsConfig,
    rows_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, Path]]:
    """Recreate training-stats plots from a raw rows CSV."""
    rows_path = Path(rows_path or config.output_dir / "_training_stats_rows.csv").expanduser()
    rows = read_csv(rows_path)
    if not rows:
        raise ValueError(f"Rows CSV is empty: {rows_path}")
    rows = sort_rows(rows)

    validate_plot_metrics_in_rows(config, rows)
    metric_names = metric_names_present_in_rows(config, rows)
    summary_rows = summarize_training_stats_rows(rows, metric_names)

    prepare_output_dir(config.output_dir)
    paths = {"rows": rows_path}
    from .plotting import plot_training_summaries

    paths.update(plot_training_summaries(summary_rows, config, config.output_dir))
    return rows, summary_rows, paths
