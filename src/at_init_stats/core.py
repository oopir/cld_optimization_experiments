from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..training_stats.sweep import (
    CORE_METRICS,
    metric_names_present_in_rows,
    normalize_common_sweep_config,
    normalize_init_seeds,
    normalize_seed_list,
    prepare_output_dir,
    read_csv,
    resolve_common_metric_fields,
    resolve_run_device,
    run_sweep_rows,
    sort_rows,
    summarize_all_seed_rows,
    summarize_data_averaged_init_variability_rows,
    summarize_init_averaged_data_variability_rows,
    validate_common_data_and_sweep_fields,
    validate_common_execution_fields,
    validate_plot_metrics_in_rows,
    write_rows_output,
)


@dataclass
class AtInitStatsConfig:
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
    training_step: int = 0
    training_step_values: Optional[List[int]] = None # Legacy: at most one value; normalized to training_step.
    eta: float = 0.001
    init_type: str = "alpha"
    # randomness
    data_seeds: Optional[List[int]] = None
    num_data_seeds: int = 1
    data_seed_start: int = 0
    init_seeds: Optional[List[int]] = None
    num_inits: int = 2
    init_seed_start: int = 10000
    report_data_seed: Optional[int] = None # Legacy only; at-init summaries use all configured data seeds.
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
    output_dir: Path = Path("plots/at_init_stats/debug")

    def __post_init__(self) -> None:
        normalize_common_sweep_config(self)
        self._normalize_fixed_training_step()
        self._normalize_data_seeds()
        normalize_init_seeds(self)
        validate_common_data_and_sweep_fields(self)
        self._validate_at_init_stats_fields()
        validate_common_execution_fields(self)
        resolve_common_metric_fields(self)

    def _normalize_fixed_training_step(self) -> None:
        self.training_step = int(self.training_step)
        if self.training_step_values is not None:
            legacy_values = sorted({int(x) for x in self.training_step_values})
            if len(legacy_values) != 1:
                raise ValueError(
                    "at_init_stats supports one fixed measurement step only. "
                    "Use the training_stats experiment for training_step_values sweeps."
                )
            if self.training_step != 0 and self.training_step != legacy_values[0]:
                raise ValueError(
                    "Conflicting at_init_stats step settings: training_step="
                    f"{self.training_step} but training_step_values={legacy_values}."
                )
            self.training_step = legacy_values[0]
        self.training_step_values = [self.training_step]

    def _normalize_data_seeds(self) -> None:
        self.data_seeds, self.num_data_seeds, self.data_seed_start = normalize_seed_list(
            self.data_seeds,
            self.num_data_seeds,
            self.data_seed_start,
            seeds_name="data_seeds",
            count_name="num_data_seeds",
        )

    def _validate_at_init_stats_fields(self) -> None:
        if self.report_data_seed is not None:
            raise ValueError(
                "report_data_seed is no longer used by at_init_stats; at_init_stats summaries "
                "average across all configured data_seeds. Use training_stats with data_seed "
                "for training-step sweeps."
            )
        if self.training_step < 0:
            raise ValueError("training_step must be non-negative.")
        if len(self.synthetic_anisotropy_powers) > 1:
            if self.dataset != "synthetic_anisotropic":
                raise ValueError(
                    "synthetic_anisotropy_powers with multiple values requires "
                    "dataset='synthetic_anisotropic'."
                )
            if len(self.m_values) != 1:
                raise ValueError(
                    "synthetic_anisotropy_powers with multiple values requires exactly one m_values entry."
                )


def run_experiment(config: AtInitStatsConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Path]]:
    if config.parallel:
        from ..training_stats.parallel import run_parallel_with_outputs
        from .plotting import plot_initialization_summaries

        return run_parallel_with_outputs(
            config,
            rows_filename="_at_init_stats_rows.csv",
            summarize_fn=summarize_all_seed_rows,
            plot_fn=plot_initialization_summaries,
            include_init_data_variability=True,
        )

    device = resolve_run_device(config)

    output_dir = config.output_dir
    rows = run_sweep_rows(
        config,
        device=device,
        data_seeds=config.data_seeds or [],
        synthetic_anisotropy_powers=config.synthetic_anisotropy_powers or [config.synthetic_anisotropy_power],
    )
    summary_rows = summarize_all_seed_rows(rows, config.tracked_metrics or [])
    data_averaged_init_variability_rows = summarize_data_averaged_init_variability_rows(rows, config.tracked_metrics or [])
    init_averaged_data_variability_rows = summarize_init_averaged_data_variability_rows(rows, config.tracked_metrics or [])

    paths = write_rows_output(output_dir, "_at_init_stats_rows.csv", rows)
    from .plotting import plot_initialization_summaries

    paths.update(
        plot_initialization_summaries(
            summary_rows,
            config,
            output_dir,
            data_averaged_init_variability_rows=data_averaged_init_variability_rows,
            init_averaged_data_variability_rows=init_averaged_data_variability_rows,
        )
    )
    return rows, summary_rows, paths


def plot_from_rows(
    config: AtInitStatsConfig,
    rows_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Path]]:
    """Regenerate summaries and plots from a saved raw rows CSV."""
    rows_path = Path(rows_path or config.output_dir / "_at_init_stats_rows.csv").expanduser()
    rows = read_csv(rows_path)
    if not rows:
        raise ValueError(f"Rows CSV is empty: {rows_path}")
    rows = sort_rows(rows)

    validate_plot_metrics_in_rows(config, rows)
    metric_names = metric_names_present_in_rows(config, rows)
    summary_rows = summarize_all_seed_rows(rows, metric_names)
    init_seed_summary_rows = summarize_data_averaged_init_variability_rows(rows, metric_names)
    data_seed_summary_rows = summarize_init_averaged_data_variability_rows(rows, metric_names)

    prepare_output_dir(config.output_dir)
    paths = {"rows": rows_path}
    from .plotting import plot_initialization_summaries

    paths.update(
        plot_initialization_summaries(
            summary_rows,
            config,
            config.output_dir,
            data_averaged_init_variability_rows=init_seed_summary_rows,
            init_averaged_data_variability_rows=data_seed_summary_rows,
        )
    )
    return rows, summary_rows, paths
