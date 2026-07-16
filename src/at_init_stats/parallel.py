"""Parallel execution wrapper for the at-init-stats experiment."""

from ..training_stats.sweep import summarize_all_seed_rows
from ..training_stats.parallel import run_parallel_with_outputs
from .plotting import plot_initialization_summaries


def run_parallel(config):
    return run_parallel_with_outputs(
        config,
        rows_filename="_at_init_stats_rows.csv",
        summarize_fn=summarize_all_seed_rows,
        plot_fn=plot_initialization_summaries,
        include_init_data_variability=True,
    )
