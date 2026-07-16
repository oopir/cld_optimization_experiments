#!/usr/bin/env python3
from stats_cli import configure_repo_root

configure_repo_root(__file__)

from stats_cli import run_experiment_cli
from src.training_stats import TrainingStatsConfig, plot_from_rows, run_experiment


if __name__ == "__main__":
    run_experiment_cli(
        experiment_name="training-stats",
        config_cls=TrainingStatsConfig,
        run_experiment=run_experiment,
        plot_from_rows=plot_from_rows,
        default_rows_filename="_training_stats_rows.csv",
    )
