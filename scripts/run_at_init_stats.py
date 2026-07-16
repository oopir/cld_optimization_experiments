#!/usr/bin/env python3
from stats_cli import configure_repo_root

configure_repo_root(__file__)

from stats_cli import run_experiment_cli
from src.at_init_stats import AtInitStatsConfig, plot_from_rows, run_experiment


if __name__ == "__main__":
    run_experiment_cli(
        experiment_name="at-initialization statistics",
        config_cls=AtInitStatsConfig,
        run_experiment=run_experiment,
        plot_from_rows=plot_from_rows,
        default_rows_filename="_at_init_stats_rows.csv",
    )
