from pathlib import Path
import os
import re
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-at-init-stats-tests")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
import torch

from src.base.model import TwoLayerNet
from src.training_stats.sweep import write_csv
from src.at_init_stats.core import AtInitStatsConfig
from src.training_stats.metrics import get_metrics
from src.training_stats.core import TrainingStatsConfig, plot_from_rows as training_stats_plot_from_rows
from src.training_stats.plotting import (
    _grouped_metric_names,
    _make_final_test_error_vs_m_figure,
    _make_nm_heatmaps_figure,
    _make_training_curves_figure,
    _save_figures_pdf_equal_width,
)


def _summary_rows(metric_name="empirical_loss", values=(4.0, 2.0, 1.0, 0.5), steps=(0, 1, 10, 100)):
    return [
        {
            "dataset": "digits",
            "init_type": "standard",
            "n": 8,
            "n_effective": 8,
            "m": 16,
            "alpha": 1.0,
            "beta": float("inf"),
            "training_steps": step,
            "synthetic_anisotropy_power": 1.0,
            "eta": 0.001,
            "data_seed": 0,
            "num_inits": 1,
            f"{metric_name}_mean": value,
            f"{metric_name}_std": 0.0,
        }
        for step, value in zip(steps, values)
    ]


def _test_error_summary_rows():
    rows = []
    for n in (8, 16):
        for m in (16, 64):
            for step in (0, 100):
                rows.append({
                    "dataset": "digits",
                    "init_type": "standard",
                    "n": n,
                    "n_effective": n,
                    "m": m,
                    "alpha": 1.0,
                    "beta": float("inf"),
                    "training_steps": step,
                    "synthetic_anisotropy_power": 1.0,
                    "eta": 0.001,
                    "data_seed": 0,
                    "num_inits": 2,
                    "test_error_mean": 0.5 if step == 0 else 0.1 + 0.01 * (n == 16) + 0.001 * (m == 64),
                    "test_error_std": 0.01,
                })
    return rows


def _multi_width_summary_rows(
    metric_name,
    steps=(0, 5, 25, 125),
    m_values=(64, 128),
    n=1000,
    beta=float("inf"),
):
    rows = []
    for m_idx, m in enumerate(m_values):
        for step_idx, step in enumerate(steps):
            rows.append({
                "dataset": "mnist",
                "init_type": "standard",
                "n": n,
                "n_effective": n,
                "m": m,
                "alpha": 1.0,
                "beta": beta,
                "training_steps": step,
                "synthetic_anisotropy_power": 1.0,
                "eta": 0.001,
                "data_seed": 0,
                "num_inits": 2,
                f"{metric_name}_mean": 0.01 * (step_idx + 1) * (m_idx + 1),
                f"{metric_name}_std": 0.0,
            })
    return rows


def _residual_over_initial_summary_rows():
    rows = []
    values_by_m = {
        64: (1.0, 0.99, 0.96, 0.85),
        128: (1.0, 0.99, 0.96, 0.90),
    }
    for m, values in values_by_m.items():
        for step, value in zip((0, 5, 25, 125), values):
            rows.append({
                "dataset": "mnist",
                "init_type": "standard",
                "n": 1000,
                "n_effective": 1000,
                "m": m,
                "alpha": 1.0,
                "beta": float("inf"),
                "training_steps": step,
                "synthetic_anisotropy_power": 1.0,
                "eta": 0.001,
                "data_seed": 0,
                "num_inits": 2,
                "residual_ntk_alignment_over_initial_mean": value,
                "residual_ntk_alignment_over_initial_std": 0.0,
            })
    return rows


def _assert_visible_text_inside_figure(test_case, fig, tolerance_px=2.0):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text():
            continue
        bbox = text.get_window_extent(renderer=renderer)
        if not bbox.overlaps(fig_bbox):
            continue
        test_case.assertGreaterEqual(bbox.x0, fig_bbox.x0 - tolerance_px, text.get_text())
        test_case.assertGreaterEqual(bbox.y0, fig_bbox.y0 - tolerance_px, text.get_text())
        test_case.assertLessEqual(bbox.x1, fig_bbox.x1 + tolerance_px, text.get_text())
        test_case.assertLessEqual(bbox.y1, fig_bbox.y1 + tolerance_px, text.get_text())


def _visible_text_values(fig):
    return [text.get_text() for text in fig.findobj(match=Text) if text.get_visible() and text.get_text()]


def _pdf_media_boxes(path):
    contents = path.read_bytes().decode("latin-1", errors="ignore")
    return re.findall(r"/MediaBox\s*\[([^\]]+)\]", contents)


class AtInitStatsMetricTests(unittest.TestCase):
    def test_parameter_norm_metrics_include_hidden_and_output_layers(self):
        model = TwoLayerNet(d_in=2, m=2, d_out=1, init_type="standard", alpha=1.0)
        with torch.no_grad():
            model.fc1.weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
            model.fc2.weight.copy_(torch.tensor([[5.0, 12.0]]))

        metrics = get_metrics(
            model,
            torch.zeros((1, 2)),
            torch.ones((1, 1)),
            [
                "fc1_weight_fro_norm",
                "fc1_weight_fro_norm_normalized",
                "fc1_weight_spectral_norm",
                "fc1_weight_spectral_norm_normalized",
                "fc2_weight_euclidean_norm",
            ],
        )

        self.assertAlmostEqual(metrics["fc1_weight_fro_norm"], 5.0)
        self.assertAlmostEqual(metrics["fc1_weight_fro_norm_normalized"], 5.0 / np.sqrt(2.0))
        self.assertAlmostEqual(metrics["fc1_weight_spectral_norm"], 5.0)
        self.assertAlmostEqual(metrics["fc1_weight_spectral_norm_normalized"], 2.5)
        self.assertAlmostEqual(metrics["fc2_weight_euclidean_norm"], 13.0)

    def test_train_and_test_error_use_sign_threshold(self):
        model = TwoLayerNet(d_in=1, m=1, d_out=1, init_type="standard", alpha=1.0)
        with torch.no_grad():
            model.fc1.weight.fill_(1.0)
            model.fc2.weight.fill_(1.0)

        X_train = torch.tensor([[-1.0], [1.0], [2.0]])
        y_train = torch.tensor([[-1.0], [1.0], [-1.0]])
        X_test = torch.tensor([[-2.0], [0.0], [3.0]])
        y_test = torch.tensor([[-1.0], [-1.0], [1.0]])

        metrics = get_metrics(
            model,
            X_train,
            y_train,
            ["train_error", "test_error"],
            batch_size=2,
            X_test=X_test,
            y_test=y_test,
        )

        self.assertAlmostEqual(metrics["train_error"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["test_error"], 1.0 / 3.0)


class ExperimentSplitConfigTests(unittest.TestCase):
    def test_at_init_stats_rejects_training_step_sweeps(self):
        with self.assertRaisesRegex(ValueError, "training_stats"):
            AtInitStatsConfig(training_step_values=[0, 1])

    def test_at_init_stats_accepts_one_fixed_nonzero_step(self):
        config = AtInitStatsConfig(training_step=3)

        self.assertEqual(config.training_step, 3)
        self.assertEqual(config.training_step_values, [3])

    def test_at_init_stats_rejects_report_data_seed(self):
        with self.assertRaisesRegex(ValueError, "report_data_seed"):
            AtInitStatsConfig(report_data_seed=0)

    def test_training_stats_requires_multiple_steps(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            TrainingStatsConfig(training_step_values=[0])

    def test_training_stats_rejects_legacy_data_seed_fields(self):
        with self.assertRaises(TypeError):
            TrainingStatsConfig(training_step_values=[0, 1], data_seeds=[0])

        for kwargs in (
            {"num_data_seeds": 2},
            {"report_data_seed": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "data_seed|report_data_seed"):
                    TrainingStatsConfig(training_step_values=[0, 1], **kwargs)


class TrainingStatsPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_training_curves_use_log_shifted_steps_with_original_labels(self):
        fig = _make_training_curves_figure(_summary_rows(), "empirical_loss")
        ax = fig.axes[0]

        self.assertEqual(ax.get_xscale(), "log")
        np.testing.assert_allclose(ax.lines[0].get_xdata(), np.asarray([1.0, 2.0, 11.0, 101.0]))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ["0", "1", "10", "100"])

    def test_training_curves_use_sparse_log_position_x_tick_labels(self):
        steps = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        values = [float(len(steps) - idx) for idx in range(len(steps))]
        fig = _make_training_curves_figure(_summary_rows(values=values, steps=steps), "empirical_loss")
        ax = fig.axes[0]

        np.testing.assert_allclose(ax.get_xticks(), np.asarray([1.0, 11.0, 101.0, 2001.0, 20001.0]))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ["0", "10", "100", "2000", "20000"])

    def test_training_curves_use_log_y_for_positive_loss(self):
        fig = _make_training_curves_figure(_summary_rows(), "empirical_loss")

        self.assertEqual(fig.axes[0].get_yscale(), "log")

    def test_training_curves_use_log_y_for_positive_residual_metrics(self):
        fig = _make_training_curves_figure(_summary_rows(metric_name="residual_ntk_alignment"), "residual_ntk_alignment")

        self.assertEqual(fig.axes[0].get_yscale(), "log")

    def test_training_curves_fall_back_to_linear_y_when_loss_has_zero(self):
        fig = _make_training_curves_figure(_summary_rows(values=(4.0, 2.0, 0.0, 0.5)), "empirical_loss")

        self.assertEqual(fig.axes[0].get_yscale(), "linear")

    def test_over_initial_training_curves_draw_one_reference_line(self):
        for metric_name in ("residual_ntk_alignment_over_initial", "task_ntk_alignment_over_initial"):
            fig = _make_training_curves_figure(_summary_rows(metric_name=metric_name, values=(1.0, 1.1, 1.2, 1.3)), metric_name)
            ax = fig.axes[0]
            has_one_line = any(
                len(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), np.asarray([1.0, 1.0]))
                for line in ax.lines
            )
            self.assertTrue(has_one_line, metric_name)

    def test_training_curves_keep_long_spectral_norm_labels_inside_canvas(self):
        fig = _make_training_curves_figure(
            _multi_width_summary_rows("fc1_weight_spectral_norm_normalized"),
            "fc1_weight_spectral_norm_normalized",
        )

        self.assertIn("mean value", _visible_text_values(fig))
        self.assertIn("\n", fig._suptitle.get_text())
        _assert_visible_text_inside_figure(self, fig)

    def test_training_curves_keep_long_ntk_drift_labels_inside_canvas(self):
        fig = _make_training_curves_figure(
            _multi_width_summary_rows("ntk_rel_fro_dist"),
            "ntk_rel_fro_dist",
        )

        self.assertIn("mean value", _visible_text_values(fig))
        self.assertIn("\n", fig._suptitle.get_text())
        _assert_visible_text_inside_figure(self, fig)

    def test_training_curves_keep_log_y_tick_labels_separate_from_axis_label(self):
        fig = _make_training_curves_figure(
            _residual_over_initial_summary_rows(),
            "residual_ntk_alignment_over_initial",
        )

        self.assertIn("mean value", _visible_text_values(fig))
        _assert_visible_text_inside_figure(self, fig)

    def test_grouped_training_curve_pdf_save_uses_uncropped_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "training_curves.pdf"
            figures = [
                _make_training_curves_figure(
                    _multi_width_summary_rows("empirical_loss"),
                    "empirical_loss",
                ),
                _make_training_curves_figure(
                    _multi_width_summary_rows("fc1_weight_spectral_norm_normalized"),
                    "fc1_weight_spectral_norm_normalized",
                ),
                _make_training_curves_figure(
                    _multi_width_summary_rows("ntk_rel_fro_dist"),
                    "ntk_rel_fro_dist",
                ),
            ]

            self.assertTrue(_save_figures_pdf_equal_width(figures, path))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(len(set(_pdf_media_boxes(path))), 1)

    def test_nm_heatmaps_keep_long_title_inside_canvas(self):
        fig = _make_nm_heatmaps_figure(
            _multi_width_summary_rows("fc1_weight_spectral_norm_normalized"),
            "fc1_weight_spectral_norm_normalized",
        )

        self.assertIsNotNone(fig)
        self.assertIn("over beta and m", "\n".join(_visible_text_values(fig)))
        _assert_visible_text_inside_figure(self, fig)

    def test_final_test_error_vs_m_uses_final_step_and_log_width_axis(self):
        fig = _make_final_test_error_vs_m_figure(_test_error_summary_rows())
        ax = fig.axes[0]

        self.assertEqual(ax.get_xscale(), "log")
        self.assertIn("final step=100", fig._suptitle.get_text())
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ["16", "64"])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), np.asarray([0.1, 0.101]))

    def test_final_test_error_vs_m_keeps_legend_and_title_inside_canvas(self):
        fig = _make_final_test_error_vs_m_figure(_test_error_summary_rows())

        _assert_visible_text_inside_figure(self, fig)

    def test_train_error_groups_with_loss_metrics(self):
        groups = _grouped_metric_names(["empirical_loss", "train_error", "test_error"])

        self.assertEqual(groups["loss"], ["empirical_loss", "train_error", "test_error"])

    def test_parameter_norms_group_together(self):
        metric_names = [
            "fc1_weight_fro_norm",
            "fc1_weight_spectral_norm",
            "fc2_weight_euclidean_norm",
        ]

        groups = _grouped_metric_names(metric_names)

        self.assertEqual(groups["parameter_norms"], metric_names)

    def test_plot_only_accepts_minimal_existing_raw_rows_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rows_path = tmp_path / "old_rows.csv"
            output_dir = tmp_path / "plots"
            summary_path = output_dir / "_at_init_stats_summary.csv"
            write_csv(
                rows_path,
                [
                    {
                        "dataset": "digits",
                        "init_type": "standard",
                        "n": 8,
                        "n_effective": 8,
                        "m": 16,
                        "alpha": 1.0,
                        "beta": float("inf"),
                        "training_steps": step,
                        "synthetic_anisotropy_power": 1.0,
                        "eta": 0.001,
                        "data_seed": 0,
                        "init_seed": 0,
                        "device": "cpu",
                        "empirical_loss": value,
                    }
                    for step, value in zip([0, 1, 10, 100], [4.0, 2.0, 1.0, 0.5])
                ],
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text("stale summary\n")
            config = TrainingStatsConfig(
                dataset="digits",
                n_values=[8],
                m_values=[16],
                beta_values=[float("inf")],
                training_step_values=[0, 1, 10, 100],
                data_seed=0,
                init_seeds=[0],
                tracked_metrics=["empirical_loss"],
                plot_metrics=["empirical_loss"],
                plot_format="individual",
                plot_heatmaps=False,
                output_dir=output_dir,
            )

            rows, summary_rows, paths = training_stats_plot_from_rows(config, rows_path=rows_path)

            self.assertEqual(len(rows), 4)
            self.assertEqual(len(summary_rows), 4)
            self.assertNotIn("summary", paths)
            self.assertFalse(summary_path.exists())
            self.assertTrue(paths["plot_empirical_loss_training_curves"].exists())

    def test_training_stats_plot_only_rejects_multiple_data_seeds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rows_path = tmp_path / "multi_data_seed_rows.csv"
            write_csv(
                rows_path,
                [
                    {
                        "dataset": "digits",
                        "init_type": "standard",
                        "n": 8,
                        "n_effective": 8,
                        "m": 16,
                        "alpha": 1.0,
                        "beta": float("inf"),
                        "training_steps": step,
                        "synthetic_anisotropy_power": 1.0,
                        "eta": 0.001,
                        "data_seed": data_seed,
                        "init_seed": 0,
                        "device": "cpu",
                        "empirical_loss": 4.0 / (step + 1),
                    }
                    for data_seed in (0, 1)
                    for step in (0, 1)
                ],
            )
            config = TrainingStatsConfig(
                dataset="digits",
                n_values=[8],
                m_values=[16],
                beta_values=[float("inf")],
                training_step_values=[0, 1],
                data_seed=0,
                init_seeds=[0],
                tracked_metrics=["empirical_loss"],
                plot_metrics=["empirical_loss"],
                plot_format="individual",
                plot_heatmaps=False,
                output_dir=tmp_path / "plots",
            )

            with self.assertRaisesRegex(ValueError, "exactly one data_seed"):
                training_stats_plot_from_rows(config, rows_path=rows_path)


if __name__ == "__main__":
    unittest.main()
