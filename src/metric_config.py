"""
Metric configuration for experiment training, checkpointing, and plotting.

This module is the source of truth for configurable metric tracking. Users can
set ExpConfig.tracked_metrics to a list of metric names, and training will only
persist those requested metrics. When tracked_metrics is omitted, the resolver
falls back to the legacy behavior derived from track_jacobian,
collect_feature_stats, and use_linearized, so old configs continue to run with
the broad default metric set.

There are three important metric sets:

- tracked_metrics:
  The ordered metric names requested by the user/config and saved as checkpoint
  metadata. This list defines what a fresh run records. During resume, however,
  the checkpoint config is the source of truth: tracked_metrics from the incoming
  YAML is not treated as an override.

- history_metrics:
  Metrics that are recorded once per tracking epoch and saved under the existing
  "<metric>_hist" naming convention. Keeping this convention lets old plotting
  and analysis code continue to work with configurable metric runs.

- final_metrics:
  Metrics computed once at finalization rather than per tracking epoch, such as
  analysis-style scalar summaries.

Training may also compute metrics that are not tracked. These live in
compute_metrics. For example, progress printing always needs train_loss,
train_acc, and test_acc, so those metrics are computed even if the user does not
request them in tracked_metrics. Similarly, early_stop_metric is automatically
added to the requested metric list so the stopping criterion is visible in saved
histories and checkpoints.

Metric names are intentionally validated in one place. Unknown names raise a
ValueError near config loading/training setup instead of silently producing
missing histories. Linearized metrics also require use_linearized=True; the
resolver errors rather than silently enabling a different training path. Tuple
metrics such as nn_lin_param_dist and jacobian_dist are not valid early-stopping
metrics until a scalar component selector is added.

The dependency properties on MetricPlan tell training which expensive auxiliary
state is required. For example, jacobian_dist needs a frozen initialization
model, and feature metrics need hidden activations from initialization. These
dependencies control what is initialized/computed; they do not imply that the
dependency itself is saved as a metric.

Checkpoint behavior:

- New checkpoints store metric_schema_version and tracked_metrics.
- Resuming new checkpoints requires metric metadata and uses the checkpoint
  config as the source of truth. Incoming YAML tracked_metrics is ignored on
  resume because tracked metrics are not an overrideable field.
- Loading and plotting old checkpoints remains allowed. Old checkpoints simply
  lack metric metadata, so they are refused only when the user tries to resume
  them.

To add a metric:

1. Add its public name to BASE_METRIC_NAMES, LIN_METRIC_NAMES,
   PAIR_HISTORY_METRICS, or FINAL_METRICS depending on storage shape.
2. Add it to a dependency group if it requires special state or expensive shared
   computation.
3. Implement the computation in stats.py or training.py.
4. Make training append it only when it appears in MetricPlan.history_metrics, or
   store it at finalization only when it appears in MetricPlan.final_metrics.
5. Update plotting docstrings/warnings if a plotting function expects the metric.
"""


from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


BASE_METRIC_NAMES = (
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc",
    "param_dist",
    "feat_rel_dist",
    "feat_cos_dist",
    "feat_gram_lambda",
)

# Linearized-model scalar histories, stored under "<metric>_hist" when tracked.
LIN_METRIC_NAMES = (
    "lin_train_loss",
    "lin_train_acc",
    "lin_test_acc",
    "lin_param_dist",
)

# Tuple-valued histories, e.g. (normalized L2 distance, cosine distance).
PAIR_HISTORY_METRICS = (
    "nn_lin_param_dist",
    "jacobian_dist",
)

# Final metrics are computed once at the end and are not stored as histories.
FINAL_METRICS = (
    "loss_floor",
)

# Progress printing stays fixed, so these may be computed even if not stored.
PRINT_METRICS = (
    "train_loss",
    "train_acc",
    "test_acc",
)

# "Known" metrics are the full validated vocabulary accepted by tracked_metrics.
KNOWN_HISTORY_METRICS = BASE_METRIC_NAMES + LIN_METRIC_NAMES + PAIR_HISTORY_METRICS
KNOWN_FINAL_METRICS = FINAL_METRICS
KNOWN_METRICS = KNOWN_HISTORY_METRICS + KNOWN_FINAL_METRICS

# Dependency groups used to validate config and initialize expensive state.
LINEARIZED_METRICS = LIN_METRIC_NAMES + ("nn_lin_param_dist",)
FEATURE_METRICS = ("feat_rel_dist", "feat_cos_dist", "feat_gram_lambda")

METRIC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MetricPlan:
    """Resolved metric request, including internal metrics needed for printing/stopping."""
    # Metrics the user asked to persist in the checkpoint.
    tracked_metrics: Tuple[str, ...]
    # Tracked metrics that get one value per tracking epoch.
    history_metrics: frozenset
    # Tracked metrics that get one value at training finalization.
    final_metrics: frozenset
    # Metrics that must be computed, even if they aren't tracked, including non-persisted print/control metrics.
    compute_metrics: frozenset

    @property
    def needs_linearized_metrics(self):
        return any(name in self.compute_metrics for name in LINEARIZED_METRICS)

    @property
    def needs_jacobian_reference(self):
        return "jacobian_dist" in self.compute_metrics

    @property
    def needs_feature_activations(self):
        return any(name in self.compute_metrics for name in FEATURE_METRICS)

    @property
    def needs_initial_features(self):
        return any(name in self.compute_metrics for name in ("feat_rel_dist", "feat_cos_dist"))


def legacy_default_tracked_metrics(use_linearized=True, track_jacobian=True, collect_feature_stats=True):
    """
    Return the pre-configurable metric set implied by the legacy boolean flags.

    This keeps old configs behavior-compatible while letting new configs specify
    tracked_metrics directly.
    """
    metrics = list(BASE_METRIC_NAMES)
    if use_linearized:
        metrics.extend(LIN_METRIC_NAMES)
        metrics.append("nn_lin_param_dist")
    if track_jacobian:
        metrics.append("jacobian_dist")
    if collect_feature_stats:
        metrics.append("loss_floor")
    return metrics


def resolve_metric_plan(
    tracked_metrics: Optional[Iterable[str]],
    use_linearized: bool,
    track_jacobian: bool = True,
    collect_feature_stats: bool = True,
    early_stop_metric: Optional[str] = None,
):
    """Validate tracked_metrics and add internal dependencies for training control."""
    # No explicit list means "act like old configs": derive metrics from legacy flags.
    if tracked_metrics is None:
        requested = legacy_default_tracked_metrics(
            use_linearized=use_linearized,
            track_jacobian=track_jacobian,
            collect_feature_stats=collect_feature_stats,
        )
    else:
        requested = list(tracked_metrics)

    # Early stopping always needs its stopping-criteria-metric computed and saved
    if early_stop_metric is not None and early_stop_metric not in requested:
        requested.append(early_stop_metric)

    # Early stopping can only use some metrics as stopping criteria.
    if early_stop_metric is not None and early_stop_metric not in KNOWN_HISTORY_METRICS:
        raise ValueError(f"early_stop_metric must be a history metric, got {early_stop_metric!r}")
    if early_stop_metric in PAIR_HISTORY_METRICS:
        raise ValueError(
            f"early_stop_metric must be scalar; {early_stop_metric!r} stores an (L2, cosine) tuple."
        )

    # Remove duplicates and catch typos
    deduped = []
    seen = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    unknown = sorted(set(deduped) - set(KNOWN_METRICS))
    if unknown:
        raise ValueError(f"Unknown tracked metric(s): {', '.join(unknown)}")

    # Ensure that use_linearized was set to True if some metrics require it.
    requested_linearized = sorted(set(deduped) & set(LINEARIZED_METRICS))
    if requested_linearized and not use_linearized:
        raise ValueError(
            "Linearized metric(s) requested while use_linearized=False: "
            + ", ".join(requested_linearized)
        )

    history_metrics = frozenset(name for name in deduped if name in KNOWN_HISTORY_METRICS)
    final_metrics = frozenset(name for name in deduped if name in KNOWN_FINAL_METRICS)
    compute_metrics = set(history_metrics)
    compute_metrics.update(PRINT_METRICS)
    return MetricPlan(
        tracked_metrics=tuple(deduped),
        history_metrics=history_metrics,
        final_metrics=final_metrics,
        compute_metrics=frozenset(compute_metrics),
    )
