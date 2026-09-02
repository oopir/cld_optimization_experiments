# Repository context for Codex

## Purpose

This repository studies how finite-width two-layer neural networks evolve under
Langevin dynamics. Its central comparison is between the nonlinear neural
network trajectory and the trajectory of its first-order linearization around
initialization, across choices such as model width, scaling, dataset size, and
inverse temperature.

The experiments use two-layer tanh networks, full-training-set gradients, and
classification through mean-squared loss against one-hot targets. The primary
datasets are Digits and MNIST.

## Mathematical context

The baseline continuous-time Langevin dynamics are

\[
d\theta_t
=
-\nabla \mathcal L(\theta_t)\,dt
-\beta^{-1}\Lambda\theta_t\,dt
+\sqrt{2\beta^{-1}}\,dW_t.
\]

Here `beta` is the inverse temperature and `Lambda` is represented by fixed
per-parameter diagonal tensors. The implemented overdamped update is the
Euler--Maruyama discretization of this equation. At `beta = inf`, stochastic
noise and the beta-scaled regularization term vanish.

The linearized model is the first-order expansion around initialization:

\[
f_{\mathrm{lin}}(\theta, x)
=
f(\theta_0, x)
+J_{\theta_0}(x)(\theta-\theta_0).
\]

The nonlinear and linearized trajectories should be constructed from the same
initial parameters. When an experiment requests shared noise, comparisons use
the same underlying standard-normal samples before each dynamics applies its
own scaling.

## Architecture and execution flow

The main flow is:

```text
YAML/JSON config
    -> ExpConfig and RunOpts
    -> alpha/beta experiment pairs
    -> per-seed workers
    -> trajectories and tracked metrics
    -> optional checkpoint
    -> plots
```

Important files:

- `src/config.py`: experiment/run dataclasses and checkpoint serialization.
- `src/exp.py`: config parsing, alpha/beta sweeps, eta tuning, checkpoint
  extension, result merging, and top-level orchestration.
- `src/training.py`: per-seed initialization, optional linearization, metric
  recording, numerical updates, early stopping, and resume-state restoration.
- `src/langevin.py`: numerical Langevin update kernels. Keep mathematical
  update ordering explicit when editing this file.
- `src/linearized.py`: initialization and forward evaluation of the first-order
  model around `theta_0`.
- `src/metric_config.py`: source of truth for accepted metric names, metric
  dependencies, and persisted metric selection.
- `src/stats.py`: metric implementations and analysis computations.
- `src/plots.py`: multiseed aggregation and experiment plots.
- `src/data.py` and `src/model.py`: datasets, preprocessing, network, loss, and
  diagonal regularization tensors.
- `scripts/run_exp_from_config.py`: primary command-line entry point. It must be
  run from the repository root.

Configurations normally have an `experiment` section for `ExpConfig` and a
`run` section for `RunOpts`. Results are grouped by alpha/beta label and then by
seed. History metrics use the `<metric>_hist` naming convention.

## Initialization, comparisons, and checkpoints

Initialization-dependent references must remain distinct from current training
state. In particular, `params0`, linearization base parameters, initial feature
statistics, and Jacobian references describe `theta_0`; resumed `start_*` state
describes the checkpoint endpoint. Reconstruct initialization-dependent objects
before restoring current trajectory positions and RNG state.

Checkpoint extension should preserve:

- Original initialization state.
- Current nonlinear and linearized parameters.
- Any additional trajectory state required by enabled dynamics.
- Python, NumPy, Torch CPU, and relevant CUDA RNG states.
- Metric histories, epoch history, effective final epoch, and early-stop state.

If a proposed change may alter checkpoint compatibility, result keys, config
semantics, metric schemas, or seeded numerical behavior, explain the risk and
let the user decide before implementing the breaking change.

## Compatibility-only constructs

Do not treat the following as preferred architecture for new code:

- `src/metric_checkpoints.py` is an import compatibility shim for old Torch
  pickles referring to historical config classes.
- `ExpConfig.track_jacobian` and `ExpConfig.collect_feature_stats` participate
  in legacy metric behavior when `tracked_metrics` is omitted. Explicit
  `tracked_metrics` is the current interface.
- `patch_loaded_config` in `src/config.py` fills fields absent from older
  pickled dataclass instances.
- Checkpoint payloads without metric metadata may be loaded and plotted but are
  intentionally rejected for resume.
- `_IGNORED_LEGACY_EXP_FIELDS` and compatibility parsing in `src/exp.py` accept
  selected obsolete config fields without making them current API.
- `infer_effective_track_every` exists for historical checkpoint plotting.
- Scripts named `fix_*`, `migrate_*`, or otherwise referring to old/deadline
  checkpoints are one-off compatibility and migration utilities, not normal
  experiment entry points.

Preserve these paths when compatibility matters, but do not copy their patterns
into new features without a concrete reason.

## Strongly encouraged code style

- Keep function calls compact. Group related arguments on the same line instead
  of placing every argument on its own line.
- Use section comments to make long orchestration functions easy to scan.
- Extract helpers only when they materially improve readability. Helpers should
  have a narrow, coherent responsibility and be similar in scope to neighboring
  helpers.
- Preserve user edits and avoid broad, unrelated refactors.
- Prefer behavior-preserving changes unless the requested task requires a
  semantic change.
- Keep numerical operations explicit and easy to audit. Avoid hiding multiple
  state mutations in a chained expression when separate lines communicate the
  update more clearly.
- Inspect callers, checkpoint paths, metrics, and tests before changing shared
  interfaces.
- When mathematical behavior is ambiguous, establish the intended mathematics
  with the user before implementing it.

These are strong preferences rather than mechanical formatting requirements.
Follow the style of the surrounding file when it conveys the same intent more
clearly.

## Verification and resource limits

The intended Python environment is `~/.venv`. Use its interpreter directly,
for example:

```sh
~/.venv/bin/python -m py_compile src/*.py scripts/*.py tests/*.py
```

Only run very low-computation sanity checks unless the user explicitly requests
more. Targeted synthetic-data unit tests are acceptable. Do not launch full
experiments, broad parameter sweeps, MNIST downloads, eta tuning, or expensive
Jacobian computations merely because CUDA is available.

Verification must not create `.pt` files. CUDA is typically available, but its
availability is not authorization to run expensive GPU work.

Before reporting completion, at minimum run an appropriate syntax/static check
and `git diff --check` when those commands are available. If dependencies or the
environment prevent a runtime test, report that limitation plainly.
