# Global Convergence in Deep Networks via the Second Law of Thermodynamics

Code for the numerical experiments shown in the paper [Global Convergence in Deep Networks via the Second Law of Thermodynamics]().

## Requirements

The final configurations require Linux with a CUDA-capable GPU. They were run
with Python 3.11, PyTorch 2.4.1, and CUDA 12.1. The supplied `environment.yml`
provides a reproducible Conda setup:

```bash
conda env create -f environment.yml
conda activate cld-opt
```

Conda is not required: another environment manager may be used if it provides
the runtime dependencies and versions specified in `environment.yml`.

Run all commands below from the repository root. The scripts intentionally
reject other working directories so that relative configuration and output
paths resolve consistently.

## Quick Start

These commands rerun the shallow-network experiment described in the paper.
Run all five seed configurations, then retain the printed checkpoint paths for
the merge step.

```bash
python scripts/run_exp_from_config.py --config configs/shallow/seed0.yaml
python scripts/run_exp_from_config.py --config configs/shallow/seed1.yaml
python scripts/run_exp_from_config.py --config configs/shallow/seed2.yaml
python scripts/run_exp_from_config.py --config configs/shallow/seed3.yaml
python scripts/run_exp_from_config.py --config configs/shallow/seed4.yaml
```

Merge those checkpoint paths and generate the plots:

```bash
python scripts/merge_metric_checkpoints.py --out cld_checkpoints/shallow/merged_metrics.pt <seed-0-checkpoint> <seed-1-checkpoint> <seed-2-checkpoint> <seed-3-checkpoint> <seed-4-checkpoint>

python scripts/plot_checkpoint.py cld_checkpoints/shallow/merged_metrics.pt --outdir plots/shallow

python scripts/plot_metric_per_seed.py cld_checkpoints/shallow/merged_metrics.pt --metric feat_gram_lambda_hist --outdir plots/shallow
```

The deep-network experiment follows the same workflow with the corresponding
files under `configs/deep/`.

## Running an experiment

Every experiment is specified by a YAML file with an `experiment` section and
a `run` section.  `run.ckpt_dir` must always be specified. Run any one 
configuration by supplying its path:

```bash
python scripts/run_exp_from_config.py --config <config-path>
```

Each configuration runs every seed in `seeds` across its requested temperature
settings. When `run.save_ckpt: true`, it saves the selected metric histories in
one checkpoint and prints that checkpoint's path on completion.

### Experiment fields

#### Execution and data

- `seeds: [INT, ...]` (default: `[0]`): random seeds to run.
- `device: {cuda|cuda:INDEX|cpu}` (default: `cuda`): compute device.
- `gpu_indices: [INT]` (optional): select a single CUDA device by index.
- `dataset: {digits|mnist}` (default: `digits`): dataset to load. 
- `n: INT` (default: `10`): number of training examples.
- `random_labels: {true|false}` (default: `false`): replace training labels
  with seed-deterministic random labels.

#### Model and dynamics

- `m: INT` (default: `1`): hidden-layer width.
- `L: INT` (default: `1`): number of tanh hidden layers; must be at least one.
- `activation: tanh` (default: `tanh`): compatibility field; `tanh` is the
  only accepted value.
- `betas: [FLOAT|.inf, ...]` (default: `[]`): inverse-temperature sweep.

#### Training

- `eta: FLOAT` (default: `1.0`): scalar step size.
- `epochs: INT` (default: `1`): total number of update steps.
- `regularization_scale: FLOAT` (default: `1.0`): multiplier applied to the
  regularization term.
- `same_noise: {true|false}` (default: `false`): share Langevin noise between
  the nonlinear and linearized models. It is meaningful only when
  `use_linearized: true`.
- `noise_free_after_epoch: INT` (optional): switch to the deterministic update
  for epochs strictly after this value.
- `early_stop_metric: METRIC` and `early_stop_value: FLOAT` (optional): stop
  when the metric reaches the threshold.
- `early_stop_goal: {min|max}` (default: `min`): direction used for the early
  stopping comparison.

#### Step-size selection

- `eta_mode: {scalar|per_beta|per_alpha_beta}` (default: `scalar`): select the
  step-size lookup strategy.
- `eta_table_path: PATH` (optional): YAML table used by non-scalar `eta_mode`.
  Missing entries use `eta_default`, or `eta` when `eta_default` is omitted.
- `eta_default: FLOAT` (optional): fallback value for `eta_table_path`.

#### Metrics and logging

- `use_linearized: {true|false}` (default: `true`): also train and evaluate the
  linearized model. It must be true when requesting a `lin_*` metric or
  `nn_lin_param_dist`.
- `tracked_metrics: [METRIC, ...]` (optional): histories to save. If omitted,
  the runner selects its default loss, feature, and enabled comparison metrics.
  Accepted metrics are:
  - nonlinear: `train_loss`, `train_acc`, `test_acc`, `param_dist`,
    `feat_gram_lambda`;
  - linearized: `lin_train_loss`, `lin_train_acc`, `lin_test_acc`,
    `lin_param_dist`;
  - nonlinear/linearized comparisons: `nn_lin_param_dist`, `jacobian_dist`.
- `track_jacobian: {true|false}` (default: `true`): include Jacobian drift in
  the default metric selection. Set `tracked_metrics` explicitly to control
  exactly what is saved.
- `jac_probe_size: INT` (default: `1`): number of examples processed per batch
  while computing the Jacobian-drift metric; must be at least one.
- `track_every: INT` (default: `10`): record metrics at the first update and
  then every specified number of updates.
- `print_every: INT` (default: `100`): print recorded loss and accuracies at
  the same cadence convention.
- `checkpoint_state: {metrics_only|resumable_state}` (default: `metrics_only`):
  save histories only, or also save model and random-number-generator state for
  an exact continuation.

### Run fields

- `ckpt_dir: PATH` (required): directory for locally generated checkpoints.
- `save_ckpt: {true|false}` (default: `false`): write a checkpoint at the end
  of the run.
- `load_ckpt: {true|false}` and `load_ckpt_name: PATH` (optional): load an
  existing checkpoint. Relative names are resolved under `ckpt_dir`.
- `resume_from_ckpt: {true|false}` (default: `false`): continue a loaded
  `resumable_state` checkpoint. It requires `load_ckpt: true`.
- `new_total_epochs: INT` (optional): final update count for a resumed run; it
  must be greater than the checkpoint's existing final update.
- `config_overrides: [FIELD, ...]` (optional): fields allowed to differ from a
  resumed checkpoint. Supported overrides are the step-size, regularization,
  noise, early-stopping, Jacobian-batch-size, device, and print-cadence fields;
  this option is intended for advanced continuation runs.

## Merging checkpoints

Pass the checkpoints to the merge script, which verifies that their 
configurations agree, combines the seed-level histories, and writes a 
single plotting checkpoint.

```bash
python scripts/merge_metric_checkpoints.py --out cld_checkpoints/shallow/merged_metrics.pt <seed-0-checkpoint> <seed-1-checkpoint> <seed-2-checkpoint> <seed-3-checkpoint> <seed-4-checkpoint>
```

## Generating plots

Generate the aggregate Jacobian-drift and training-loss plots from the merged
checkpoint:

```bash
python scripts/plot_checkpoint.py cld_checkpoints/shallow/merged_metrics.pt --outdir plots/shallow
```

To reproduce the shallow experiment's per-seed, per-temperature feature-Gram
plot in Figure 1c, run:

```bash
python scripts/plot_metric_per_seed.py cld_checkpoints/shallow/merged_metrics.pt --metric feat_gram_lambda_hist --outdir plots/shallow
```

Use `--help` with any script for its available options. Checkpoints are Python
serialization files; only load files generated by you or obtained from a
trusted source.
