from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

from sharded.checkpoint import load_checkpoint_with_metadata, load_state_shard
from sharded.config import ShardedExpConfig, ShardedRunOpts
from sharded.distributed import DistContext
from sharded.exp import run_exp


def _cpu_ctx() -> DistContext:
    return DistContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), backend="gloo")


def _tiny_config(epochs: int, beta: float) -> ShardedExpConfig:
    return ShardedExpConfig(
        seeds=[0],
        device="cpu",
        dataset="digits",
        n=20,
        m=6,
        L=2,
        epochs=epochs,
        eta=1.0e-5,
        betas=[beta],
        use_linearized=True,
        same_noise=True,
        tracked_metrics=["train_loss", "lin_train_loss", "lin_param_dist", "nn_lin_param_dist"],
        track_every=1,
        print_every=1000,
        checkpoint_state="sharded_state",
    )


def _run_saved(tmp_path: Path, cfg: ShardedExpConfig):
    ctx = _cpu_ctx()
    run_opts = ShardedRunOpts(ckpt_dir=tmp_path, save_ckpt=True)
    results, ckpt_path = run_exp(cfg, run_opts, ctx)
    assert ckpt_path is not None
    return results, ckpt_path


def _resume_saved(tmp_path: Path, cfg: ShardedExpConfig, ckpt_path: Path, new_total_epochs: int):
    ctx = _cpu_ctx()
    run_opts = ShardedRunOpts(
        ckpt_dir=tmp_path,
        save_ckpt=True,
        load_ckpt=True,
        load_ckpt_name=ckpt_path.name,
        resume_from_ckpt=True,
        new_total_epochs=new_total_epochs,
        config_overrides=[],
    )
    results, resumed_path = run_exp(cfg, run_opts, ctx)
    assert resumed_path is not None
    return results, resumed_path


def _single_state(ckpt_path: Path):
    loaded = load_checkpoint_with_metadata(ckpt_path)
    label = next(iter(loaded.results))
    metrics = loaded.results[label][0]
    state = load_state_shard(loaded.path, metrics, _cpu_ctx())
    return loaded.results, state


def _assert_rng_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy"][0] == right["numpy"][0]
    assert np.array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])
    assert torch.equal(left["noise_gen"], right["noise_gen"])


@pytest.mark.parametrize("beta", [float("inf"), 10.0])
def test_sharded_resume_matches_continuous_run(tmp_path, beta):
    continuous_cfg = _tiny_config(epochs=4, beta=beta)
    _, continuous_ckpt = _run_saved(tmp_path / "continuous", continuous_cfg)

    split_cfg = _tiny_config(epochs=2, beta=beta)
    _, split_ckpt = _run_saved(tmp_path / "split", split_cfg)
    _, resumed_ckpt = _resume_saved(tmp_path / "split", split_cfg, split_ckpt, new_total_epochs=4)

    continuous_results, continuous_state = _single_state(continuous_ckpt)
    resumed_results, resumed_state = _single_state(resumed_ckpt)
    label = next(iter(continuous_results))

    assert continuous_results[label][0]["epoch_hist"] == resumed_results[label][0]["epoch_hist"]
    assert continuous_results[label][0]["train_loss_hist"] == resumed_results[label][0]["train_loss_hist"]
    assert continuous_results[label][0]["lin_train_loss_hist"] == resumed_results[label][0]["lin_train_loss_hist"]

    assert set(continuous_state.state_tensors) == set(resumed_state.state_tensors)
    for name, tensor in continuous_state.state_tensors.items():
        assert torch.equal(tensor.cpu(), resumed_state.state_tensors[name].cpu())
    _assert_rng_equal(continuous_state.rng_state, resumed_state.rng_state)


def test_resume_requires_new_total_epochs(tmp_path):
    cfg = _tiny_config(epochs=1, beta=float("inf"))
    _, ckpt_path = _run_saved(tmp_path / "ckpts", cfg)
    run_opts = ShardedRunOpts(
        ckpt_dir=tmp_path / "ckpts",
        load_ckpt=True,
        load_ckpt_name=ckpt_path.name,
        resume_from_ckpt=True,
    )
    with pytest.raises(ValueError, match="new_total_epochs"):
        run_exp(cfg, run_opts, _cpu_ctx())


def test_resume_rejects_tracked_metric_override(tmp_path):
    cfg = _tiny_config(epochs=1, beta=float("inf"))
    _, ckpt_path = _run_saved(tmp_path / "ckpts", cfg)
    changed_cfg = _tiny_config(epochs=1, beta=float("inf"))
    changed_cfg.tracked_metrics = ["test_acc"]
    run_opts = ShardedRunOpts(
        ckpt_dir=tmp_path / "ckpts",
        load_ckpt=True,
        load_ckpt_name=ckpt_path.name,
        resume_from_ckpt=True,
        new_total_epochs=2,
        config_overrides=["tracked_metrics"],
    )
    with pytest.raises(ValueError, match="Unsupported resume config override"):
        run_exp(changed_cfg, run_opts, _cpu_ctx())


def test_resume_rejects_missing_rng_state(tmp_path):
    cfg = _tiny_config(epochs=1, beta=10.0)
    _, ckpt_path = _run_saved(tmp_path / "ckpts", cfg)
    loaded = load_checkpoint_with_metadata(ckpt_path)
    label = next(iter(loaded.results))
    shard_dir = loaded.path / loaded.results[label][0]["state_shard_dir"] / "rank_000"
    manifest_path = shard_dir / "manifest.pt"
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    manifest.pop("rng_file")
    torch.save(manifest, manifest_path)

    run_opts = ShardedRunOpts(
        ckpt_dir=tmp_path / "ckpts",
        load_ckpt=True,
        load_ckpt_name=ckpt_path.name,
        resume_from_ckpt=True,
        new_total_epochs=2,
        config_overrides=[],
    )
    with pytest.raises(ValueError, match="no RNG state"):
        run_exp(cfg, run_opts, _cpu_ctx())
