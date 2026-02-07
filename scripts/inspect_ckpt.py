# example run: PYTHONPATH=. python3 scripts/inspect_ckpt.py ~/cld_checkpoints/expr1/exp1_digits_20260127_135649.pt

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

def _load_ckpt(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: top-level object is not a dict")
    return obj


def _config_to_dict(cfg: Any) -> Dict[str, Any]:
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def _infer_epochs(metrics: Dict[str, Any]) -> int | None:
    for k, v in metrics.items():
        if not k.endswith("_hist"):
            continue
        if v is None:
            continue
        try:
            return len(v)
        except TypeError:
            continue
    return None


def _check_results_consistency(
    results: Dict[Any, Any], 
    beta_labels: List[Any]
) -> Tuple[bool, Optional[List[Any]], Optional[int], Optional[Set[str]]]:
    if not beta_labels:
        return True, None, None, None

    ref_beta = beta_labels[0]
    
    # --------- save the first run's seeds, epochs and metric keys as reference points --------- #
    ref_seeds_dict = results[ref_beta]
    if not isinstance(ref_seeds_dict, dict):
        print(f"[consistency] {ref_beta}: expected dict seed→metrics, got {type(ref_seeds_dict)}")
        return False, None, None, None
    ref_seed_ids = sorted(ref_seeds_dict.keys())
    
    ref_example_metrics = None
    for m in ref_seeds_dict.values():
        if isinstance(m, dict):
            ref_example_metrics = m
            break
    if ref_example_metrics is None:
        print(f"[consistency] {ref_beta}: no metrics dicts found for seeds")
        return False, None, None, None
    
    ref_epochs = _infer_epochs(ref_example_metrics)
    
    ref_metric_keys = set(ref_example_metrics.keys())

    # ------------- check that other runs are consistent with the reference points ------------- #
    for beta_label in beta_labels[1:]:
        # get seed data
        seeds_dict = results[beta_label]
        if not isinstance(seeds_dict, dict):
            print(f"[consistency] {beta_label}: expected dict seed→metrics, got {type(seeds_dict)}")
            return False, None, None, None
        seed_ids = sorted(seeds_dict.keys())

        # get metrics data
        example_metrics = None
        for m in seeds_dict.values():
            if isinstance(m, dict):
                example_metrics = m
                break
        if example_metrics is None:
            print(f"[consistency] no metrics dicts found for beta={beta_label}")
            return False, None, None, None

        epochs = _infer_epochs(example_metrics)
        metric_keys = set(example_metrics.keys())

        # test consistency
        if seed_ids != ref_seed_ids:
            print(f"[consistency] seed mismatch for beta={beta_label}: {seed_ids} != {ref_seed_ids}")
            return False, None, None, None
        if epochs != ref_epochs:
            print(f"[consistency] epoch mismatch for beta={beta_label}: {epochs} != {ref_epochs}")
            return False, None, None, None
        if metric_keys != ref_metric_keys:
            only_ref = sorted(ref_metric_keys - metric_keys)
            only_this = sorted(metric_keys - ref_metric_keys)
            print(f"[consistency] metric key mismatch for beta={beta_label}:")
            print(f"  only in {ref_beta}: {only_ref}")
            print(f"  only in {beta_label}: {only_this}")
            return False, None, None, None

    return True, ref_seed_ids, ref_epochs, ref_metric_keys


def inspect_checkpoint(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path} (or maybe you forgot to run the script from project root with PYTHONPATH=.)")

    stat = path.stat()
    print(f"\n===== {path} =====")
    print("\n[general]")
    print(f"size: {stat.st_size} bytes")
    print(f"modified: {datetime.fromtimestamp(stat.st_mtime).isoformat(sep=' ', timespec='seconds')}")

    payload = _load_ckpt(path)
    ckpt_type = payload.get("type", "<missing>")
    print(f"ckpt type: {ckpt_type}")

    # validate 'config' and 'results'
    config = payload.get("config", None)
    results = payload.get("results", None)
    if config is None or results is None:
        print("Missing 'config' or 'results' in payload; nothing more to show.")
        return
    if results is not None and not isinstance(results, dict):
        print("'results' is not a dict.")
        return

    # print config
    print("\n[config]")
    cfg_dict = _config_to_dict(config)
    for k in cfg_dict:
        print(f"{k}: {cfg_dict[k]}")

    # print run labels
    print("\n[results]")
    beta_labels = sorted(results.keys())
    if not beta_labels:
        return
    print(f"beta labels: {', '.join(str(b) for b in beta_labels)}")

    # validate consistency of runs across seeds, epochs, metrics
    ok, _, _, ref_metric_keys = _check_results_consistency(results, beta_labels)
    if not ok:
        print("consistency check for seeds/epochs/metric keys: failed")
        return
    print("consistency check for seeds/epochs/metric keys: passed")

    # print metrics
    print("metric keys:")
    if ref_metric_keys is None:
        return
    hist_metric_names = sorted(k for k in ref_metric_keys if k.endswith("_hist"))
    print("  hist metrics:")
    for name in hist_metric_names:
        print(f"    {name}")
    non_hist_keys = sorted(k for k in ref_metric_keys if not k.endswith("_hist"))
    print("  non-hist metrics:")
    for name in non_hist_keys:
        print(f"    {name}")


def diff_configs(cfg_a: Any, cfg_b: Any) -> None:
    d_a = _config_to_dict(cfg_a)
    d_b = _config_to_dict(cfg_b)
    keys = sorted(set(d_a) | set(d_b))

    print("[config differences]")
    for k in keys:
        v_a = d_a.get(k, "<missing>")
        v_b = d_b.get(k, "<missing>")
        if v_a == v_b:
            continue
        print(f"  {k}:")
        print(f"    A: {v_a}")
        print(f"    B: {v_b}")
    print()


def diff_results(res_a: Dict[str, Any], res_b: Dict[str, Any]) -> None:
    if not isinstance(res_a, dict) or not isinstance(res_b, dict):
        print("[results] one of the results objects is not a dict; skipping.")
        return

    betas_a, betas_b = sorted(res_a.keys()), sorted(res_b.keys())

    print("[beta labels]")
    print(f"  only in A: {sorted(set(betas_a) - set(betas_b))}")
    print(f"  only in B: {sorted(set(betas_b) - set(betas_a))}")
    print(f"  in both : {sorted(set(betas_a) & set(betas_b))}")
    print()

    print("[per-checkpoint consistency]")
    ok_a, seeds_a, _, keys_a = _check_results_consistency(res_a, betas_a)
    print(f"  A: seeds/epochs/metric keys identical across betas: {'YES' if ok_a else 'NO'}")
    ok_b, seeds_b, _, keys_b = _check_results_consistency(res_b, betas_b)
    print(f"  B: seeds/epochs/metric keys identical across betas: {'YES' if ok_b else 'NO'}")
    if not ok_a or not ok_b:
        print("Inconsistent structure within at least one checkpoint; aborting detailed diff.")
        return
    print()

    print("[global seeds]")
    if seeds_a is not None and seeds_b is not None:
        print(f"  seeds only in A: {sorted(set(seeds_a) - set(seeds_b))}")
        print(f"  seeds only in B: {sorted(set(seeds_b) - set(seeds_a))}")
        print(f"  seeds in both : {sorted(set(seeds_a) & set(seeds_b))}")
    else:
        print("  seeds: unknown (no seeds detected)")
    print("[global metric keys]")
    if keys_a is not None and keys_b is not None:
        print(f"  metric keys only in A: {sorted(keys_a - keys_b)}")
        print(f"  metric keys only in B: {sorted(keys_b - keys_a)}")
        print(f"  metric keys in both : {sorted(keys_a & keys_b)}")
    else:
        print("  metric keys: unknown")
    print()

    # from each ckpt, get a metrics dict from one shared (beta,seed)
    common_betas = sorted(set(betas_a) & set(betas_b))
    if not common_betas or seeds_a is None or seeds_b is None:
        return
    rep_beta = common_betas[0]
    common_seeds = sorted(set(seeds_a) & set(seeds_b))
    if not common_seeds:
        return
    rep_seed = common_seeds[0]
    m_a = res_a[rep_beta][rep_seed]
    m_b = res_b[rep_beta][rep_seed]

    # check for histogram length differences
    keys_common = sorted(set(m_a.keys()) & set(m_b.keys()))
    print("[histogram length differences (one shared beta/seed)]")
    print(f"  representative beta: {rep_beta}")
    print(f"  representative seed: {rep_seed}")
    for k in keys_common:
        if not k.endswith("_hist"):
            continue
        v_a, v_b = m_a.get(k), m_b.get(k)
        try:
            len_a, len_b = len(v_a), len(v_b)
        except TypeError:
            continue
        if len_a != len_b:
            print(f"  {k}: len A={len_a}, len B={len_b}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or diff exp1-style checkpoints.")
    parser.add_argument("paths", nargs="+", help="Checkpoint path(s). 1+ for inspect, 2 with --diff for diff.")
    parser.add_argument("--diff", action="store_true", help="Diff two checkpoints instead of inspecting.")
    args = parser.parse_args()

    if args.diff:
        # diff between ckpts
        if len(args.paths) != 2:
            raise SystemExit("In diff mode you must pass exactly two paths.")
        
        path_a, path_b = Path(args.paths[0]).expanduser().resolve(), Path(args.paths[1]).expanduser().resolve()
        ckpt_a, ckpt_b = _load_ckpt(path_a), _load_ckpt(path_b)
        
        type_a, type_b = ckpt_a.get("type", "<missing>"), ckpt_b.get("type", "<missing>")
        print(f"type A: {type_a}")
        print(f"type B: {type_b}")
        
        cfg_a, cfg_b = ckpt_a.get("config", None), ckpt_b.get("config", None)
        if cfg_a is not None and cfg_b is not None:
            diff_configs(cfg_a, cfg_b)
        else:
            print("Missing config in one of the checkpoints; skipping config diff.\n")

        res_a, res_b = ckpt_a.get("results", None), ckpt_b.get("results", None) 
        if res_a is not None and res_b is not None:
            diff_results(res_a, res_b)
        else:
            print("Missing results in one of the checkpoints; skipping results diff.\n")
    else:
        # check each ckpt w/o comparing between them
        for p in args.paths:
            inspect_checkpoint(Path(p))

if __name__ == "__main__":
    main()
