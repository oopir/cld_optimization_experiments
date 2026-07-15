#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from numbers import Number
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit(
        "Run this script from the repository root so relative paths, configs, "
        f"and imports resolve consistently:\n  cd {REPO_ROOT}\n"
        "  python scripts/run_init_scale.py --config <path>"
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

from src.init_scale import InitScaleConfig, plot_from_rows, run_experiment

SHORT_LIST_LIMIT = 12
INIT_SEED_LIST_LIMIT = 8
PREVIEW_COUNT = 5
METRIC_NAME_LIMIT = 10


def compact_config(value, field_name=None):
    """Convert config objects into compact, stable JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): compact_config(val, field_name=str(key)) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return _compact_list(list(value), field_name=field_name)
    return value


def dump_compact_json(value, indent=2, level=0):
    """Dump JSON with dictionaries indented but lists kept on one line."""
    if isinstance(value, dict):
        if not value:
            return "{}"
        pad = " " * (indent * level)
        child_pad = " " * (indent * (level + 1))
        lines = ["{"]
        for idx, key in enumerate(sorted(value)):
            suffix = "," if idx < len(value) - 1 else ""
            rendered_key = json.dumps(str(key))
            rendered_value = dump_compact_json(value[key], indent=indent, level=level + 1)
            lines.append(f"{child_pad}{rendered_key}: {rendered_value}{suffix}")
        lines.append(f"{pad}}}")
        return "\n".join(lines)
    if isinstance(value, list):
        return json.dumps(value)
    return json.dumps(value)


def _compact_list(values, field_name=None):
    """Summarize long scalar lists while leaving short, readable lists expanded."""
    if field_name in {"negative_classes", "positive_classes"}:
        return [compact_config(value) for value in values]
    if field_name in {"tracked_metrics", "plot_metrics"}:
        return _compact_name_list(values)
    if field_name == "init_seeds" and len(values) > INIT_SEED_LIST_LIMIT:
        return _summarize_scalar_list(values)
    if _is_scalar_list(values) and len(values) > SHORT_LIST_LIMIT:
        return _summarize_scalar_list(values)
    return [compact_config(value) for value in values]


def _compact_name_list(values):
    """Compact metric-name lists, which are usually useful by count first."""
    if len(values) <= METRIC_NAME_LIMIT:
        return {"count": len(values), "names": list(values)}
    return {
        "count": len(values),
        "first": list(values[:PREVIEW_COUNT]),
        "last": list(values[-PREVIEW_COUNT:]),
    }


def _summarize_scalar_list(values):
    """Summarize a long scalar list as a range when possible, otherwise previews."""
    summary = {"count": len(values)}
    if not values:
        return summary
    if _is_numeric_list(values):
        step = _constant_step(values)
        if step is not None:
            summary["range"] = f"{_format_scalar(values[0])}..{_format_scalar(values[-1])}"
            summary["step"] = step
            return summary
        summary["min"] = min(values)
        summary["max"] = max(values)
    summary["first"] = list(values[:PREVIEW_COUNT])
    summary["last"] = list(values[-PREVIEW_COUNT:])
    return summary


def _is_scalar_list(values):
    """Return True for JSON scalar lists that can be summarized safely."""
    return all(value is None or isinstance(value, (str, Number)) for value in values)


def _is_numeric_list(values):
    """Return True for numeric lists, excluding bools despite bool being int-like."""
    return bool(values) and all(isinstance(value, Number) and not isinstance(value, bool) for value in values)


def _constant_step(values):
    """Return the arithmetic step if values form an exact/simple numeric range."""
    if len(values) < 2 or not _is_numeric_list(values):
        return None
    step = values[1] - values[0]
    for prev, curr in zip(values, values[1:]):
        delta = curr - prev
        if isinstance(step, float) or isinstance(delta, float):
            if abs(delta - step) > 1e-12:
                return None
        elif delta != step:
            return None
    return step


def _format_scalar(value):
    """Format scalar values compactly for range strings."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def load_mapping(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        with path.open("r") as f:
            return yaml.safe_load(f)
    with path.open("r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the initialization-scale experiment.")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML/JSON experiment config.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override experiment.output_dir from the config.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override experiment.device from the config, e.g. cpu, cuda, cuda:0, auto.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate summaries and plots from a saved raw rows CSV without recomputing rows.",
    )
    p.add_argument(
        "--rows-csv",
        type=Path,
        default=None,
        help="Raw rows CSV for --plot-only. Defaults to experiment.output_dir/_init_scale_rows.csv.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.config)
    if "probe" in mapping:
        raise ValueError("Config key 'probe' was renamed to 'experiment'. Please update the YAML/JSON file.")
    config_kwargs = dict(mapping.get("experiment", mapping))
    config = InitScaleConfig(**config_kwargs)

    if args.output_dir is not None:
        config.output_dir = args.output_dir.expanduser()
    if args.device is not None:
        config.device = args.device

    print("effective config:")
    print(dump_compact_json(compact_config(asdict(config))))

    if args.plot_only:
        rows, summary_rows, paths = plot_from_rows(config, rows_path=args.rows_csv)
    else:
        rows, summary_rows, paths = run_experiment(config)
    print(f"raw rows: {len(rows)}")
    print(f"summary rows: {len(summary_rows)}")
    print(f"rows CSV: {paths['rows']}")
    plot_count = len([key for key in paths if key.startswith("plot_")])
    print(f"plots: {plot_count}")


if __name__ == "__main__":
    print(datetime.now())
    main()
    print(datetime.now())
