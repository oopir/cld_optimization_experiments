#!/usr/bin/env python3
"""
Migrate old Exp1 checkpoint result labels from scientific beta notation to the
new beta-as-multiple-of-n convention.

Examples:
  β=1.00e+02        -> β=10n      when n=10
  β=5.00e+02        -> β=50n      when n=10
  β=1.00e+03        -> β=100n     when n=10
  β=inf             -> inf
  α=1e+00 β=1.00e+03 -> α=1e+00 β=100n
  α=1e+00 β=inf      -> α=1e+00 inf
"""


from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
import argparse
import math
import re
import shutil
from pathlib import Path
from typing import Any

import torch

# Importing this helps torch unpickle old Exp1Config objects when the script is
# run from the updated repo root.
try:
    import src.metric_checkpoints  # noqa: F401
except Exception:
    pass


BETA_RE = re.compile(r"β=([+-]?(?:inf|nan|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?))")


def get_config_n(config: Any) -> int | None:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("n")
    return getattr(config, "n", None)


def parse_beta(beta_text: str) -> float:
    if beta_text.lower() == "inf":
        return math.inf
    return float(beta_text)


def format_new_beta_label(beta: float, n: int | None) -> str:
    if math.isinf(beta):
        return "inf"
    if n is None:
        return f"β={int(beta)}"
    return f"β={int(beta // n)}n"


def migrate_label(label: str, n: int | None) -> str:
    """
    Replace the beta component in a label while leaving any alpha prefix intact.
    Labels without a beta token are returned unchanged.
    """

    match = BETA_RE.search(label)
    if not match:
        return label

    beta = parse_beta(match.group(1))
    new_beta = format_new_beta_label(beta, n)

    # Preserve everything before/after the old beta token. This keeps labels like
    # "α=1e+00 β=1.00e+03" as "α=1e+00 β=100n".
    new_label = label[: match.start()] + new_beta + label[match.end() :]

    # Clean up accidental double spaces.
    return " ".join(new_label.split())


def migrate_checkpoint(input_path: Path, output_path: Path, n_override: int | None, dry_run: bool) -> None:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint payload to be a dict, got {type(payload).__name__}")

    if "results" not in payload:
        raise KeyError("Checkpoint payload has no 'results' key")

    results = payload["results"]
    if not isinstance(results, dict):
        raise TypeError(f"Expected payload['results'] to be a dict, got {type(results).__name__}")

    n = n_override if n_override is not None else get_config_n(payload.get("config"))
    if n is None:
        raise ValueError("Could not infer n from payload['config']; pass --n explicitly")

    migrated_results = {}
    mapping = {}

    for old_label, value in results.items():
        new_label = migrate_label(str(old_label), n)
        mapping[str(old_label)] = new_label

        if new_label in migrated_results and new_label != old_label:
            raise ValueError(
                f"Label collision while migrating: {old_label!r} maps to {new_label!r}, "
                "but that key already exists. Inspect the checkpoint manually."
            )

        migrated_results[new_label] = value

    print(f"Using n={n}")
    print("Label mapping:")
    for old, new in mapping.items():
        marker = "changed" if old != new else "unchanged"
        print(f"  [{marker}] {old!r} -> {new!r}")

    if dry_run:
        print("\nDry run only; no file written.")
        return

    payload["results"] = migrated_results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"\nWrote migrated checkpoint to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Input .pt checkpoint")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output checkpoint. Defaults to '<input>.migrated.pt'",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input checkpoint after making a .bak copy",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Dataset sample size. Overrides config.n from the checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the label mapping without writing anything.",
    )
    args = parser.parse_args()

    input_path = args.checkpoint.expanduser().resolve()

    if args.in_place and args.output is not None:
        raise ValueError("Use either --in-place or --output, not both")

    if args.in_place:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        if not args.dry_run:
            shutil.copy2(input_path, backup_path)
            print(f"Backup written to: {backup_path}")
        output_path = input_path
    else:
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else input_path.with_suffix(".migrated.pt")
        )

    migrate_checkpoint(
        input_path=input_path,
        output_path=output_path,
        n_override=args.n,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()