"""Experiment sweep harness.

Reads a small YAML describing a *sweep*::

    name: axis2_algorithms
    base:                 # TrainConfig field overrides shared by every run
      task: cifar10
      rounds: 25
    grid:                 # field -> list of values; cartesian product is expanded
      algorithm: [dense, diloco, sparseloco]
      link: [wifi, awdl]

and runs ``run_training`` once per point of the cartesian product, collecting the
per-run ``summary`` dicts and writing ``{runs_root}/{name}/index.json``.

Runs are *resumable*: a config whose ``summary.json`` already exists is skipped
(its summary is re-read from disk) unless ``overwrite`` is set.

Run as a module::

    uv run python -m macluster.experiment --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys

import yaml

from .train import TrainConfig, run_training

# Fields the harness understands -> used to validate base/grid keys early so a
# typo in a YAML fails loudly instead of being silently ignored.
_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


# --------------------------------------------------------------------------- #
# YAML -> [TrainConfig]
# --------------------------------------------------------------------------- #
def _check_fields(where: str, keys) -> None:
    bad = [k for k in keys if k not in _FIELDS]
    if bad:
        raise KeyError(
            f"{where} references unknown TrainConfig field(s) {bad!r}; "
            f"valid fields: {sorted(_FIELDS)}"
        )


def load_sweep(config_path: str) -> tuple[str, list[TrainConfig]]:
    """Load a sweep YAML and expand it into ``(name, [TrainConfig, ...])``."""
    with open(config_path) as f:
        spec = yaml.safe_load(f) or {}

    name = spec.get("name")
    if not name:
        # fall back to the file stem so an unnamed sweep still has a home
        name = os.path.splitext(os.path.basename(config_path))[0]

    base = spec.get("base") or {}
    grid = spec.get("grid") or {}
    if not isinstance(base, dict):
        raise TypeError(f"'base' must be a mapping, got {type(base).__name__}")
    if not isinstance(grid, dict):
        raise TypeError(f"'grid' must be a mapping, got {type(grid).__name__}")

    _check_fields("base", base.keys())
    _check_fields("grid", grid.keys())

    grid_keys = list(grid.keys())
    # normalise each grid value to a list so a scalar grid entry still works
    grid_values = [v if isinstance(v, (list, tuple)) else [v] for v in grid.values()]

    configs: list[TrainConfig] = []
    combos = itertools.product(*grid_values) if grid_keys else [()]
    for combo in combos:
        overrides = dict(base)
        overrides.update(dict(zip(grid_keys, combo)))
        configs.append(TrainConfig(**overrides))
    return name, configs


# --------------------------------------------------------------------------- #
# sweep runner
# --------------------------------------------------------------------------- #
def run_sweep(config_path: str, runs_root: str = "runs", overwrite: bool = False) -> dict:
    name, configs = load_sweep(config_path)
    sweep_dir = os.path.join(runs_root, name)
    os.makedirs(sweep_dir, exist_ok=True)

    n = len(configs)
    print(f"[sweep] {name}: {n} run(s) -> {sweep_dir}", flush=True)

    results: list[dict] = []
    for i, cfg in enumerate(configs, start=1):
        run_dir = os.path.join(sweep_dir, cfg.slug())
        summary_path = os.path.join(run_dir, "summary.json")

        if not overwrite and os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"[{i}/{n}] SKIP (exists) {cfg.slug()}", flush=True)
            results.append(summary)
            continue

        print(f"[{i}/{n}] RUN  {cfg.slug()}", flush=True)
        summary = run_training(cfg, run_dir)
        metric_key = next(
            (k for k in summary if k.startswith("final_") and k != "final_train_loss"),
            None,
        )
        metric_str = f" {metric_key}={summary[metric_key]}" if metric_key else ""
        print(
            f"[{i}/{n}] DONE {cfg.slug()}"
            f"{metric_str} sim_time_s={summary.get('sim_time_s')}",
            flush=True,
        )
        results.append(summary)

    index = {"name": name, "n": len(results), "results": results}
    index_path = os.path.join(sweep_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, default=str)
    print(f"[sweep] {name}: wrote {index_path} ({len(results)} results)", flush=True)
    return index


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(
        prog="macluster.experiment",
        description="Run a TrainConfig sweep described by a YAML file.",
    )
    parser.add_argument("--config", required=True, help="path to sweep YAML")
    parser.add_argument("--runs-root", default="runs", help="root dir for run outputs")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run configs even if a summary.json already exists",
    )
    args = parser.parse_args(argv)
    return run_sweep(args.config, runs_root=args.runs_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main(sys.argv[1:])
