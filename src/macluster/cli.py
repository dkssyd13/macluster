"""Command-line entry point: ``macluster {train,sweep,plot} ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .train import TrainConfig, run_training


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", default="cifar10")
    p.add_argument("--model", default="resnet20")
    p.add_argument("--algorithm", default="diloco",
                   choices=["dense", "diloco", "sparseloco", "adaptive"])
    p.add_argument("--world-size", type=int, default=2, dest="world_size")
    p.add_argument("--backend", default="sim", choices=["sim", "grove"],
                   help="sim = single-machine emulation; grove = real 2-MacBook cluster")
    p.add_argument("--parallelism", default="data", choices=["data", "pipeline"],
                   help="data = DiLoCo/etc. replicas; pipeline = model-parallel GPT split (Phase 8)")
    p.add_argument("--cut", default=None,
                   help="pipeline: explicit block-cut indices, e.g. '22' or '8,16' (else auto from --stage-mem-gb)")
    p.add_argument("--n-micro", type=int, default=4, dest="n_micro",
                   help="pipeline: micro-batches per optimizer step (1F1B)")
    p.add_argument("--stage-mem-gb", default=None, dest="stage_mem_gb",
                   help="pipeline: per-stage RAM budget for the memory-aware auto cut, e.g. '48,24'")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=None, dest="max_steps",
                   help="equal total-local-step budget across algorithms (overrides rounds)")
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p.add_argument("--seq-len", type=int, default=128, dest="seq_len")
    p.add_argument("--H", type=int, default=20)
    p.add_argument("--k-frac", type=float, default=0.02, dest="k_frac")
    p.add_argument("--outer-lr", type=float, default=0.7, dest="outer_lr")
    p.add_argument("--inner-opt", default="adam", choices=["adam", "sgd"], dest="inner_opt")
    p.add_argument("--inner-lr", type=float, default=1e-3, dest="inner_lr")
    p.add_argument("--link", default="wifi",
                   choices=["wifi", "awdl", "wifi_degraded", "datacenter"])
    p.add_argument("--link-switch-to", default=None, dest="link_switch_to",
                   choices=["wifi", "awdl", "wifi_degraded", "datacenter"])
    p.add_argument("--link-switch-at", type=int, default=None, dest="link_switch_at")
    p.add_argument("--eval-every", type=int, default=5, dest="eval_every")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--data-dir", default="data/cache", dest="data_dir")
    p.add_argument("--target-metric", type=float, default=None, dest="target_metric")
    p.add_argument("--run-dir", default=None, dest="run_dir")
    p.add_argument("--runs-root", default="runs", dest="runs_root")


def _cfg_from_args(args) -> TrainConfig:
    fields = {f for f in TrainConfig.__dataclass_fields__}
    return TrainConfig(**{k: v for k, v in vars(args).items() if k in fields})


def cmd_train(args) -> None:
    cfg = _cfg_from_args(args)
    run_dir = args.run_dir or os.path.join(args.runs_root, cfg.slug())
    print(f"[macluster] training -> {run_dir}")
    summary = run_training(cfg, run_dir)
    print(json.dumps(summary, indent=2, default=str))


def cmd_sweep(args) -> None:
    from .experiment import run_sweep

    run_sweep(args.config, runs_root=args.runs_root, overwrite=args.overwrite)


def run_plot(plot_args: list[str]) -> int:
    # The plotting code lives in scripts/plot.py (standalone, matplotlib-Agg);
    # we load it by path and forward the args to its main().
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "plot.py"
    if not script.exists():
        raise SystemExit(f"plot script not found: {script}")
    spec = importlib.util.spec_from_file_location("macluster_plot_script", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(plot_args)


def main(argv=None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # Intercept `plot` before argparse: its args (--runs, --sweep, ...) are
    # forwarded verbatim to scripts/plot.py and must not be parsed here
    # (argparse.REMAINDER mishandles a leading optional like --runs).
    if argv and argv[0] == "plot":
        raise SystemExit(run_plot(argv[1:]))

    parser = argparse.ArgumentParser(prog="macluster", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", help="run one training configuration")
    _add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_sweep = sub.add_parser("sweep", help="run a TrainConfig sweep from a YAML")
    p_sweep.add_argument("--config", required=True, help="path to sweep YAML")
    p_sweep.add_argument("--runs-root", default="runs", dest="runs_root")
    p_sweep.add_argument("--overwrite", action="store_true")
    p_sweep.set_defaults(func=cmd_sweep)

    # Listed for `--help`; actually handled by the early intercept above.
    sub.add_parser("plot", help="plot run metrics (forwards args to scripts/plot.py)")

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
