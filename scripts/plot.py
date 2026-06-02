#!/usr/bin/env python3
"""Plotting / analysis for macluster runs (Phase 6).

Reads the per-run metrics schema produced by ``run_training``:

  metrics.jsonl  -- one JSON object per round with keys including
                    round, train_loss, sim_time_s, comm_bytes_cum, H,
                    k_frac (sparse/adaptive); on eval rounds also val_loss
                    and (accuracy for cifar / perplexity for text).
  summary.json   -- final_*, total_comm_MB, sim_time_s, time_to_target_s,
                    config (dict).

Produces (Agg backend, no display):
  fig 1  primary metric (val accuracy if present else train_loss) vs
         sim_time_s, overlaying the given runs (time-to-accuracy story).
  fig 2  comm_bytes_cum vs round per run (log y) -- compression savings.
  fig 3  (--sweep) final metric and total_comm_MB across the swept axis.
  fig 4  for an adaptive run: H and k_frac over rounds on twin y-axes.

Usage:
  uv run python scripts/plot.py --runs runs/real_diloco runs/smoke_sparseloco \
      runs/smoke_adaptive --out figures/ [--sweep runs/<name>/index.json]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_metrics(run_dir: str) -> list[dict]:
    """Load metrics.jsonl as a list of dicts (one per round)."""
    path = os.path.join(run_dir, "metrics.jsonl")
    records: list[dict] = []
    if not os.path.exists(path):
        return records
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_summary(run_dir: str) -> dict:
    """Load summary.json (empty dict if missing)."""
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def run_label(run_dir: str) -> str:
    """A short, human-readable label for a run directory."""
    base = os.path.basename(os.path.normpath(run_dir))
    summary = load_summary(run_dir)
    cfg = summary.get("config", {}) or {}
    algo = cfg.get("algorithm")
    if algo:
        return f"{base} ({algo})"
    return base


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _eval_metric_key(records: list[dict]) -> Optional[str]:
    """Pick the primary *eval* metric key present in the records.

    Preference order: accuracy (cifar), perplexity (text), val_loss.
    Returns None if no eval rounds are present.
    """
    for key in ("accuracy", "perplexity", "val_loss"):
        if any(key in r and r[key] is not None for r in records):
            return key
    return None


def primary_series(records: list[dict]) -> tuple[str, list[float], list[float]]:
    """Return (metric_name, x=sim_time_s, y=metric) for fig 1.

    Uses an eval metric (accuracy / perplexity / val_loss) on the rounds
    where it is present; falls back to train_loss on every round.
    """
    key = _eval_metric_key(records)
    if key is not None:
        xs, ys = [], []
        for r in records:
            if key in r and r[key] is not None and r.get("sim_time_s") is not None:
                xs.append(float(r["sim_time_s"]))
                ys.append(float(r[key]))
        if xs:
            return key, xs, ys
    # fallback: train_loss vs sim_time
    xs, ys = [], []
    for r in records:
        if r.get("train_loss") is not None and r.get("sim_time_s") is not None:
            xs.append(float(r["sim_time_s"]))
            ys.append(float(r["train_loss"]))
    return "train_loss", xs, ys


def _metric_axis_label(name: str) -> str:
    return {
        "accuracy": "validation accuracy",
        "perplexity": "validation perplexity",
        "val_loss": "validation loss",
        "train_loss": "train loss",
    }.get(name, name)


# ---------------------------------------------------------------------------
# Figure 1: primary metric vs sim_time
# ---------------------------------------------------------------------------


def plot_metric_vs_time(run_dirs: list[str], out_path: str) -> Optional[str]:
    """Overlay primary metric vs sim_time_s for the given runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    metric_names: set[str] = set()
    plotted = 0
    for rd in run_dirs:
        recs = load_metrics(rd)
        if not recs:
            continue
        name, xs, ys = primary_series(recs)
        if not xs:
            continue
        metric_names.add(name)
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8, label=run_label(rd))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    # If all runs share one metric, label the axis precisely; else generic.
    if len(metric_names) == 1:
        ylabel = _metric_axis_label(next(iter(metric_names)))
    else:
        ylabel = "primary metric (val if available, else train loss)"
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title("Time-to-accuracy: primary metric vs simulated time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: cumulative comm bytes vs round (log y)
# ---------------------------------------------------------------------------


def plot_comm_vs_round(run_dirs: list[str], out_path: str) -> Optional[str]:
    """Cumulative communication bytes vs round, log-y, per run."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for rd in run_dirs:
        recs = load_metrics(rd)
        if not recs:
            continue
        xs, ys = [], []
        for r in recs:
            if r.get("comm_bytes_cum") is not None and r.get("round") is not None:
                xs.append(int(r["round"]))
                ys.append(float(r["comm_bytes_cum"]))
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8, label=run_label(rd))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative communication (bytes, log scale)")
    ax.set_title("Communication cost: cumulative bytes vs round")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: adaptive H and k_frac over rounds (twin y-axes)
# ---------------------------------------------------------------------------


def find_adaptive_run(run_dirs: list[str]) -> Optional[str]:
    """Return the first run dir whose config.algorithm == 'adaptive',
    else the first run whose metrics vary in H or k_frac."""
    for rd in run_dirs:
        cfg = (load_summary(rd).get("config") or {})
        if cfg.get("algorithm") == "adaptive":
            return rd
    # fallback: a run where H or k_frac actually changes round to round
    for rd in run_dirs:
        recs = load_metrics(rd)
        hs = {r.get("H") for r in recs if r.get("H") is not None}
        ks = {r.get("k_frac") for r in recs if r.get("k_frac") is not None}
        if len(hs) > 1 or len(ks) > 1:
            return rd
    return None


def plot_adaptive_schedule(run_dir: str, out_path: str) -> Optional[str]:
    """H and k_frac over rounds on twin y-axes for an adaptive run."""
    recs = load_metrics(run_dir)
    if not recs:
        return None
    rounds = [int(r["round"]) for r in recs if r.get("round") is not None]
    h_vals = [r.get("H") for r in recs if r.get("round") is not None]
    k_vals = [r.get("k_frac") for r in recs if r.get("round") is not None]

    have_h = any(v is not None for v in h_vals)
    have_k = any(v is not None for v in k_vals)
    if not (have_h or have_k):
        return None

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_h = "tab:blue"
    color_k = "tab:red"

    lines = []
    if have_h:
        (l1,) = ax1.plot(
            rounds, h_vals, marker="o", markersize=4, color=color_h,
            linewidth=1.8, label="H (local steps)",
        )
        lines.append(l1)
        ax1.set_ylabel("H (local steps per round)", color=color_h)
        ax1.tick_params(axis="y", labelcolor=color_h)
    ax1.set_xlabel("round")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if have_k:
        (l2,) = ax2.plot(
            rounds, k_vals, marker="s", markersize=4, color=color_k,
            linewidth=1.8, label="k_frac (sparsity)",
        )
        lines.append(l2)
        ax2.set_ylabel("k_frac (top-k fraction)", color=color_k)
        ax2.tick_params(axis="y", labelcolor=color_k)

    ax1.set_title(f"Adaptive sync schedule over rounds ({run_label(run_dir)})")
    if lines:
        ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: sweep -- final metric & total_comm_MB across the swept axis
# ---------------------------------------------------------------------------


def _varying_config_fields(configs: list[dict]) -> list[str]:
    """Config field names whose value differs across the swept runs."""
    keys: set[str] = set()
    for c in configs:
        keys.update(c.keys())
    varying = [k for k in sorted(keys) if len({str(c.get(k)) for c in configs}) > 1]
    for noise in ("run_dir", "target_metric"):
        if noise in varying:
            varying.remove(noise)
    return varying


def load_sweep(index_path: str) -> tuple[str, list[dict]]:
    """Parse a sweep index.json into (axis_name, entries).

    Native format written by ``macluster.experiment``::

        {"name": ..., "n": N, "results": [summary, ...]}

    where each summary carries ``run_dir``, ``config`` and ``final_*`` metrics.
    The axis label is built from the config field(s) that vary across runs, and
    the full summary is carried inline so no re-read from disk is needed.

    A few simpler/legacy shapes are also accepted for robustness:
      A) {"axis": "k_frac", "runs": [{"k_frac": 0.02, "run_dir": "..."}, ...]}
      B) {"axis": "link", "runs": {"wifi": "runs/a", "lte": "runs/b"}}
      C) [{"axis": <val>, "run_dir": "..."}, ...]   (axis name inferred)
      D) {"runs": ["runs/a", "runs/b"]}             (axis = run label)
    """
    with open(index_path, "r") as f:
        data = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(index_path))

    def _resolve(p: str) -> str:
        if not p or os.path.isabs(p):
            return p
        for cand in (p, os.path.join(base_dir, p),
                     os.path.join(base_dir, os.path.basename(os.path.normpath(p)))):
            if os.path.exists(cand):
                return cand
        return p

    # --- native macluster.experiment format: {"results": [summary, ...]} ---
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        results = data["results"]
        configs = [r.get("config", {}) or {} for r in results]
        varying = _varying_config_fields(configs)
        axis_name = "+".join(varying) if varying else "run"
        entries: list[dict] = []
        for r, cfg in zip(results, configs):
            if varying:
                axis_val = ", ".join(f"{k}={cfg.get(k)}" for k in varying)
            else:
                axis_val = os.path.basename(os.path.normpath(r.get("run_dir", "")))
            entries.append(
                {"axis": axis_val, "run_dir": _resolve(r.get("run_dir", "")), "summary": r}
            )
        return axis_name, entries

    axis_name = "config"
    entries = []

    if isinstance(data, dict):
        axis_name = data.get("axis", data.get("axis_name", "config"))
        runs = data.get("runs", data.get("entries", data))
        if isinstance(runs, dict) and "runs" not in data and "entries" not in data:
            # data itself is the mapping (rare); skip meta keys
            runs = {k: v for k, v in data.items()
                    if k not in ("axis", "axis_name")}
        if isinstance(runs, dict):
            # mapping axis-value -> run_dir
            for axis_val, rd in runs.items():
                if isinstance(rd, dict):
                    entry = dict(rd)
                    entry.setdefault("axis", axis_val)
                    entry["run_dir"] = _resolve(entry.get("run_dir", ""))
                else:
                    entry = {"axis": axis_val, "run_dir": _resolve(str(rd))}
                entries.append(entry)
        elif isinstance(runs, list):
            for item in runs:
                if isinstance(item, dict):
                    entry = dict(item)
                    rd = entry.get("run_dir") or entry.get("dir") or ""
                    entry["run_dir"] = _resolve(str(rd))
                    if "axis" not in entry and axis_name in entry:
                        entry["axis"] = entry[axis_name]
                    entry.setdefault("axis", entry["run_dir"])
                    entries.append(entry)
                else:
                    entries.append({"axis": str(item), "run_dir": _resolve(str(item))})
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                entry = dict(item)
                rd = entry.get("run_dir") or entry.get("dir") or ""
                entry["run_dir"] = _resolve(str(rd))
                entry.setdefault("axis", entry.get("run_dir"))
                entries.append(entry)
            else:
                entries.append({"axis": str(item), "run_dir": _resolve(str(item))})

    return axis_name, entries


def _final_metric(summary: dict) -> tuple[str, Optional[float]]:
    """Pull the final primary metric from a summary dict."""
    for key, name in (
        ("final_accuracy", "accuracy"),
        ("final_perplexity", "perplexity"),
        ("final_val_loss", "val_loss"),
        ("final_train_loss", "train_loss"),
    ):
        if summary.get(key) is not None:
            return name, float(summary[key])
    return "metric", None


def plot_sweep(index_path: str, out_path: str) -> Optional[str]:
    """Final metric and total_comm_MB across the swept axis (grouped)."""
    axis_name, entries = load_sweep(index_path)
    if not entries:
        return None

    labels: list[str] = []
    metric_vals: list[float] = []
    comm_vals: list[float] = []
    metric_name = "metric"
    for e in entries:
        rd = e.get("run_dir", "")
        summary = e.get("summary") or (load_summary(rd) if rd else {})
        if not summary:
            continue
        mname, mval = _final_metric(summary)
        if mval is None:
            continue
        metric_name = mname
        labels.append(str(e.get("axis", os.path.basename(os.path.normpath(rd)))))
        metric_vals.append(mval)
        comm_vals.append(float(summary.get("total_comm_MB", 0.0)))

    if not labels:
        return None

    x = range(len(labels))
    fig, ax1 = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    width = 0.4
    bars1 = ax1.bar(
        [i - width / 2 for i in x], metric_vals, width=width,
        color="tab:blue", label=f"final {metric_name}",
    )
    ax1.set_ylabel(f"final {_metric_axis_label(metric_name)}", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_xlabel(axis_name)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        [i + width / 2 for i in x], comm_vals, width=width,
        color="tab:orange", label="total comm (MB)",
    )
    ax2.set_ylabel("total communication (MB)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title(f"Sweep over '{axis_name}': final metric and communication cost")
    ax1.legend([bars1, bars2],
               [f"final {metric_name}", "total comm (MB)"], loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot macluster run metrics (Agg backend, saves PNGs)."
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="one or more run directories to plot/overlay")
    parser.add_argument("--out", default="figures/",
                        help="output directory for PNGs (default: figures/)")
    parser.add_argument("--sweep", default=None,
                        help="path to a sweep index.json for fig 3")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    written: list[str] = []

    p1 = plot_metric_vs_time(args.runs, os.path.join(args.out, "fig1_metric_vs_time.png"))
    if p1:
        written.append(p1)

    p2 = plot_comm_vs_round(args.runs, os.path.join(args.out, "fig2_comm_vs_round.png"))
    if p2:
        written.append(p2)

    if args.sweep:
        if os.path.exists(args.sweep):
            p3 = plot_sweep(args.sweep, os.path.join(args.out, "fig3_sweep.png"))
            if p3:
                written.append(p3)
            else:
                print(f"[plot] sweep index produced no plottable entries: {args.sweep}")
        else:
            print(f"[plot] --sweep path not found, skipping fig 3: {args.sweep}")

    adaptive_rd = find_adaptive_run(args.runs)
    if adaptive_rd:
        p4 = plot_adaptive_schedule(
            adaptive_rd, os.path.join(args.out, "fig4_adaptive_schedule.png")
        )
        if p4:
            written.append(p4)
    else:
        print("[plot] no adaptive run among --runs, skipping fig 4")

    for p in written:
        print(f"[plot] wrote {p}")
    if not written:
        print("[plot] WARNING: no figures written (no plottable data found)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
