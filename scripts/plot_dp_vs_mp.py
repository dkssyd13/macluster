#!/usr/bin/env python3
"""DP-vs-MP comparison figures + table for the 2-Mac report (Phase 8).

`plot.py` plots single runs / sweeps; it has no notion of the data-parallel vs
model-parallel head-to-head or of per-node/per-stage peak memory. This script
fills that gap: it scans a runs root, groups the grove runs by
(model, parallelism, algorithm) using each run's ``config.json`` (robust to the
slug format), pairs rank0/rank1, and emits everything the report needs:

  figures/dp_vs_mp/
    convergence.png   val-loss (or perplexity) vs samples AND vs wall-clock,
                      DP and MP overlaid on the matched-budget mid model.
    comm.png          total communication volume (MB), DP vs MP.
    walltime.png      total wall-clock (s), DP vs MP.
    memory.png        peak unified memory PER NODE/STAGE for every model, with the
                      24GB / 48GB node budgets drawn in -- the memory-wall headline
                      (gpt3b's full-replica state exceeds the 24GB node, so DP is
                      infeasible; only the split fits).
    summary.md / summary.csv   one row per (model, paradigm, rank) with the
                      headline numbers, ready to paste into the paper.

Which rank carries which number (verified against train.py):
  * MP  -> the LAST rank (rank1, the 24GB Mac) logs val_loss / perplexity.
  * DP  -> rank0 (the 48GB Mac) logs val_loss / perplexity.
  * comm / total wall-clock are symmetric; peak_mem is per node/stage (both ranks).

Usage:
  uv run python scripts/plot_dp_vs_mp.py --runs runs --out figures/dp_vs_mp
  uv run python scripts/plot_dp_vs_mp.py --runs runs --out figures/dp_vs_mp \
      --node-mem-gb 48 24            # node budgets for the memory-wall lines
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- io
def _load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _load_metrics(run_dir: str) -> list[dict]:
    recs: list[dict] = []
    p = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return recs
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def _series(recs: list[dict], key: str) -> tuple[list[float], list[float]]:
    """(x=samples, y=key) over records that have `key`."""
    xs, ys = [], []
    for r in recs:
        if r.get(key) is not None and r.get("samples") is not None:
            xs.append(float(r["samples"]))
            ys.append(float(r[key]))
    return xs, ys


def _xy(recs: list[dict], xkey: str, ykey: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in recs:
        if r.get(xkey) is not None and r.get(ykey) is not None:
            xs.append(float(r[xkey]))
            ys.append(float(r[ykey]))
    return xs, ys


# --------------------------------------------------------------- run discovery
class Run:
    """A grove run = a (rank0, rank1) pair sharing a base slug."""

    def __init__(self, base: str):
        self.base = base
        self.ranks: dict[int, str] = {}  # rank -> run_dir
        self.cfg: dict = {}
        self.summ: dict[int, dict] = {}

    @property
    def model(self) -> str:
        return self.cfg.get("model", "?")

    @property
    def parallelism(self) -> str:
        return self.cfg.get("parallelism", "?")

    @property
    def algorithm(self) -> str:
        return self.cfg.get("algorithm", "?")

    @property
    def paradigm(self) -> str:
        # report-facing label
        if self.parallelism == "pipeline":
            return "MP (pipeline)"
        return f"DP ({self.algorithm})"

    def label(self) -> str:
        return f"{self.model} | {self.paradigm}"

    def metric_rank_dir(self) -> Optional[str]:
        """Rank that logs val_loss/perplexity: rank0 for DP, last rank for MP."""
        if not self.ranks:
            return None
        r = max(self.ranks) if self.parallelism == "pipeline" else 0
        return self.ranks.get(r, self.ranks.get(max(self.ranks)))


def discover(runs_root: str) -> list[Run]:
    runs: dict[str, Run] = {}
    for cfgp in glob.glob(os.path.join(runs_root, "*", "config.json")):
        run_dir = os.path.dirname(cfgp)
        name = os.path.basename(run_dir)
        if "-grove-rank" not in name:
            continue  # only the real 2-Mac grove runs
        base, _, tail = name.rpartition("-rank")
        try:
            rank = int(tail)
        except ValueError:
            continue
        run = runs.setdefault(base, Run(base))
        run.ranks[rank] = run_dir
        cfg = _load_json(cfgp)
        # config.json from RunLogger may nest under "config"; accept both
        run.cfg = cfg.get("config", cfg) if isinstance(cfg, dict) else {}
        run.summ[rank] = _load_json(os.path.join(run_dir, "summary.json"))
    return [r for r in runs.values() if r.ranks]


# --------------------------------------------------------------------- figures
def _bar(ax, labels, values, title, ylabel, colors=None):
    xs = range(len(labels))
    ax.bar(xs, values, color=colors)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for x, v in zip(xs, values):
        if v is not None:
            ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)


def plot_convergence(mid_runs: list[Run], out: str) -> Optional[str]:
    if not mid_runs:
        return None
    fig, (axs, axw) = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = False
    for run in mid_runs:
        md = run.metric_rank_dir()
        if not md:
            continue
        recs = _load_metrics(md)
        ykey = "perplexity" if any(r.get("perplexity") is not None for r in recs) else "val_loss"
        xs, ys = _series(recs, ykey)
        wx, wy = _xy(recs, "wall_s", ykey)
        if xs:
            axs.plot(xs, ys, marker="o", label=run.paradigm)
            plotted = True
        if wx:
            axw.plot(wx, wy, marker="o", label=run.paradigm)
    if not plotted:
        plt.close(fig)
        return None
    ylab = "perplexity" if ykey == "perplexity" else "val loss"
    for ax, xl in ((axs, "samples seen"), (axw, "wall-clock (s)")):
        ax.set_xlabel(xl)
        ax.set_ylabel(ylab)
        ax.legend()
        ax.grid(True, alpha=0.3)
    if ykey == "perplexity":
        axs.set_yscale("log")
        axw.set_yscale("log")
    fig.suptitle("DP vs MP convergence (matched budget, identical untied-head model)")
    fig.tight_layout()
    p = os.path.join(out, "convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_comm_and_walltime(mid_runs: list[Run], out: str) -> list[str]:
    made = []
    if not mid_runs:
        return made
    labels = [r.paradigm for r in mid_runs]
    comm = [(_pick_summ(r, "total_comm_MB")) for r in mid_runs]
    # total wall-clock = last record's wall_s on any rank (use rank0)
    wall = []
    for r in mid_runs:
        rd = r.ranks.get(0) or next(iter(r.ranks.values()))
        recs = _load_metrics(rd)
        wall.append(recs[-1]["wall_s"] if recs and recs[-1].get("wall_s") is not None else None)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    _bar(ax, labels, [c or 0 for c in comm], "Communication volume (mid model, matched budget)", "total comm (MB)",
         colors=["#4c72b0", "#dd8452"][: len(labels)])
    fig.tight_layout()
    p = os.path.join(out, "comm.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    if any(w is not None for w in wall):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        _bar(ax, labels, [w or 0 for w in wall], "Wall-clock (mid model, matched 3200-sample budget)", "wall-clock (s)",
             colors=["#4c72b0", "#dd8452"][: len(labels)])
        fig.tight_layout()
        p = os.path.join(out, "walltime.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)
    return made


def _pick_summ(run: Run, key: str):
    for rank in sorted(run.summ):
        v = run.summ[rank].get(key)
        if v is not None:
            return v
    return None


def plot_memory(all_runs: list[Run], out: str, node_mem_gb: list[float]) -> Optional[str]:
    """Peak unified memory per node/stage for every model, node budgets drawn in.

    The headline: a DP replica must hold the FULL model on EVERY node, so for a
    model whose full state exceeds the smallest node, DP is infeasible -- only the
    pipeline split fits. We annotate the estimated full-replica state per model.
    """
    rows = []  # (label, rank, peak_mb, paradigm, model, full_state_gb)
    for run in all_runs:
        full_state_gb = None
        mp = _pick_summ(run, "model_param_count")
        if mp:
            full_state_gb = mp * 16 / 1e9  # fp32 weight+grad+adam(m,v) = 16 B/param
        for rank in sorted(run.ranks):
            peak = run.summ.get(rank, {}).get("peak_mem_mb")
            rows.append((f"{run.model}\n{run.paradigm}\nrank{rank}", rank, peak, run.paradigm, run.model, full_state_gb))
    rows = [r for r in rows if r[2] is not None]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(rows)), 5))
    labels = [r[0] for r in rows]
    peaks_gb = [r[2] / 1024 for r in rows]
    colors = ["#dd8452" if "MP" in r[3] else "#4c72b0" for r in rows]
    xs = range(len(rows))
    ax.bar(xs, peaks_gb, color=colors)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("measured peak unified memory (GB)")
    ax.set_title("Peak memory per node/stage (orange=MP split, blue=DP replica)")
    for x, v in zip(xs, peaks_gb):
        ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for mem in node_mem_gb:
        ax.axhline(mem, ls="--", color="red", alpha=0.6)
        ax.text(len(rows) - 0.5, mem, f"{mem:.0f}GB node", color="red", fontsize=8, va="bottom", ha="right")
    fig.tight_layout()
    p = os.path.join(out, "memory.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------- table
def write_table(all_runs: list[Run], out: str) -> tuple[str, str]:
    headers = [
        "model", "paradigm", "rank", "peak_mem_mb", "total_comm_MB",
        "final_val_loss", "final_perplexity", "wall_s", "stage_param_M",
        "model_param_M", "cut", "samples",
    ]
    rows = []
    for run in sorted(all_runs, key=lambda r: (r.model, r.parallelism)):
        for rank in sorted(run.ranks):
            s = run.summ.get(rank, {})
            recs = _load_metrics(run.ranks[rank])
            last = recs[-1] if recs else {}
            spc = s.get("stage_param_counts")
            stage_m = round(spc[rank] / 1e6, 1) if (spc and rank < len(spc)) else None
            mp = s.get("model_param_count")
            # Fall back to the last metrics.jsonl record when summary.json is
            # absent -- e.g. an OOM'd-but-collected 3b never writes summary.json,
            # but every per-round record was flushed, so the curve survives.
            last_val = last.get("val_loss")
            for r in reversed(recs):
                if r.get("val_loss") is not None:
                    last_val = r["val_loss"]
                    break
            rows.append({
                "model": run.model,
                "paradigm": run.paradigm,
                "rank": rank,
                "peak_mem_mb": s.get("peak_mem_mb") or last.get("peak_mem_mb"),
                "total_comm_MB": s.get("total_comm_MB") or (
                    round(last["comm_bytes_cum"] / 1e6, 3) if last.get("comm_bytes_cum") is not None else None),
                "final_val_loss": s.get("final_val_loss") if s.get("final_val_loss") is not None else last_val,
                "final_perplexity": last.get("perplexity"),
                "wall_s": round(last["wall_s"], 1) if last.get("wall_s") is not None else None,
                "stage_param_M": stage_m,
                "model_param_M": round(mp / 1e6, 1) if mp else None,
                "cut": s.get("cut"),
                "samples": last.get("samples"),
            })
    csv_p = os.path.join(out, "summary.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    md_p = os.path.join(out, "summary.md")
    with open(md_p, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for r in rows:
            f.write("| " + " | ".join("" if r[h] is None else str(r[h]) for h in headers) + " |\n")
    return md_p, csv_p


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DP-vs-MP comparison figures + table")
    ap.add_argument("--runs", default="runs", help="runs root to scan")
    ap.add_argument("--out", default="figures/dp_vs_mp", help="output dir")
    ap.add_argument("--node-mem-gb", default="48,24", help="node budgets for the memory-wall lines")
    ap.add_argument("--mid-model", default="gpt2_untied", help="model used for the matched-budget DP-vs-MP comparison")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    node_mem = [float(x) for x in args.node_mem_gb.split(",") if x.strip()]
    runs = discover(args.runs)
    if not runs:
        print(f"[plot_dp_vs_mp] no grove runs found under {args.runs!r} "
              f"(expected dirs like '<slug>-grove-rank0'). Nothing to plot.")
        return 1

    print(f"[plot_dp_vs_mp] found {len(runs)} grove run(s):")
    for r in sorted(runs, key=lambda x: x.base):
        print(f"  - {r.label():40s} ranks={sorted(r.ranks)}")

    mid = [r for r in runs if r.model == args.mid_model]
    # order DP first then MP for stable colors
    mid.sort(key=lambda r: 0 if r.parallelism != "pipeline" else 1)

    made = []
    c = plot_convergence(mid, args.out)
    if c:
        made.append(c)
    made += plot_comm_and_walltime(mid, args.out)
    m = plot_memory(runs, args.out, node_mem)
    if m:
        made.append(m)
    md_p, csv_p = write_table(runs, args.out)
    made += [md_p, csv_p]

    print(f"[plot_dp_vs_mp] wrote {len(made)} file(s) to {args.out}/:")
    for p in made:
        print(f"  - {p}")
    if not mid:
        print(f"[plot_dp_vs_mp] WARN: no runs for mid-model {args.mid_model!r}; "
              f"convergence/comm/walltime figures skipped (need mp_mid + dp_mid).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
