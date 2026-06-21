"""Per-run logging: one JSONL file of per-round records plus config/summary.

A run directory contains:
  - ``config.json``   the resolved TrainConfig
  - ``metrics.jsonl`` one JSON record per sync round
  - ``summary.json``  final aggregates (time-to-accuracy, total comm bytes, ...)
"""

from __future__ import annotations

import json
import os
import time


class RunLogger:
    def __init__(self, run_dir: str, config: dict):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.config = config
        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary_path):
            os.remove(summary_path)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)
        self._fh = open(os.path.join(run_dir, "metrics.jsonl"), "w")
        self._t0 = time.perf_counter()
        self.records: list[dict] = []

    def log(self, record: dict) -> None:
        record = {"wall_s": round(time.perf_counter() - self._t0, 4), **record}
        self.records.append(record)
        self._fh.write(json.dumps(record, default=float) + "\n")
        self._fh.flush()

    def summary(self, extra: dict | None = None) -> dict:
        s: dict = {"run_dir": self.run_dir, "n_records": len(self.records)}
        if extra:
            s.update(extra)
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(s, f, indent=2, default=str)
        return s

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
