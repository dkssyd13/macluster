"""Adaptive synchronization policy (axis 3) — the project's novel contribution.

A small control layer on top of SparseLoCo that adjusts, each round, *how
often* to synchronize (H) and *how aggressively* to compress (k_frac) from
cheaply-measured signals:

  - ``sync_s / compute_s`` ratio: if communication dominates compute, raise H
    (sync less often); if compute dominates, lower H (sync more, better
    convergence).
  - link type: on a degraded/AWDL-only link, shrink k_frac (compress harder);
    on healthy Wi-Fi, relax k_frac toward less compression.

This is intentionally a thin reactive controller — not a reproduction of
Asteroid's pipeline planner or StellaTrain's full optimizer — matching the
proposal's "small adaptive layer" framing.
"""

from __future__ import annotations

import math

from .sparseloco import SparseLoCo

# Link profiles considered "degraded" — compress harder on these.
_DEGRADED_LINKS = {"awdl", "wifi_degraded"}


class AdaptiveSync(SparseLoCo):
    name = "adaptive"

    def __init__(
        self,
        H: int = 30,
        outer_lr: float = 1.0,
        k_frac: float = 0.02,
        error_decay: float = 0.95,
        H_bounds: tuple[int, int] = (8, 256),
        k_bounds: tuple[float, float] = (0.005, 0.1),
        ratio_high: float = 1.0,    # sync >= compute -> back off
        ratio_low: float = 0.25,    # sync cheap -> sync more
        ema_beta: float = 0.6,
    ):
        super().__init__(H=H, outer_lr=outer_lr, k_frac=k_frac, error_decay=error_decay)
        self._H_min, self._H_max = H_bounds
        self._k_min, self._k_max = k_bounds
        self._ratio_high = ratio_high
        self._ratio_low = ratio_low
        self._beta = ema_beta
        self._ratio_ema: float | None = None

    def observe(self, compute_s: float, sync_s: float, link_name: str) -> None:
        ratio = sync_s / max(compute_s, 1e-6)
        self._ratio_ema = (
            ratio if self._ratio_ema is None else self._beta * self._ratio_ema + (1 - self._beta) * ratio
        )

        # --- adjust sync interval H from the comm/compute ratio ---
        if self._ratio_ema > self._ratio_high:
            self._H = min(self._H_max, int(math.ceil(self._H * 1.5)))
        elif self._ratio_ema < self._ratio_low:
            self._H = max(self._H_min, int(math.floor(self._H / 1.5)) or self._H_min)

        # --- adjust compression k_frac from link health ---
        if link_name in _DEGRADED_LINKS:
            self._k_frac = max(self._k_min, self._k_frac * 0.5)
        else:
            self._k_frac = min(self._k_max, self._k_frac * 1.25)

    def knobs(self) -> dict:
        return {
            "H": self._H,
            "k_frac": round(self._k_frac, 5),
            "ratio_ema": None if self._ratio_ema is None else round(self._ratio_ema, 4),
        }
