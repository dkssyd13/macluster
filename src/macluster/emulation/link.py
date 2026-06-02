"""Link emulation: model AWDL-only vs Wi-Fi-upgraded transfer cost (axis 1).

On a single development machine we cannot exercise real AWDL / Wi-Fi, so we
attach an analytic cost model to every synchronization round. Given the number
of bytes a worker must move during a sync, a :class:`LinkProfile` returns the
wall-clock time that transfer would take on a real link. This lets axis-1
(AWDL vs Wi-Fi) and axis-3 (the adaptive policy) be developed and validated
before the 2-MacBook runs (Phase 7) provide ground-truth numbers.

The numbers are deliberately representative, not measured; the 2-MacBook runs
are what calibrate them. They are easy to override from a sweep config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinkProfile:
    """An emulated network link characterised by goodput, latency and jitter."""

    name: str
    bandwidth_mbps: float       # effective application goodput, megabits / sec
    latency_ms: float           # base one-way latency
    jitter_ms: float = 0.0      # latency noise, std-dev (truncated at 0)

    def transfer_time(self, num_bytes: float, rng: np.random.Generator | None = None) -> float:
        """Seconds to move ``num_bytes`` over this link (latency + serialization)."""
        base = self.latency_ms / 1000.0
        if num_bytes > 0:
            bits = num_bytes * 8.0
            base += bits / (self.bandwidth_mbps * 1e6)
        if self.jitter_ms and rng is not None:
            base += abs(float(rng.normal(0.0, self.jitter_ms / 1000.0)))
        return base


# Canonical profiles. Representative goodput figures for the report's emulation.
WIFI_UPGRADED = LinkProfile("wifi", bandwidth_mbps=300.0, latency_ms=3.0, jitter_ms=1.0)
AWDL_ONLY = LinkProfile("awdl", bandwidth_mbps=60.0, latency_ms=10.0, jitter_ms=6.0)
WIFI_DEGRADED = LinkProfile("wifi_degraded", bandwidth_mbps=40.0, latency_ms=20.0, jitter_ms=10.0)
DATACENTER = LinkProfile("datacenter", bandwidth_mbps=10000.0, latency_ms=0.1)

PROFILES: dict[str, LinkProfile] = {
    p.name: p for p in (WIFI_UPGRADED, AWDL_ONLY, WIFI_DEGRADED, DATACENTER)
}


def get_profile(name: str) -> LinkProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown link profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


class LinkSchedule:
    """The active :class:`LinkProfile` as a function of sync round.

    Supports a constant link or a scripted switch (e.g. Wi-Fi that drops to
    AWDL-only partway through) to test the adaptive policy's reaction.
    """

    def __init__(self, segments: list[tuple[int, LinkProfile]]):
        assert segments, "need at least one segment"
        self._segments = sorted(segments, key=lambda s: s[0])

    def at(self, round_idx: int) -> LinkProfile:
        active = self._segments[0][1]
        for start, prof in self._segments:
            if round_idx >= start:
                active = prof
            else:
                break
        return active

    @classmethod
    def constant(cls, profile: LinkProfile) -> "LinkSchedule":
        return cls([(0, profile)])

    @classmethod
    def switch(cls, first: LinkProfile, second: LinkProfile, at_round: int) -> "LinkSchedule":
        return cls([(0, first), (at_round, second)])


def allreduce_bytes(payload_bytes: float, world_size: int, topology: str = "ring") -> float:
    """Bytes a single worker sends+receives to all-reduce ``payload_bytes``.

    Ring all-reduce moves ~``2*(W-1)/W * payload`` per worker. For a tiny
    cluster (W=2) that is ~``payload``. Sparse syncs gather raw (idx, val)
    buffers, modelled with ``topology="gather"``.
    """
    if world_size <= 1:
        return 0.0
    if topology == "ring":
        return 2.0 * (world_size - 1) / world_size * payload_bytes
    if topology == "gather":
        # send own payload once, receive (W-1) others
        return payload_bytes * world_size
    return payload_bytes
