"""Unit tests for emulation/link.py (transfer cost, allreduce, link schedule)."""

from __future__ import annotations

import numpy as np
import pytest

from macluster.emulation.link import (
    AWDL_ONLY,
    WIFI_UPGRADED,
    LinkProfile,
    LinkSchedule,
    allreduce_bytes,
    get_profile,
)


def test_transfer_time_increases_with_bytes():
    p = WIFI_UPGRADED
    t_small = p.transfer_time(1_000)
    t_big = p.transfer_time(10_000_000)
    assert t_big > t_small


def test_transfer_time_monotonic_no_jitter():
    # DATACENTER has zero jitter -> strictly deterministic + monotone
    p = get_profile("datacenter")
    sizes = [0, 1_000, 100_000, 5_000_000, 50_000_000]
    times = [p.transfer_time(b) for b in sizes]
    assert times == sorted(times)
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))


def test_zero_bytes_is_just_latency():
    p = WIFI_UPGRADED
    assert p.transfer_time(0) == pytest.approx(p.latency_ms / 1000.0)


def test_awdl_slower_than_wifi_equal_payload():
    payload = 5_000_000
    # disable jitter influence by passing no rng (jitter only applied with rng)
    t_wifi = WIFI_UPGRADED.transfer_time(payload)
    t_awdl = AWDL_ONLY.transfer_time(payload)
    assert t_awdl > t_wifi


def test_jitter_only_with_rng():
    p = LinkProfile("j", bandwidth_mbps=100.0, latency_ms=5.0, jitter_ms=50.0)
    # without rng, jitter term is skipped -> deterministic
    a = p.transfer_time(1000)
    b = p.transfer_time(1000)
    assert a == b
    # with rng, the (non-negative) jitter can only add time
    rng = np.random.default_rng(0)
    with_jitter = p.transfer_time(1000, rng)
    assert with_jitter >= a


def test_transfer_time_serialization_formula():
    p = LinkProfile("x", bandwidth_mbps=8.0, latency_ms=0.0, jitter_ms=0.0)
    # 8 Mbps = 1e6 bytes/sec; 1e6 bytes -> 8e6 bits / 8e6 bps = 1.0 s
    assert p.transfer_time(1_000_000) == pytest.approx(1.0)


def test_allreduce_ring_formula():
    payload = 1000.0
    W = 4
    expected = 2.0 * (W - 1) / W * payload  # 2 * 3/4 * 1000 = 1500
    assert allreduce_bytes(payload, W, "ring") == pytest.approx(expected)


def test_allreduce_ring_w2_is_payload():
    # W=2 ring -> 2 * 1/2 * payload = payload
    assert allreduce_bytes(1234.0, 2, "ring") == pytest.approx(1234.0)


def test_allreduce_gather_formula():
    payload = 1000.0
    W = 4
    assert allreduce_bytes(payload, W, "gather") == pytest.approx(payload * W)


def test_allreduce_single_worker_is_zero():
    assert allreduce_bytes(1000.0, 1, "ring") == 0.0
    assert allreduce_bytes(1000.0, 1, "gather") == 0.0


def test_allreduce_gather_exceeds_ring_for_same_payload():
    payload, W = 1000.0, 4
    assert allreduce_bytes(payload, W, "gather") > allreduce_bytes(payload, W, "ring")


def test_schedule_constant():
    sched = LinkSchedule.constant(WIFI_UPGRADED)
    for rnd in (0, 1, 50, 999):
        assert sched.at(rnd) is WIFI_UPGRADED


def test_schedule_switch_boundary():
    at_round = 10
    sched = LinkSchedule.switch(WIFI_UPGRADED, AWDL_ONLY, at_round)
    # before at_round -> first profile
    assert sched.at(0) is WIFI_UPGRADED
    assert sched.at(at_round - 1) is WIFI_UPGRADED
    # at and after -> second profile
    assert sched.at(at_round) is AWDL_ONLY
    assert sched.at(at_round + 5) is AWDL_ONLY


def test_get_profile_unknown_raises():
    with pytest.raises(KeyError):
        get_profile("nonexistent_link")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
