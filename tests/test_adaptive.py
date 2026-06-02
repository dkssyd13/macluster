"""Unit tests for algorithms/adaptive.py (AdaptiveSync reactive controller).

We drive ``observe`` directly with synthetic compute/sync timings and link
names and check the two control behaviours: raising H when the sync/compute
ratio is high, and shrinking/growing k_frac with link health.
"""

from __future__ import annotations

import pytest

from macluster.algorithms.adaptive import AdaptiveSync


def test_raises_H_when_sync_dominates():
    a = AdaptiveSync(H=20)
    H0 = a._H
    # ratio = sync/compute is huge (>> ratio_high=1.0) -> H grows by *1.5
    a.observe(compute_s=0.01, sync_s=10.0, link_name="wifi")
    assert a._H > H0
    # ceil(20 * 1.5) = 30
    assert a._H == 30


def test_repeated_high_ratio_keeps_growing_then_caps():
    a = AdaptiveSync(H=20, H_bounds=(8, 64))
    for _ in range(50):
        a.observe(compute_s=0.001, sync_s=100.0, link_name="wifi")
    # H must have grown and saturated at the upper bound
    assert a._H == 64


def test_lowers_H_when_compute_dominates():
    a = AdaptiveSync(H=30)
    H0 = a._H
    # warm the EMA below ratio_low (=0.25) with cheap-sync rounds
    for _ in range(8):
        a.observe(compute_s=10.0, sync_s=0.001, link_name="wifi")
    assert a._H < H0


def test_H_does_not_go_below_min():
    a = AdaptiveSync(H=10, H_bounds=(8, 256))
    for _ in range(50):
        a.observe(compute_s=100.0, sync_s=0.0001, link_name="wifi")
    assert a._H >= 8
    assert a._H == 8


def test_shrinks_k_frac_on_awdl():
    a = AdaptiveSync(k_frac=0.02)
    k0 = a._k_frac
    a.observe(compute_s=1.0, sync_s=1.0, link_name="awdl")
    # degraded link -> *0.5
    assert a._k_frac < k0
    assert a._k_frac == pytest.approx(0.01)


def test_shrinks_k_frac_on_wifi_degraded():
    a = AdaptiveSync(k_frac=0.04)
    a.observe(compute_s=1.0, sync_s=1.0, link_name="wifi_degraded")
    assert a._k_frac == pytest.approx(0.02)


def test_grows_k_frac_on_wifi():
    a = AdaptiveSync(k_frac=0.02)
    k0 = a._k_frac
    a.observe(compute_s=1.0, sync_s=1.0, link_name="wifi")
    # healthy link -> *1.25
    assert a._k_frac > k0
    assert a._k_frac == pytest.approx(0.025)


def test_k_frac_clamped_to_bounds():
    a = AdaptiveSync(k_frac=0.02, k_bounds=(0.005, 0.1))
    # shrink hard on awdl repeatedly -> floor
    for _ in range(50):
        a.observe(compute_s=1.0, sync_s=0.5, link_name="awdl")
    assert a._k_frac >= 0.005
    assert a._k_frac == pytest.approx(0.005)

    b = AdaptiveSync(k_frac=0.02, k_bounds=(0.005, 0.1))
    # grow on wifi repeatedly -> ceiling
    for _ in range(50):
        b.observe(compute_s=0.5, sync_s=0.5, link_name="wifi")
    assert b._k_frac <= 0.1
    assert b._k_frac == pytest.approx(0.1)


def test_knobs_report_current_state():
    a = AdaptiveSync(H=20, k_frac=0.02)
    a.observe(compute_s=0.01, sync_s=10.0, link_name="wifi")
    knobs = a.knobs()
    assert knobs["H"] == a._H
    assert knobs["k_frac"] == pytest.approx(round(a._k_frac, 5))
    assert "ratio_ema" in knobs and knobs["ratio_ema"] is not None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
