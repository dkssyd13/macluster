"""Minimal 2-Mac grove data-plane diagnostic — NO model, NO data, NO macluster.

Isolates "can the two Macs actually move bytes" from any app logic. Run it the
same way as the real cluster:

    # 48GB Mac (rank0)
    uv run grove start scripts/grove_ping.py -n 2 --name ping --logs
    # 24GB Mac (rank1) -- within ~1 min
    uv run grove join ping --logs

It tests, in order:
  1. all_sum of a 1-element array   (the tiny collective config/data-consensus use)
  2. all_sum of a ~4MB array        (a bulk transfer, like a real payload)
  3. send/recv rank0 -> rank1       (the point-to-point op the PIPELINE uses)
Each step prints PASS/FAIL on each rank. If step 1 already fails with a timeout,
the data plane itself is blocked (firewall / AWDL), not the macluster code.
"""
from __future__ import annotations

import time

import grove
import mlx.core as mx


def main() -> None:
    if grove._comm is None and grove.world_size <= 1:
        grove.init()
    r, w = int(grove.rank), int(grove.world_size)
    print(f"[ping] rank {r}/{w} starting")

    # 1) tiny all_sum
    try:
        t = time.perf_counter()
        s = float(grove.all_sum(mx.array([float(r) + 1.0]))[0])
        dt = time.perf_counter() - t
        expect = float(sum(range(1, w + 1)))
        ok = abs(s - expect) < 0.5
        print(f"[ping] rank {r}: STEP1 all_sum(1) -> {s} (expect {expect}) "
              f"{'PASS' if ok else 'WRONG'} in {dt:.2f}s")
    except Exception as e:
        print(f"[ping] rank {r}: STEP1 all_sum(1) FAILED: {type(e).__name__}: {e}")
        return

    # 2) bulk all_sum (~4 MB)
    try:
        t = time.perf_counter()
        big = mx.ones((1_000_000,), dtype=mx.float32)
        sb = grove.all_sum(big)
        mx.eval(sb)
        dt = time.perf_counter() - t
        ok = abs(float(sb[0]) - float(w)) < 0.5
        print(f"[ping] rank {r}: STEP2 all_sum(4MB) {'PASS' if ok else 'WRONG'} in {dt:.2f}s")
    except Exception as e:
        print(f"[ping] rank {r}: STEP2 all_sum(4MB) FAILED: {type(e).__name__}: {e}")
        return

    # 3) point-to-point send/recv (rank0 -> rank1), what the pipeline seam uses
    if w < 2:
        print(f"[ping] rank {r}: STEP3 skipped (needs 2 ranks). STEPS 1-2 PASSED.")
        return
    try:
        grove.barrier()
        t = time.perf_counter()
        if r == 0:
            grove.send(mx.arange(16, dtype=mx.float32), 1)
            print(f"[ping] rank 0: STEP3 sent 16 floats -> rank1")
        elif r == 1:
            got = grove.recv((16,), mx.float32, 0)
            mx.eval(got)
            ok = abs(float(got[15]) - 15.0) < 0.5
            print(f"[ping] rank 1: STEP3 recv from rank0 {'PASS' if ok else 'WRONG'} "
                  f"in {time.perf_counter()-t:.2f}s (last={float(got[15])})")
        grove.barrier()
    except Exception as e:
        print(f"[ping] rank {r}: STEP3 send/recv FAILED: {type(e).__name__}: {e}")
        return

    print(f"[ping] rank {r}: ALL STEPS PASSED — the data plane works end-to-end.")


if __name__ == "__main__":
    main()
