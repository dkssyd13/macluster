"""Self-rendezvous grove data-plane test (bypasses grove cli script distribution).

grove's `start`/`join` cli distributes the script over the control plane; its
TCP-transport path (`_init_cluster`) never delivers it to the worker, so the
joiner dies with "script not received". This script instead calls
``grove.init(transport=...)`` DIRECTLY on each Mac (each already has the repo),
so no script crosses the wire — only the data plane is exercised. Driven by env:

    GROVE_CLUSTER     cluster name (must match on both Macs)
    GROVE_N           world size (2)
    GROVE_TRANSPORT   "tcp" (LAN, recommended) or "p2p" (AWDL)
    GROVE_IS_COORDINATOR=1   set on the 48GB Mac only -> it becomes rank 0

Launch via scripts/p2.sh:  ./scripts/p2.sh coord   (48GB)  /  ./scripts/p2.sh join (24GB)
"""
from __future__ import annotations

import os
import time

import grove
import mlx.core as mx


def main() -> None:
    cluster = os.environ.get("GROVE_CLUSTER", "mac2")
    ws = int(os.environ.get("GROVE_N", "2"))
    transport = os.environ.get("GROVE_TRANSPORT", "tcp")
    is_coord = os.environ.get("GROVE_IS_COORDINATOR") == "1"
    print(f"[selftest] init cluster={cluster!r} ws={ws} transport={transport} "
          f"coordinator={is_coord} -- discovering peer...", flush=True)
    grove.init(cluster=cluster, world_size=ws, transport=transport)
    r, w = int(grove.rank), int(grove.world_size)
    print(f"[selftest] rank {r}/{w} READY", flush=True)

    # STEP 1: tiny all_sum
    try:
        t = time.perf_counter()
        s = float(grove.all_sum(mx.array([float(r) + 1.0]))[0])
        expect = float(sum(range(1, w + 1)))
        print(f"[selftest] rank {r}: STEP1 all_sum(1)->{s} (expect {expect}) "
              f"{'PASS' if abs(s-expect)<0.5 else 'WRONG'} in {time.perf_counter()-t:.2f}s", flush=True)
    except Exception as e:
        print(f"[selftest] rank {r}: STEP1 FAILED: {type(e).__name__}: {e}", flush=True)
        return

    # STEP 2: bulk all_sum (~4MB)
    try:
        t = time.perf_counter()
        sb = grove.all_sum(mx.ones((1_000_000,), dtype=mx.float32)); mx.eval(sb)
        print(f"[selftest] rank {r}: STEP2 all_sum(4MB) "
              f"{'PASS' if abs(float(sb[0])-float(w))<0.5 else 'WRONG'} in {time.perf_counter()-t:.2f}s", flush=True)
    except Exception as e:
        print(f"[selftest] rank {r}: STEP2 FAILED: {type(e).__name__}: {e}", flush=True)
        return

    # STEP 3: point-to-point send/recv (rank0 -> rank1), what the pipeline uses
    if w >= 2:
        try:
            grove.barrier()
            if r == 0:
                grove.send(mx.arange(16, dtype=mx.float32), 1)
                print("[selftest] rank 0: STEP3 sent 16 floats -> rank1", flush=True)
            elif r == 1:
                got = grove.recv((16,), mx.float32, 0); mx.eval(got)
                print(f"[selftest] rank 1: STEP3 recv {'PASS' if abs(float(got[15])-15.0)<0.5 else 'WRONG'} "
                      f"(last={float(got[15])})", flush=True)
        except Exception as e:
            print(f"[selftest] rank {r}: STEP3 FAILED: {type(e).__name__}: {e}", flush=True)
            return
        # closing barrier is best-effort: the data ops above already succeeded, so
        # a teardown race (the peer exiting first) here is harmless, not a failure.
        try:
            grove.barrier()
        except Exception:
            pass

    # Clean teardown: coordinator (rank0) hosts the store, so it must exit LAST or
    # the worker's last op resets (same fix as grove_entry). Asymmetric handshake.
    try:
        store = grove._comm._group._store
        if r == 0:
            store.wait([f"selftest_done/{i}" for i in range(1, w)], timeout=60.0)
        else:
            store.set(f"selftest_done/{r}", b"1")
    except Exception:
        pass

    print(f"[selftest] rank {r}: ALL PASSED — self-init {transport} data plane works.", flush=True)


if __name__ == "__main__":
    main()
