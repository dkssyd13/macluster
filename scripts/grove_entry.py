"""grove worker entry point for the 2-MacBook (`--backend grove`) run.

`grove start` / `grove join` can only launch a **Python file that defines
`main()`** (they `importlib`-load the script; the joiner receives only the
script *source*, no argv). So this is the single launchable form of a grove
run, and the `TrainConfig` travels via `MACLUSTER_*` environment variables that
must be exported **identically on every Mac** (a config-consensus check below
aborts loudly on mismatch, since a divergent config silently breaks the
deterministic data sharding).

Launch (see docs/PHASE7_TODO.md §3):

    # mac-A (launcher)
    export MACLUSTER_TASK=cifar10 MACLUSTER_ALGORITHM=diloco MACLUSTER_ROUNDS=25 MACLUSTER_H=20
    uv run grove start scripts/grove_entry.py -n 2 --name macluster --logs

    # mac-B (joiner) -- export the SAME MACLUSTER_* env, then:
    uv run grove join macluster --logs

Single-machine smoke (collectives are W=1 no-ops):

    MACLUSTER_SYNTHETIC=1 MACLUSTER_ALGORITHM=dense MACLUSTER_ROUNDS=2 \
        uv run python scripts/grove_entry.py
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import asdict

import grove
import mlx.core as mx

from macluster.train import TrainConfig, run_training


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "", "false", "no")


def _cfg_from_env() -> TrainConfig:
    """Build a grove TrainConfig from MACLUSTER_* env vars (TrainConfig defaults
    fill the gaps). ``world_size`` is bound to the real cluster size so the task
    builds exactly one data shard per rank."""
    d = TrainConfig()
    g = os.environ.get

    def _opt_int(name, fallback):
        return int(os.environ[name]) if name in os.environ else fallback

    def _opt_float(name, fallback):
        return float(os.environ[name]) if name in os.environ else fallback

    def _opt_str(name, fallback):
        return os.environ[name] if name in os.environ else fallback

    return TrainConfig(
        task=g("MACLUSTER_TASK", d.task),
        model=g("MACLUSTER_MODEL", d.model),
        algorithm=g("MACLUSTER_ALGORITHM", d.algorithm),
        world_size=int(grove.world_size),  # one shard/stage per rank; ignore env
        backend="grove",
        parallelism=g("MACLUSTER_PARALLELISM", d.parallelism),
        cut=_opt_str("MACLUSTER_CUT", d.cut),
        n_micro=int(g("MACLUSTER_N_MICRO", str(d.n_micro))),
        stage_mem_gb=_opt_str("MACLUSTER_STAGE_MEM_GB", d.stage_mem_gb),
        rounds=int(g("MACLUSTER_ROUNDS", str(d.rounds))),
        max_steps=_opt_int("MACLUSTER_MAX_STEPS", d.max_steps),
        batch_size=int(g("MACLUSTER_BATCH_SIZE", str(d.batch_size))),
        seq_len=int(g("MACLUSTER_SEQ_LEN", str(d.seq_len))),
        H=int(g("MACLUSTER_H", str(d.H))),
        k_frac=float(g("MACLUSTER_K_FRAC", str(d.k_frac))),
        outer_lr=float(g("MACLUSTER_OUTER_LR", str(d.outer_lr))),
        inner_opt=g("MACLUSTER_INNER_OPT", d.inner_opt),
        inner_lr=float(g("MACLUSTER_INNER_LR", str(d.inner_lr))),
        link=g("MACLUSTER_LINK", d.link),
        eval_every=int(g("MACLUSTER_EVAL_EVERY", str(d.eval_every))),
        seed=int(g("MACLUSTER_SEED", str(d.seed))),
        synthetic=_env_flag("MACLUSTER_SYNTHETIC", d.synthetic),
        data_dir=g("MACLUSTER_DATA_DIR", d.data_dir),
        target_metric=_opt_float("MACLUSTER_TARGET_METRIC", d.target_metric),
    )


def _assert_config_consensus(cfg: TrainConfig) -> None:
    """Abort if ranks resolved different configs (env mismatch across Macs)."""
    if grove.world_size <= 1:
        return
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str)
    local = int(hashlib.sha256(blob.encode()).hexdigest(), 16) % (2 ** 23)  # exact in fp32
    total = float(grove.all_sum(mx.array([float(local)]))[0])
    if abs(total - local * grove.world_size) > 0.5:
        raise SystemExit(
            f"[grove_entry] rank {grove.rank}: config disagrees across ranks. "
            f"Every Mac must export identical MACLUSTER_* env. (local hash={local})"
        )


def _init_static_tcp(peers_spec: str, rank: int, world_size: int, timeout: float) -> None:
    """Initialize grove over TCP with explicit rank->host mapping.

    This bypasses grove's Bonjour/mDNS discovery path, which can be blocked by
    campus Wi-Fi or macOS Local Network privacy even when direct LAN TCP works.
    """
    from grove._init import _init_packer
    from grove._types import DEFAULT_BASE_PORT, TransportType
    from grove.comm import Communicator
    from grove.group import Group
    from grove.store.tcp_store import TCPStore

    peers = [p.strip() for p in peers_spec.split(",") if p.strip()]
    if len(peers) != world_size:
        raise SystemExit(
            f"[grove_entry] GROVE_PEERS has {len(peers)} host(s), "
            f"but GROVE_N/world_size is {world_size}: {peers!r}"
        )
    if not 0 <= rank < world_size:
        raise SystemExit(f"[grove_entry] GROVE_RANK={rank} out of range for world_size={world_size}")

    coord_host = peers[0]
    if rank == 0:
        try:
            local_ips = {addr[4][0] for addr in socket.getaddrinfo(socket.gethostname(), None)}
        except socket.gaierror:
            local_ips = set()
        local_ips.update({"127.0.0.1", "::1"})
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            local_ips.add(probe.getsockname()[0])
            probe.close()
        except OSError:
            pass
        if coord_host not in local_ips:
            raise SystemExit(
                f"[grove_entry] GROVE_PEERS starts with {coord_host}, but this rank0 Mac "
                f"does not currently own that IP (local={sorted(local_ips)}). "
                "Reconnect Wi-Fi or rerun ./scripts/run2_coord.sh so it detects "
                "the current IP."
            )
    grove.rank = rank
    grove.world_size = world_size
    store = TCPStore(
        rank=rank,
        world_size=world_size,
        host=coord_host,
        port=DEFAULT_BASE_PORT - 199,
        timeout=timeout,
    )
    # grove's TCPStore.wait() uses a hard-coded 30s server-side wait. In static
    # peer mode the joiner can be delayed by shell startup or result collection,
    # so use the requested GROVE_TIMEOUT for the transport address rendezvous.
    def _long_wait(keys: list[str], wait_timeout: float | None = None) -> None:
        deadline = time.monotonic() + (wait_timeout or timeout)
        missing = list(keys)
        while time.monotonic() < deadline:
            still_missing = []
            for key in missing:
                try:
                    store.get_nowait(key)
                except KeyError:
                    still_missing.append(key)
            if not still_missing:
                return
            missing = still_missing
            time.sleep(0.05)
        raise TimeoutError(f"Wait timed out for keys: {missing}")

    store.wait = _long_wait  # type: ignore[method-assign]
    group = Group(rank, world_size, store, TransportType.TCP)

    # Static peer mode is for a fixed 2-Mac experiment, not elastic/fault-tolerant
    # membership. Avoid grove's CoordinatorServer heartbeat path: long MLX calls
    # on the slower rank can starve its Python heartbeat thread for >60s, causing
    # false "Node missed heartbeat" reform while the data plane is still valid.
    grove._coordinator = None
    grove._worker_client = None
    grove._comm = Communicator(group, None)
    _init_packer()
    print(
        f"[grove_entry] static TCP init rank {rank}/{world_size} "
        f"peers={peers} coord={coord_host}",
        flush=True,
    )


def _patch_grove_socket_timeout(timeout: float) -> None:
    """Raise grove's point-to-point socket timeout for slow Wi-Fi pipeline steps."""
    from grove.transport.socket_conn import SocketConnection

    if getattr(SocketConnection, "_macluster_timeout_patch", False):
        return
    orig_init = SocketConnection.__init__

    def _init_with_timeout(self, sock):
        orig_init(self, sock)
        self._sock.settimeout(timeout)

    SocketConnection.__init__ = _init_with_timeout
    SocketConnection._macluster_timeout_patch = True
    print(f"[grove_entry] socket timeout set to {timeout:.1f}s", flush=True)


def main() -> None:
    # Three launch paths:
    #  1) via `grove start`/`join`: the world is already up (grove._comm set) and
    #     the script was distributed -- nothing to init here.
    #  2) via scripts/p2.sh (GROVE_CLUSTER + GROVE_N>1 set): SELF-rendezvous with
    #     grove.init(transport=...). This bypasses the grove cli, whose tcp path
    #     never delivers the script to the joiner ("script not received"); each Mac
    #     runs its own local copy instead.
    #  3) standalone `python scripts/grove_entry.py`: world_size=1 smoke.
    if grove._comm is None:
        cluster = os.environ.get("GROVE_CLUSTER")
        ws = int(os.environ.get("GROVE_N", "1"))
        peers = os.environ.get("GROVE_PEERS", "")
        timeout = float(os.environ.get("GROVE_TIMEOUT", "120.0"))
        _patch_grove_socket_timeout(float(os.environ.get("GROVE_SOCKET_TIMEOUT", str(timeout))))
        if peers and ws > 1:
            _init_static_tcp(
                peers,
                rank=int(os.environ.get("GROVE_RANK", "0")),
                world_size=ws,
                timeout=timeout,
            )
        elif cluster and ws > 1:
            grove.init(
                cluster=cluster,
                world_size=ws,
                transport=os.environ.get("GROVE_TRANSPORT", "tcp"),
                timeout=timeout,
            )
        elif grove.world_size <= 1:
            grove.init()

    cfg = _cfg_from_env()
    mx.random.seed(cfg.seed)  # rank-identical model init (no broadcast needed)
    _assert_config_consensus(cfg)

    runs_root = os.environ.get("MACLUSTER_RUNS_ROOT", "runs")
    run_dir = os.path.join(runs_root, f"{cfg.slug()}-rank{grove.rank}")
    print(f"[grove_entry] rank {grove.rank}/{grove.world_size} -> {run_dir}")
    summary = run_training(cfg, run_dir)
    print(f"[grove_entry] rank {grove.rank} done: {json.dumps(summary, default=str)}")

    # Clean teardown for the self-init (scripts/p2.sh / run2.sh) path: the
    # coordinator (rank 0) hosts the TCPStore + CoordinatorServer *inside its own
    # process*, so if it exits first the workers' in-flight store ops reset
    # ("Control connection lost" -> barrier ConnectionResetError). The grove
    # start/join cli manages this lifecycle; here we do it ourselves with an
    # asymmetric done-handshake so the store owner always exits LAST.
    if grove.world_size > 1 and grove._comm is not None:
        try:
            store = grove._comm._group._store
            if grove.rank == 0:
                store.wait(
                    [f"macluster_done/{r}" for r in range(1, int(grove.world_size))],
                    timeout=300.0,
                )
            else:
                store.set(f"macluster_done/{grove.rank}", b"1")
        except Exception as e:  # never let teardown bookkeeping fail a finished run
            print(f"[grove_entry] rank {grove.rank} teardown handshake skipped: {e}")


if __name__ == "__main__":
    main()
