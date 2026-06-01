#!/usr/bin/env python3
"""Stand-alone TDSQL Multimodal benchmark runner.

Wraps `python -m vectordb_bench.cli.vectordbbench tdsqlmultimodal*` with
sensible defaults for the cases / indexes we validated in Phase 1/2/3.

Usage examples (run inside the repo's venv):

    # Smoke test: OpenAI-50K, IVF_SQ, single batch, concurrent only
    ./scripts_a1d/run_tdsql_benchmark.py --case openai50k --index ivf_sq

    # 1M case, hnsw_sq with tuned params
    ./scripts_a1d/run_tdsql_benchmark.py --case bioasq1m --index ivf_hnsw_sq \
        --m 32 --ef-construction 300 --nprobs 8

    # 10M case with reduced serial test set to save time
    ./scripts_a1d/run_tdsql_benchmark.py --case bioasq10m --index ivf_sq \
        --serial-limit 500

    # External instance with custom credentials
    ./scripts_a1d/run_tdsql_benchmark.py --case openai500k --index ivf_sq \
        --host 10.0.0.5 --port 13306 --user bench --password 'secret'

    # Dry-run to print the command without executing
    ./scripts_a1d/run_tdsql_benchmark.py --case openai500k --index ivf_sq --dry-run

Connection params can also come from env vars:
    TDSQL_HOST, TDSQL_PORT, TDSQL_USER, TDSQL_PASSWORD

Critical environment knobs this script sets for you:
    TMPDIR / TMP / TEMP   = --tmpdir (default /data/tmp/lance_scratch)
        Required for 10M+ cases: Lance index build writes ~10-20 GB scratch
        and the default /tmp on root fs may not fit.
    TDSQL_MULTIMODAL_PREWARM = 1 (unless --no-prewarm)
    TDSQL_MULTIMODAL_PREWARM_PROBES = --prewarm-probes (default 100)
    VDBBENCH_SERIAL_TEST_LIMIT = --serial-limit (default 0 = full set)

Note: this script only configures the client side. If you change TDSQL
server config (max_allowed_packet etc.) you still need to restart the
server with the same TMPDIR env. See
.agent/tdsql-concurrent-rerun/plan.md section 七-A for restart procedure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (case_name, vdbbench case-type enum value, dim, default batch size)
CASES: dict[str, tuple[str, int, int]] = {
    "openai50k":  ("Performance1536D50K",  1536,  4000),
    "openai500k": ("Performance1536D500K", 1536,  5000),
    "openai5m":   ("Performance1536D5M",   1536,  5000),
    "bioasq1m":   ("Performance1024D1M",   1024,  8000),
    "bioasq10m":  ("Performance1024D10M",  1024,  8000),
    "cohere1m":   ("Performance768D1M",     768, 10000),
    "cohere10m":  ("Performance768D10M",    768, 10000),
    "glove1m":    ("Performance200D1M",     200, 20000),
}

# (index_name, vdbbench CLI subcommand)
INDEXES: dict[str, str] = {
    "ivf_flat":      "tdsqlmultimodalivfflat",
    "ivf_pq":        "tdsqlmultimodalivfpq",
    "ivf_sq":        "tdsqlmultimodalivfsq",
    "ivf_rq":        "tdsqlmultimodalivfrq",
    "ivf_hnsw_flat": "tdsqlmultimodalivfhnswflat",
    "ivf_hnsw_sq":   "tdsqlmultimodalivfhnswsq",
    "ivf_hnsw_pq":   "tdsqlmultimodalivfhnswpq",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # Connection
    g = p.add_argument_group("TDSQL connection (env fallbacks: TDSQL_HOST/PORT/USER/PASSWORD)")
    g.add_argument("--host", default=os.environ.get("TDSQL_HOST", "127.0.0.1"))
    g.add_argument("--port", type=int, default=int(os.environ.get("TDSQL_PORT", "10046")))
    g.add_argument("--user", default=os.environ.get("TDSQL_USER", "root"))
    g.add_argument("--password", default=os.environ.get("TDSQL_PASSWORD", ""))

    # Case + index selection
    g = p.add_argument_group("case + index")
    g.add_argument("--case", required=True, choices=sorted(CASES.keys()),
                   help="Dataset case (mapped to vdbbench --case-type)")
    g.add_argument("--index", required=True, choices=sorted(INDEXES.keys()),
                   help="Lance index family")

    # Load
    g = p.add_argument_group("load")
    g.add_argument("--load-concurrency", type=int, default=10,
                   help="Parallel insert workers (default 10)")
    g.add_argument("--num-per-batch", type=int, default=None,
                   help="Rows per INSERT batch (default: auto by dim)")
    g.add_argument("--skip-load", action="store_true",
                   help="Don't load data; assumes table already exists")
    g.add_argument("--no-drop-old", action="store_true",
                   help="Keep existing table when loading (default: drop + reload)")
    g.add_argument("--rebuild-index", action="store_true",
                   help="With --skip-load, force CREATE INDEX over existing table")

    # Index params
    g = p.add_argument_group("index parameters")
    g.add_argument("--num-partitions", type=int, default=64,
                   help="IVF partitions (default 64)")
    g.add_argument("--nprobs", type=int, default=16,
                   help="IVF probes at search time (default 16)")
    g.add_argument("--m", type=int, default=32,
                   help="HNSW M (hnsw families only, default 32)")
    g.add_argument("--ef-construction", type=int, default=300,
                   help="HNSW ef_construction (hnsw families only, default 300)")
    g.add_argument("--refine-factor", type=int, default=None,
                   help="Refine factor for *_sq/*_pq (optional)")

    # Search
    g = p.add_argument_group("search")
    g.add_argument("--num-concurrency", default="1,5,10",
                   help="Concurrent search worker counts, comma list (default 1,5,10)")
    g.add_argument("--concurrency-duration", type=int, default=45,
                   help="Seconds per concurrency level (default 45)")
    g.add_argument("--k", type=int, default=100, help="Top-k (default 100)")
    g.add_argument("--skip-serial", action="store_true",
                   help="Skip serial search (no recall)")
    g.add_argument("--skip-concurrent", action="store_true",
                   help="Skip concurrent search (no QPS)")
    g.add_argument("--serial-limit", type=int, default=0,
                   help="Truncate serial test set to N queries (0 = full set)")

    # Prewarm
    g = p.add_argument_group("prewarm")
    g.add_argument("--no-prewarm", action="store_true",
                   help="Disable TDSQL prewarm probes")
    g.add_argument("--prewarm-probes", type=int, default=100,
                   help="Random-vector probes per worker init (default 100)")

    # TMPDIR / housekeeping
    g = p.add_argument_group("scratch + housekeeping")
    g.add_argument("--tmpdir", default="/data/tmp/lance_scratch",
                   help="Lance scratch dir (set TMPDIR/TMP/TEMP for vdbbench)")
    g.add_argument("--task-label", default=None,
                   help="Task label (default: <case>_<index>_<timestamp>)")
    g.add_argument("--db-label", default=None,
                   help="DB label (default: tdsql_<case>_<index>)")
    g.add_argument("--out-dir", default=None,
                   help="Output dir for runner artifacts (default: /data/tmp/vdbbench_tdsql_<label>)")
    g.add_argument("--venv-python", default=None,
                   help="Path to python interpreter (default: current sys.executable)")

    # Misc
    p.add_argument("--dry-run", action="store_true",
                   help="Print command without executing, plus pre-flight checks")
    p.add_argument("--no-preflight", action="store_true",
                   help="Skip MySQL connectivity check")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Pass-through extra args to vdbbench (use after '--')")

    return p.parse_args()


def auto_label(args: argparse.Namespace) -> tuple[str, str]:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = f"{args.case}_{args.index}"
    task_label = args.task_label or f"tdsql_{short}_{ts}"
    db_label = args.db_label or f"tdsql_{short}"
    return task_label, db_label


def preflight(args: argparse.Namespace) -> None:
    if args.no_preflight:
        return
    try:
        import mysql.connector  # type: ignore
    except ImportError:
        print("preflight: mysql.connector not available, skipping connectivity probe",
              file=sys.stderr)
        return
    try:
        conn = mysql.connector.connect(
            host=args.host, port=args.port,
            user=args.user, password=args.password,
            buffered=True, connection_timeout=10,
        )
        cur = conn.cursor()
        cur.execute("SELECT @@version")
        ver = cur.fetchone()[0]
        cur.close()
        conn.close()
        print(f"preflight: connected to {args.host}:{args.port}, version={ver}")
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"preflight FAILED: cannot connect to {args.host}:{args.port} as {args.user!r}: {e}\n"
            "Use --no-preflight to skip this check.")


def ensure_tmpdir(path: str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    try:
        p.chmod(0o1777)  # sticky, world-writable
    except PermissionError:
        pass
    free_gb = shutil.disk_usage(path).free / (1 << 30)
    print(f"tmpdir: {path} ({free_gb:.1f} GB free)")
    if free_gb < 30:
        print(f"WARNING: {path} has < 30 GB free; large CREATE INDEX may fail "
              "with 'No space left on device'.", file=sys.stderr)


def build_cmd(args: argparse.Namespace, task_label: str, db_label: str) -> list[str]:
    case_type, dim, default_batch = CASES[args.case]
    subcmd = INDEXES[args.index]
    if args.num_per_batch is None:
        args.num_per_batch = default_batch

    python = args.venv_python or sys.executable
    cmd = [
        python, "-m", "vectordb_bench.cli.vectordbbench", subcmd,
        "--host", args.host,
        "--port", str(args.port),
        "--user", args.user,
        "--password", args.password,
        "--case-type", case_type,
        "--metric-type", "COSINE",
        "--num-partitions", str(args.num_partitions),
        "--task-label", task_label,
        "--db-label", db_label,
        "--num-concurrency", args.num_concurrency,
        "--concurrency-duration", str(args.concurrency_duration),
        "--k", str(args.k),
        "--load-concurrency", str(args.load_concurrency),
        "--num-per-batch", str(args.num_per_batch),
    ]

    # Stages
    if not args.skip_load:
        cmd.append("--load")
        if not args.no_drop_old:
            cmd.append("--drop-old")
        else:
            cmd.append("--skip-drop-old")
    else:
        cmd.append("--skip-load")
        cmd.append("--skip-drop-old")
        if args.rebuild_index:
            cmd.append("--rebuild-index")

    if not args.skip_serial:
        cmd.append("--search-serial")
    else:
        cmd.append("--skip-search-serial")

    if not args.skip_concurrent:
        cmd.append("--search-concurrent")
    else:
        cmd.append("--skip-search-concurrent")

    # Index-family-specific params
    if args.index.startswith("ivf_hnsw"):
        cmd += ["--m", str(args.m), "--ef-construction", str(args.ef_construction)]

    if args.index in ("ivf_sq", "ivf_pq", "ivf_rq", "ivf_hnsw_sq",
                       "ivf_hnsw_pq", "ivf_hnsw_flat"):
        cmd += ["--nprobs", str(args.nprobs)]

    if args.refine_factor is not None:
        cmd += ["--refine-factor", str(args.refine_factor)]

    if args.extra:
        cmd += args.extra

    return cmd


def make_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["TMPDIR"] = args.tmpdir
    env["TMP"] = args.tmpdir
    env["TEMP"] = args.tmpdir
    env["TDSQL_MULTIMODAL_PREWARM"] = "0" if args.no_prewarm else "1"
    env["TDSQL_MULTIMODAL_PREWARM_PROBES"] = str(args.prewarm_probes)
    if args.serial_limit > 0:
        env["VDBBENCH_SERIAL_TEST_LIMIT"] = str(args.serial_limit)
    return env


def main() -> int:
    args = parse_args()
    task_label, db_label = auto_label(args)
    out_dir = Path(args.out_dir or f"/data/tmp/vdbbench_{task_label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_tmpdir(args.tmpdir)
    preflight(args)

    cmd = build_cmd(args, task_label, db_label)
    env = make_env(args)

    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    cmd_file = out_dir / "command.sh"
    log_file = out_dir / "run.log"

    cmd_file.write_text(
        "#!/bin/bash\n"
        "# Auto-generated by run_tdsql_benchmark.py\n"
        f"# Generated: {dt.datetime.now().isoformat()}\n\n"
        f"export TMPDIR={shlex.quote(args.tmpdir)}\n"
        f"export TMP={shlex.quote(args.tmpdir)}\n"
        f"export TEMP={shlex.quote(args.tmpdir)}\n"
        f"export TDSQL_MULTIMODAL_PREWARM={env['TDSQL_MULTIMODAL_PREWARM']}\n"
        f"export TDSQL_MULTIMODAL_PREWARM_PROBES={env['TDSQL_MULTIMODAL_PREWARM_PROBES']}\n"
        + (f"export VDBBENCH_SERIAL_TEST_LIMIT={args.serial_limit}\n"
           if args.serial_limit > 0 else "")
        + f"\ncd {shlex.quote(str(REPO_ROOT))}\n{cmd_str}\n"
    )
    cmd_file.chmod(0o755)

    print()
    print("=" * 70)
    print(f"case={args.case} ({CASES[args.case][0]}, dim={CASES[args.case][1]})")
    print(f"index={args.index} ({INDEXES[args.index]})")
    print(f"target={args.host}:{args.port} user={args.user}")
    print(f"task_label={task_label}  db_label={db_label}")
    print(f"out_dir={out_dir}")
    print(f"cmd_file={cmd_file}")
    print(f"log_file={log_file}")
    print(f"TMPDIR={args.tmpdir}")
    print(f"prewarm={'off' if args.no_prewarm else f'on ({args.prewarm_probes} probes)'}")
    if args.serial_limit > 0:
        print(f"serial_limit={args.serial_limit} queries")
    print("=" * 70)
    print()
    print(f"command:\n  {cmd_str}\n")

    if args.dry_run:
        print("dry-run: not executing")
        return 0

    started = dt.datetime.now()
    print(f"start: {started.isoformat()}")
    print(f"streaming output to {log_file}")
    print()

    with log_file.open("w") as logf:
        logf.write(f"# {cmd_str}\n# started {started.isoformat()}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
        except KeyboardInterrupt:
            print("\ninterrupted, terminating child...")
            proc.terminate()
            proc.wait(timeout=30)
            return 130
        rc = proc.wait()

    finished = dt.datetime.now()
    wall = (finished - started).total_seconds()
    print(f"\nend: {finished.isoformat()}  rc={rc}  wall={wall:.1f}s")
    print(f"log: {log_file}")
    print("result JSON: vectordb_bench/results/TDSQLMultimodal/result_*"
          f"{task_label}*_tdsqlmultimodal.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
