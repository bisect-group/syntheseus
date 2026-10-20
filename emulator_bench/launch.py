"""Launch the full run in two phases.

1. Predict: for each model x direction, `--workers` shard processes (`emulator_bench.predict`) fill the
   prediction cache for the unique inputs. At most `--gpus` x `--procs_per_gpu` run at once.
2. Evaluate: one CPU process (`emulator_bench.run_model`) per split folder x fold reads the cache.

Shards and finished runs (DONE marker) are resumable, so the launcher can be restarted.
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

from emulator_bench.common import (
    DONE_MARKER,
    FOLDS,
    FORWARD_MODELS,
    OUT_ROOT,
    RETRO_MODELS,
    discover_splits,
    run_dir,
)


def build_jobs(args) -> list[tuple[str, str, object, str]]:
    models = {"retro": RETRO_MODELS, "forward": FORWARD_MODELS}
    splits = [
        s
        for s in discover_splits()
        if not args.splits or f"{s.group}/{s.name}" in args.splits or s.group in args.splits
    ]
    jobs = []
    for direction in args.directions:
        for model, split, fold in itertools.product(models[direction], splits, args.folds):
            if args.models and model not in args.models:
                continue
            if (
                run_dir(args.runs_root or args.out_root, direction, model, split, fold)
                / DONE_MARKER
            ).exists():
                continue
            jobs.append((model, direction, split, fold))
    return jobs


def predict_commands(args, jobs) -> list[tuple[str, list[str]]]:
    """(log name, command) for every shard of every (model, direction) with runs left to do."""
    pairs = sorted({(model, direction) for model, direction, _, _ in jobs})
    return [
        (
            f"predict_{direction}_{model}_shard{i}of{args.workers}",
            [
                sys.executable,
                "-m",
                "emulator_bench.predict",
                "--model",
                model,
                "--direction",
                direction,
                "--folds",
                *args.folds,
                "--shard",
                str(i),
                "--nshards",
                str(args.workers),
                "--out_root",
                str(args.out_root),
            ]
            + (["--limit", str(args.limit)] if args.limit else []),
        )  # fmt: skip
        for model, direction in pairs
        for i in range(args.workers)
    ]


def run_commands(args, jobs) -> list[tuple[str, list[str]]]:
    commands = []
    for model, direction, split, fold in jobs:
        cmd = [
            sys.executable, "-m", "emulator_bench.run_model",
            "--model", model, "--direction", direction, "--split_group", split.group,
            "--split_name", split.name, "--fold", fold, "--out_root", str(args.out_root),
        ]  # fmt: skip
        if args.runs_root:
            cmd += ["--runs_root", str(args.runs_root)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        commands.append((f"{direction}_{model}_{split.group}_{split.name}_{fold}", cmd))
    return commands


def run_pool(args, commands, slots: list[str]) -> int:
    """Run commands with one process per slot. A slot is a GPU id, or 'cpu'. Returns the failure count."""
    running: dict[int, tuple[subprocess.Popen, Path]] = {}
    free = list(range(len(slots)))
    queue = list(commands)
    failed = 0
    while queue or running:
        while queue and free:
            slot = free.pop()
            name, cmd = queue.pop(0)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="" if slots[slot] == "cpu" else slots[slot],
                PYTHONHASHSEED="0",
            )
            log = args.out_root / "logs" / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            running[slot] = (
                subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT),
                log,
            )
        time.sleep(5)
        for slot, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                del running[slot]
                free.append(slot)
                if proc.returncode:
                    failed += 1
                    print(f"FAILED ({proc.returncode}): see {log}", flush=True)
    return failed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument(
        "--runs_root",
        type=Path,
        default=None,
        help="Run folders (default: out_root); for smoke tests",
    )
    p.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids")
    p.add_argument("--procs_per_gpu", type=int, default=4, help="Concurrent predict shards per GPU")
    p.add_argument("--workers", type=int, default=4, help="Predict shards per model and direction")
    p.add_argument("--eval_procs", type=int, default=4, help="Concurrent evaluation processes")
    p.add_argument("--models", nargs="*", default=None, help="Default: all models")
    p.add_argument(
        "--directions", nargs="+", default=["retro", "forward"], choices=["retro", "forward"]
    )
    p.add_argument(
        "--splits", nargs="*", default=None, help="Split group or group/name; default: all on disk"
    )
    p.add_argument(
        "--folds", nargs="+", default=["test"], choices=FOLDS, help="Default: test only (zero-shot)"
    )
    p.add_argument("--limit", type=int, default=None, help="First N rows per run (smoke tests)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    jobs = build_jobs(args)
    predict = predict_commands(args, jobs)
    evaluate = run_commands(args, jobs)
    print(f"{len(predict)} predict shards, {len(evaluate)} runs to do")
    if args.dry_run:
        for _, cmd in predict + evaluate:
            print(" ".join(cmd))
        return

    failed = run_pool(args, predict, [g for g in args.gpus for _ in range(args.procs_per_gpu)])
    if failed:
        raise SystemExit(f"{failed} predict shards failed; no evaluation started")
    failed = run_pool(args, evaluate, ["cpu"] * args.eval_procs)
    print(f"Done with {failed} failed runs")


if __name__ == "__main__":
    main()
