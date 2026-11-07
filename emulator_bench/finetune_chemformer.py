"""Fine-tune Chemformer on a curated split, warm-started from the released USPTO-50K checkpoint.

`--direction forward` trains the reverse task, reactants -> product, on the same curated rows and
the same pickle: Chemformer's dataset swaps the two sides itself. The released forward checkpoints
are USPTO-MIT files the authors distribute by hand and are not present here, so a forward run warm
starts from the released *backward* USPTO-50K checkpoint instead. Architecture and tokeniser are
identical, so this transfers the chemistry the model already knows, and only the direction is new.

Chemformer is a sequence model, so the whole released checkpoint transfers: `molbart.fine_tune`
takes it as `--model_path` and continues training on the enzymatic data. Nothing is re-initialised,
unlike the template models whose output layer depends on the template set.

Its trainer keeps the checkpoint with the best `val_molecular_accuracy`, which is the val-based
model selection this benchmark uses in place of a hyper-parameter search.

Two upstream settings do not fit a 64k-reaction dataset and are overridden, with the published
learning rate and cycle schedule kept:

- `acc_batches` defaults to 8, which with the default batch size leaves ~125 optimiser steps per
  epoch, so the 8000-step warm-up would never finish. The effective batch is formed from a smaller
  `batch_size` and `acc_batches` that fit the GPU instead.
- `warm_up_steps` is set as a fraction of the real schedule length rather than the default 8000.

Chemformer's trainer targets PyTorch Lightning 1.x; `emulator_bench.chemformer_train` runs the same
model and data module under Lightning 2 (see that module for the renamed pieces).

    $PY -m emulator_bench.finetune_chemformer --split random_splits_grouped_rxn_key
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from emulator_bench.common import OUT_ROOT, atomic_write_text, discover_splits
from emulator_bench.finetune_data import FINETUNE_DIR

RELEASED = "checkpoints/syntheseus_default/Chemformer_backward"
VOCAB = "bart_vocab_downstream.txt"


def chemformer_root() -> Path:
    """Chemformer is a namespace package, so it has no __file__; take its path entry."""
    import chemformer

    return Path(list(chemformer.__path__)[0]).resolve()


def best_checkpoint(log_dir: Path) -> Path:
    """The checkpoint the trainer kept for the best validation molecular accuracy."""
    candidates = [p for p in log_dir.rglob("*.ckpt") if p.name != "last.ckpt"]
    if not candidates:
        raise SystemExit(f"No checkpoint under {log_dir}; see the training log.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--split", required=True, help="Split group or group/name")
    p.add_argument("--direction", default="retro", choices=["retro", "forward"])
    p.add_argument("--seed", type=int, default=0, help="Training seed; 0 keeps the original paths")
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--acc_batches", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Chemformer default")
    p.add_argument("--warm_up_fraction", type=float, default=0.08)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument(
        "--limit_val_batches",
        type=float,
        default=0.25,
        help="Fraction of the val set decoded each epoch for model selection",
    )
    args = p.parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    split = next(
        s for s in discover_splits() if f"{s.group}/{s.name}" == args.split or s.group == args.split
    )
    dataset = f"{split.group}_{split.name}"
    task = "backward_prediction" if args.direction == "retro" else "forward_prediction"
    suffix = f"_seed{args.seed}" if args.seed else ""
    workdir = args.out_root / FINETUNE_DIR / "Chemformer" / args.direction / f"{dataset}{suffix}"
    # one pickle serves both directions: Chemformer's dataset swaps the roles itself
    data_path = args.out_root / FINETUNE_DIR / "Chemformer" / dataset / f"{dataset}.pickle"
    if not data_path.exists():
        raise SystemExit(
            f"Missing {data_path}. Run: "
            "$PY -m emulator_bench.finetune_data --model Chemformer --skip_maps"
        )
    released = next((args.out_root / RELEASED).glob("*.ckpt"))
    root = chemformer_root()
    log_dir = workdir / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    n_train = int((pd.read_pickle(data_path)["set"] == "train").sum())
    steps_per_epoch = max(1, n_train // (args.batch_size * args.acc_batches))
    warm_up = max(100, int(steps_per_epoch * args.epochs * args.warm_up_fraction))
    print(f"{n_train} training rows, {steps_per_epoch} steps/epoch, warm-up {warm_up}", flush=True)

    cmd = [
        sys.executable, "-m", "emulator_bench.chemformer_train",
        "--task", task,
        "--data_path", str(data_path),
        "--model_path", str(released),
        "--vocab_path", str(root / VOCAB),
        "--log_dir", str(log_dir),
        "--chemformer_root", str(root),
        "--limit_val_batches", str(args.limit_val_batches),
        "--seed", str(73 + args.seed),  # 73 is Chemformer's own default
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--acc_batches", str(args.acc_batches),
        "--lr", str(args.learning_rate),
        "--warm_up_steps", str(warm_up),
        "--gpus", str(args.gpus),
    ]  # fmt: skip
    log = workdir / "logs/train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"train: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=dict(
                os.environ,
                PYTHONHASHSEED="0",
                PYTHONPATH=os.pathsep.join(
                    [
                        str(Path(__file__).resolve().parents[1]),
                        str(root),
                        os.environ.get("PYTHONPATH", ""),
                    ]
                ).rstrip(os.pathsep),
            ),
        )
    seconds = time.time() - t0
    if proc.returncode:
        raise SystemExit(f"training failed ({proc.returncode}); see {log}")

    best = best_checkpoint(log_dir)
    dest = (
        args.out_root
        / "checkpoints/Chemformer"
        / f"{args.direction}_finetuned"
        / f"{dataset}{suffix}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("*.ckpt"):
        stale.unlink()  # syntheseus needs exactly one checkpoint in the directory
    shutil.copy(best, dest / "model.ckpt")
    summary = {
        "model": "Chemformer",
        "direction": args.direction,
        "seed": args.seed,
        "dataset": dataset,
        "model_dir": str(dest),
        "best_checkpoint": str(best),
        "seconds_training": seconds,
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "acc_batches": args.acc_batches,
        "learning_rate": args.learning_rate,
        "warm_up_steps": warm_up,
        "train_rows": n_train,
    }
    atomic_write_text(workdir / "finetune_summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
