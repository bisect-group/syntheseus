"""Fine-tune MEGAN on a curated split, warm-started from the released USPTO-50K checkpoint.

MEGAN predicts a sequence of graph edits drawn from an action vocabulary built from the training
data. The enzymatic actions are not the USPTO-50K ones, so, as for LocalRetro's templates, the
layers whose shape follows the vocabulary cannot be transferred: the warm start copies every
released weight whose shape still matches and leaves the rest freshly initialised.

MEGAN resumes from `model.pt` in its save directory, reading the model and optimiser state from it,
so the warm start is written in that form with a fresh optimiser and epoch -1.

Hyper-parameters come from the released `config.gin` unchanged (batch 4, learning rate 1e-4, 20k
training samples per epoch, early stopping after 16 epochs without progress). Only `max_n_epochs`
is lowered from the published 200, to bound the wall-clock of a shared GPU; early stopping decides
when training actually ends.

    $PY -m emulator_bench.finetune_megan --split random_splits_grouped_rxn_key
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from emulator_bench.common import OUT_ROOT, atomic_write_text, discover_splits
from emulator_bench.finetune_data import FINETUNE_DIR

RELEASED = "checkpoints/syntheseus_default/MEGAN_backward"

# The authors' USPTO-MIT forward featuriser (`src/config.py`): 8 generation steps, canonical DFS
# action order, and `forward=True`, which reads `substrates` as the source and `product` as the
# target and leads the action priorities with bond additions instead of deletions.
FORWARD_FEATURIZER = "megan_for_8_dfs_cano"
FEATURIZER_RE = re.compile(r'train_megan\.featurizer_key\s*=\s*"([^"]+)"')


def megan_root() -> Path:
    import megan

    return Path(list(megan.__path__)[0]).resolve()


def child_env(workdir: Path) -> dict:
    """MEGAN's scripts import `src.*` from its package root and read DATA_DIR from the environment."""
    root = megan_root()
    return dict(
        os.environ,
        PYTHONHASHSEED="0",
        DATA_DIR=str(workdir / "data"),
        # src/__init__ builds CONFIGS_DIR from PROJECT_ROOT eagerly, so both must be set
        PROJECT_ROOT=str(root),
        CONFIG_DIR=str(root / "configs"),
        PYTHONPATH=os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[1]),
                str(root),
                os.environ.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep),
    )


NO_REACTION_CLASS = 1


def write_megan_csv(src: Path, dest: Path) -> None:
    """Copy a curated CSV, writing an integer reaction class.

    MEGAN's featuriser stores a reaction type per sample in a sparse matrix, so the `UNK` of the
    curated files makes it fail (`scipy.sparse does not support dtype <U21`). USPTO-50K uses the
    integers 1-10 there. These reactions have no class, and the released `config.gin` sets
    `Megan.reaction_type_given = False`, so the model never reads the value; a constant integer
    keeps the featuriser numeric and tells the model nothing.
    """
    import pandas as pd

    table = pd.read_csv(src)
    table["class"] = NO_REACTION_CLASS
    table.to_csv(dest, index=False)


def run_step(name: str, cmd: list[str], cwd: Path, log: Path, env: dict) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode:
        raise SystemExit(f"{name} failed ({proc.returncode}); see {log}")
    seconds = time.time() - t0
    print(f"{name}: {seconds:.0f} s", flush=True)
    return seconds


def warm_start(workdir: Path, save_path: Path, released: Path, config: Path, env: dict) -> dict:
    """Write save_path/model.pt: released weights where the shape fits, fresh action heads."""
    script = f"""
import json, re, sys, torch, gin
sys.path.insert(0, {str(megan_root())!r})
from src.config import get_featurizer
from src.get_instances import get_dataset
from src.model.megan import Megan
from src.utils import load_state_dict

# `train_megan` lives in bin/train.py, which is not imported here, so its bindings are skipped;
# the Megan.* bindings this needs are registered by importing Megan above.
gin.parse_config_file({str(config)!r}, skip_unknown=True)
text = open({str(config)!r}).read()
featurizer_key = re.search(r'train_megan\\.featurizer_key\\s*=\\s*"([^"]+)"', text).group(1)
dataset = get_dataset("uspto_50k")
featurizer = get_featurizer(featurizer_key)
vocab = featurizer.get_actions_vocabulary(dataset.feat_dir)
model = Megan(n_atom_actions=vocab["n_atom_actions"], n_bond_actions=vocab["n_bond_actions"],
              prop2oh=vocab["prop2oh"])
pretrained = load_state_dict({str(released)!r})["model"]
own = model.state_dict()
copied = {{k: v for k, v in pretrained.items() if k in own and own[k].shape == v.shape}}
own.update(copied)
model.load_state_dict(own)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
torch.save({{"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": -1}},
           {str(save_path / "model.pt")!r})
print(json.dumps({{"tensors_total": len(own), "tensors_from_released": len(copied),
                   "n_atom_actions": vocab["n_atom_actions"],
                   "n_bond_actions": vocab["n_bond_actions"]}}))
"""
    save_path.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=workdir, env=env, capture_output=True, text=True
    )
    if out.returncode:
        raise SystemExit(f"warm start failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--split", required=True, help="Split group or group/name")
    p.add_argument("--seed", type=int, default=0, help="Training seed; 0 keeps the original paths")
    p.add_argument(
        "--direction",
        choices=["retro", "forward"],
        default="retro",
        help="forward trains the reverse task on the same curated rows, with MEGAN's own "
        f"{FORWARD_FEATURIZER} featuriser",
    )
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--max_n_epochs", type=int, default=60, help="Published default is 200")
    p.add_argument("--max_n_nodes", type=int, default=200)
    p.add_argument("--skip_prepare", action="store_true", help="Reuse featurised data on disk")
    args = p.parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    split = next(
        s for s in discover_splits() if f"{s.group}/{s.name}" == args.split or s.group == args.split
    )
    dataset = f"{split.group}_{split.name}"
    suffix = f"_seed{args.seed}" if args.seed else ""
    # Retro keeps its original path; forward is a separate tree beside it, so the two directions
    # never share a featurised dataset (their action vocabularies differ).
    base = args.out_root / FINETUNE_DIR / "MEGAN"
    workdir = (base if args.direction == "retro" else base / "forward") / f"{dataset}{suffix}"
    raw_dir = workdir / "data/uspto_50k"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = args.out_root / FINETUNE_DIR / "LocalRetro" / dataset / "data" / dataset
    for fold in ("train", "val", "test"):
        src = source / f"raw_{fold}.csv"
        if not src.exists():
            raise SystemExit(
                f"Missing {src}. Run: $PY -m emulator_bench.finetune_data --model LocalRetro"
            )
        write_megan_csv(src, raw_dir / f"raw_{fold}.csv")

    released_dir = args.out_root / RELEASED
    config = workdir / "config.gin"
    text = (released_dir / "config.gin").read_text().replace(
        "train_megan.max_n_epochs = 200", f"train_megan.max_n_epochs = {args.max_n_epochs}"
    )  # fmt: skip
    if args.direction == "forward":
        text = FEATURIZER_RE.sub(f'train_megan.featurizer_key = "{FORWARD_FEATURIZER}"', text)
    config.write_text(text)
    featurizer_key = FEATURIZER_RE.search(text).group(1)

    env = child_env(workdir)
    timings = {}
    if not args.skip_prepare:
        timings["seconds_prepare"] = run_step(
            "prepare",
            [
                sys.executable,
                "-m",
                "emulator_bench.megan_prepare",
                "--max_n_nodes",
                str(args.max_n_nodes),
                "--featurizer",
                featurizer_key,
            ],  # fmt: skip
            workdir,
            workdir / "logs/prepare.log",
            env,
        )

    save_path = workdir / "model"
    info = warm_start(workdir, save_path, next(released_dir.glob("*.pt")), config, env)
    print(json.dumps(info), flush=True)

    timings["seconds_training"] = run_step(
        "train",
        [
            sys.executable,
            "-m",
            "emulator_bench.megan_train",
            "--config",
            str(config),
            "--save_path",
            str(save_path),
            "--seed",
            str(args.seed),
        ],  # fmt: skip
        workdir,
        workdir / "logs/train.log",
        env,
    )

    dest = (
        args.out_root / "checkpoints/MEGAN" / f"{args.direction}_finetuned" / f"{dataset}{suffix}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(save_path / "model_best.pt", dest / "model_best.pt")
    shutil.copy(config, dest / "config.gin")
    for feat_dir in save_path.iterdir():  # the featuriser vocabulary MEGAN copied next to the model
        if feat_dir.is_dir() and any(feat_dir.glob("*.json")):
            shutil.copytree(feat_dir, dest / feat_dir.name, dirs_exist_ok=True)
    summary = {
        "model": "MEGAN",
        "direction": args.direction,
        "featurizer_key": featurizer_key,
        "seed": args.seed,
        "dataset": dataset,
        "model_dir": str(dest),
        "max_n_epochs": args.max_n_epochs,
        "max_n_nodes": args.max_n_nodes,
        "warm_start": info,
        **timings,
    }
    atomic_write_text(workdir / "finetune_summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
