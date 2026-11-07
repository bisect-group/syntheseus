"""Fine-tune LocalRetro on a curated split, warm-started from the released USPTO-50K checkpoint.

LocalRetro predicts a template index, so its output layers are sized by the template set extracted
from the training data. The enzymatic templates are not the USPTO-50K ones, so the classifier heads
cannot be transferred: the warm start copies every released weight whose shape still matches (the
message-passing encoder and the attention stack) and leaves the template heads freshly initialised.

The upstream scripts use paths relative to their working directory, so everything runs inside
`OUT/finetune/LocalRetro/<dataset>/` with this layout:

    data/<dataset>/raw_{train,val,test}.csv   (emulator_bench.finetune_data)
    data/configs/default_config.json          (copied from the released checkpoint)
    scripts/                                  (working directory of the upstream scripts)
    models/LocalRetro_<dataset>.pth           (warm start, then the best checkpoint)

Validation drives early stopping (`--patience`), which is the only use of the val fold. The best
checkpoint is exported as a syntheseus model directory so the usual predict/evaluate path can serve
it unchanged.

    $PY -m emulator_bench.finetune_localretro --split random_splits_grouped_rxn_key --device cuda:0
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

RELEASED = "checkpoints/syntheseus_default/LocalRetro_backward"


def run_step(name: str, cmd: list[str], cwd: Path, log: Path, stdin: str | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            input=stdin,
            text=True,
            # The upstream scripts run from the workdir, so emulator_bench needs its repo root.
            env=dict(
                os.environ,
                PYTHONHASHSEED="0",
                PYTHONPATH=os.pathsep.join(
                    [str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep),
            ),
        )
    if proc.returncode:
        raise SystemExit(f"{name} failed ({proc.returncode}); see {log}")
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)


def warm_start(workdir: Path, dataset: str, released: Path, device: str) -> dict:
    """Write the initial checkpoint: released weights where the shape still fits, fresh heads."""
    import torch
    from local_retro.scripts.models import LocalRetro_model
    from local_retro.scripts.utils import get_configure, init_featurizer

    args = init_featurizer(
        {
            "mode": "train",
            "device": device,
            "config_path": str(workdir / "data/configs/default_config.json"),
            "data_dir": str(workdir / "data" / dataset),
            "model_name": "default",
        }
    )
    config = get_configure(args)
    model = LocalRetro_model(
        node_in_feats=config["in_node_feats"],
        edge_in_feats=config["in_edge_feats"],
        node_out_feats=config["node_out_feats"],
        edge_hidden_feats=config["edge_hidden_feats"],
        num_step_message_passing=config["num_step_message_passing"],
        attention_heads=config["attention_heads"],
        attention_layers=config["attention_layers"],
        AtomTemplate_n=config["AtomTemplate_n"],
        BondTemplate_n=config["BondTemplate_n"],
        activation=config.get("activation", "relu"),
    )
    pretrained = torch.load(next(released.glob("*.pth")), map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    own = model.state_dict()
    copied = {k: v for k, v in pretrained.items() if k in own and own[k].shape == v.shape}
    own.update(copied)
    model.load_state_dict(own)
    path = workdir / "models" / f"LocalRetro_{dataset}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "timestep": 0}, path)
    skipped = sorted(set(own) - set(copied))
    return {
        "tensors_total": len(own),
        "tensors_from_released": len(copied),
        "freshly_initialised": skipped,
        "atom_templates": config["AtomTemplate_n"],
        "bond_templates": config["BondTemplate_n"],
    }


def export_model_dir(workdir: Path, dataset: str, out_root: Path, released: Path) -> Path:
    """A syntheseus model directory: one *.pth, one *.json and data/*.csv."""
    dest = out_root / "checkpoints/LocalRetro/retro_finetuned" / dataset
    (dest / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(workdir / "models" / f"LocalRetro_{dataset}.pth", dest / "LocalRetro.pth")
    shutil.copy(workdir / "data/configs/default_config.json", dest / "default_config.json")
    for name in ("atom_templates.csv", "bond_templates.csv", "template_infos.csv"):
        shutil.copy(workdir / "data" / dataset / name, dest / "data" / name)
    _ = released
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--split", required=True, help="Split group or group/name")
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument(
        "--skip_extract", action="store_true", help="Reuse the extracted templates on disk"
    )
    p.add_argument(
        "--skip_preprocessing", action="store_true", help="Also reuse the labelled data on disk"
    )
    args = p.parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    split = next(
        s for s in discover_splits() if f"{s.group}/{s.name}" == args.split or s.group == args.split
    )
    dataset = f"{split.group}_{split.name}"
    workdir = args.out_root / FINETUNE_DIR / "LocalRetro" / dataset
    released = args.out_root / RELEASED
    logs = workdir / "logs"
    scripts = workdir / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (workdir / "data/saved_graphs").mkdir(parents=True, exist_ok=True)
    (workdir / "data/configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(released / "default_config.json", workdir / "data/configs/default_config.json")
    if not (workdir / "data" / dataset / "raw_train.csv").exists():
        raise SystemExit(
            f"Missing {workdir}/data/{dataset}/raw_train.csv. "
            "Run: $PY -m emulator_bench.finetune_data --model LocalRetro"
        )

    t0 = time.time()
    if not (args.skip_extract or args.skip_preprocessing):
        run_step(
            "extract templates",
            [
                sys.executable,
                "-m",
                "local_retro.preprocessing.Extract_from_train_data",
                "-d",
                dataset,
            ],
            scripts,
            logs / "extract.log",
        )
    if not args.skip_preprocessing:
        run_step(
            "preprocess",
            [sys.executable, "-m", "emulator_bench.localretro_preprocess", "-d", dataset],
            scripts,
            logs / "preprocess.log",
        )
    t_prep = time.time() - t0

    info = warm_start(workdir, dataset, released, args.device)
    print(json.dumps(info | {"freshly_initialised": len(info["freshly_initialised"])}), flush=True)

    t0 = time.time()
    run_step(
        "train",
        [
            sys.executable,
            "-m",
            "local_retro.scripts.Train",
            "-d",
            dataset,
            "-g",
            args.device,
            "-b",
            str(args.batch_size),
            "-n",
            str(args.epochs),
            "-p",
            str(args.patience),
            "-lr",
            str(args.learning_rate),
        ],  # fmt: skip
        scripts,
        logs / "train.log",
        stdin="b\n",  # continue from the warm-start checkpoint
    )
    t_train = time.time() - t0
    dest = export_model_dir(workdir, dataset, args.out_root, released)
    summary = {
        "model": "LocalRetro",
        "dataset": dataset,
        "model_dir": str(dest),
        "seconds_preprocessing": t_prep,
        "seconds_training": t_train,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warm_start": info,
    }
    atomic_write_text(workdir / "finetune_summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "warm_start"}), flush=True)


if __name__ == "__main__":
    main()
