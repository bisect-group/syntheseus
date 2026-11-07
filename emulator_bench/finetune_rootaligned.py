"""Fine-tune RootAligned (R-SMILES) on a curated split, warm-started from the released checkpoint.

RootAligned is an OpenNMT transformer over SMILES. Its preprocessing augments every reaction by
writing the product and its reactants from several root atoms, which needs the atom maps, so the
mapped CSVs written for LocalRetro are reused (`emulator_bench.finetune_data --model LocalRetro`).

The released checkpoint carries the authors' own training options, which are read from it and kept:
6+6 transformer layers, 256 units, 8 heads, feed-forward 2048, dropout 0.3, Adam with the noam
schedule, 8000 warm-up steps and 128-sentence batches. Two things necessarily differ:

- The published run trains 1.5M steps from scratch. Fine-tuning starts from those weights, so
  `--train_steps` is far smaller; validation runs regularly and the best checkpoint is kept.
- The published data uses 20 augmentations per reaction. At 20x this split would be 1.3M training
  samples, so `--augmentation` defaults to 5. Inference still uses the released 20 augmentations,
  which is a property of the model wrapper, not of the training data.
- The published batches hold 128 sentences, which suits USPTO-50K's short SMILES. Enzymatic SMILES
  reach a thousand tokens and attention memory grows with the square of the length, so a batch of
  128 of them needs more than the whole GPU. Batching is by token count instead, with gradient
  accumulation keeping the effective batch comparable, and the few sequences beyond
  `--max_seq_length` are dropped.

`--direction forward` trains the reverse task on the same curated rows, using the authors' own
reactant-to-product generator and their forward translate options. No released forward checkpoint
is available here, so it warm starts from the released retro one: the architecture and the SMILES
vocabulary are the same, and only the direction is new.

The enzymatic SMILES contain tokens the USPTO-50K vocabulary lacks, so the vocabulary is rebuilt
and OpenNMT's `update_vocab` maps the released embeddings onto it.

    $PY -m emulator_bench.finetune_rootaligned --split random_splits_grouped_rxn_key
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

RELEASED = "checkpoints/syntheseus_default/RootAligned_backward"
UPSTREAM_DATASET = "USPTO_50K"  # the generators only accept their own dataset names
# product -> reactants for retro, reactants -> product for forward: the same rows either way
GENERATOR = {
    "retro": ("root_aligned.preprocessing.generate_PtoR_data", "PtoR"),
    "forward": ("root_aligned.preprocessing.generate_RtoP_data", "RtoP"),
}
FORWARD_TRANSLATE_YML = Path(__file__).with_name("rsmiles_forward_translate.yml")


def run_step(name: str, cmd: list[str], cwd: Path, log: Path, env: dict | None = None) -> float:
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


def onmt_tool(name: str) -> str:
    """OpenNMT's console scripts sit next to the interpreter, which may not be on PATH."""
    candidate = Path(sys.executable).resolve().parent / name
    return str(candidate) if candidate.exists() else name


def write_config(path: Path, data_dir: Path, workdir: Path, released_ckpt: Path, args) -> None:
    """OpenNMT config: the released model's architecture plus this run's data and schedule."""
    config = f"""# Written by emulator_bench.finetune_rootaligned
save_data: {workdir / "onmt"}
src_vocab: {workdir / "onmt/vocab.src"}
tgt_vocab: {workdir / "onmt/vocab.tgt"}
overwrite: True
share_vocab: False

data:
    corpus_1:
        path_src: {data_dir / "train/src-train.txt"}
        path_tgt: {data_dir / "train/tgt-train.txt"}
    valid:
        path_src: {data_dir / "val/src-val.txt"}
        path_tgt: {data_dir / "val/tgt-val.txt"}

# warm start from the released USPTO-50K checkpoint
train_from: {released_ckpt}
reset_optim: all
update_vocab: True

save_model: {workdir / "onmt/model"}
keep_checkpoint: 5
save_checkpoint_steps: {args.valid_steps}
train_steps: {args.train_steps}
valid_steps: {args.valid_steps}
report_every: 500
early_stopping: {args.early_stopping}
early_stopping_criteria: accuracy

seed: {3435 + args.seed}
world_size: 1
gpu_ranks: [0]

# architecture and optimisation of the released checkpoint
encoder_type: transformer
decoder_type: transformer
enc_layers: 6
dec_layers: 6
heads: 8
rnn_size: 256
word_vec_size: 256
transformer_ff: 2048
dropout: [0.3]
attention_dropout: [0.3]
position_encoding: true
param_init: 0.0
param_init_glorot: true
optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
decay_method: noam
warmup_steps: {args.warmup_steps}
learning_rate: {args.learning_rate}
label_smoothing: 0.0
batch_size: {args.batch_size}
batch_type: {args.batch_type}
normalization: {args.batch_type}
accum_count: [{args.accum_count}]
# enzymatic SMILES are far longer than USPTO's; drop the few beyond what the model can use
transforms: [filtertoolong]
src_seq_length: {args.max_seq_length}
tgt_seq_length: {args.max_seq_length}
max_grad_norm: 0.0
max_generator_batches: 0
"""
    path.write_text(config)


def best_model(onmt_dir: Path) -> Path:
    """The last checkpoint OpenNMT kept, which early stopping leaves as the best one."""
    checkpoints = sorted(onmt_dir.glob("model_step_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise SystemExit(f"No checkpoint under {onmt_dir}; see the training log.")
    return checkpoints[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--split", required=True, help="Split group or group/name")
    p.add_argument("--direction", default="retro", choices=sorted(GENERATOR))
    p.add_argument("--seed", type=int, default=0, help="Training seed; 0 keeps the original paths")
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--augmentation", type=int, default=5)
    p.add_argument("--train_steps", type=int, default=60000)
    p.add_argument("--valid_steps", type=int, default=2500)
    p.add_argument("--early_stopping", type=int, default=5)
    p.add_argument("--warmup_steps", type=int, default=8000)
    p.add_argument("--learning_rate", type=float, default=1.0, help="noam scale, as published")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--batch_type", default="tokens", choices=["tokens", "sents"])
    p.add_argument("--accum_count", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=1000)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--skip_data", action="store_true", help="Reuse the augmented data on disk")
    p.add_argument(
        "--data_only", action="store_true", help="Augment only; train later with --skip_data"
    )
    args = p.parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    split = next(
        s for s in discover_splits() if f"{s.group}/{s.name}" == args.split or s.group == args.split
    )
    dataset = f"{split.group}_{split.name}"
    module, data_tag = GENERATOR[args.direction]
    suffix = f"_seed{args.seed}" if args.seed else ""
    workdir = args.out_root / FINETUNE_DIR / "RootAligned" / args.direction / f"{dataset}{suffix}"
    raw_dir = workdir / "dataset" / UPSTREAM_DATASET
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = args.out_root / FINETUNE_DIR / "RootAligned" / dataset / "data" / dataset
    for fold in ("train", "val", "test"):
        src = source / f"raw_{fold}.csv"
        if not src.exists():
            raise SystemExit(
                f"Missing {src}. Run: $PY -m emulator_bench.finetune_data --model RootAligned"
            )
        shutil.copy(src, raw_dir / f"raw_{fold}.csv")

    env = dict(os.environ, PYTHONHASHSEED="0")
    timings = {}
    data_dir = workdir / "dataset" / f"{UPSTREAM_DATASET}_{data_tag}_aug{args.augmentation}"
    if not args.skip_data:
        timings["seconds_augmentation"] = run_step(
            "augment",
            [
                sys.executable,
                "-m",
                module,
                "-dataset",
                UPSTREAM_DATASET,
                "-augmentation",
                str(args.augmentation),
                "-processes",
                str(args.processes),
                # a different seed draws different augmentation roots, which is part of the
                # variation between training runs this benchmark reports
                "-seed",
                str(33 + args.seed),
            ],  # fmt: skip
            workdir,
            workdir / "logs/augment.log",
            env,
        )

    if args.data_only:
        print(json.dumps({"dataset": dataset, "data_dir": str(data_dir), **timings}), flush=True)
        return

    config = workdir / "onmt_config.yml"
    (workdir / "onmt").mkdir(parents=True, exist_ok=True)
    write_config(config, data_dir, workdir, next((args.out_root / RELEASED).glob("*.pt")), args)
    timings["seconds_vocab"] = run_step(
        "build vocab",
        [onmt_tool("onmt_build_vocab"), "-config", str(config), "-n_sample", "-1"],
        workdir,
        workdir / "logs/vocab.log",
        env,
    )
    timings["seconds_training"] = run_step(
        "train",
        [onmt_tool("onmt_train"), "-config", str(config)],
        workdir,
        workdir / "logs/train.log",
        env,
    )

    best = best_model(workdir / "onmt")
    dest = (
        args.out_root
        / "checkpoints/RootAligned"
        / f"{args.direction}_finetuned"
        / f"{dataset}{suffix}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    for stale in list(dest.glob("*.pt")) + list(dest.glob("*.yml")):
        stale.unlink()  # syntheseus needs exactly one checkpoint and one config
    shutil.copy(best, dest / "model.pt")
    released_yml = (
        FORWARD_TRANSLATE_YML
        if args.direction == "forward"
        else next((args.out_root / RELEASED).glob("*.yml"))
    )
    shutil.copy(released_yml, dest / "translate.yml")
    summary = {
        "model": "RootAligned",
        "direction": args.direction,
        "seed": args.seed,
        "dataset": dataset,
        "model_dir": str(dest),
        "best_checkpoint": str(best),
        "augmentation": args.augmentation,
        "train_steps": args.train_steps,
        "valid_steps": args.valid_steps,
        **timings,
    }
    atomic_write_text(workdir / "finetune_summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
