"""Chemformer's fine-tuning run under PyTorch Lightning 2.

`molbart.fine_tune` targets Lightning 1.x and fails on this environment's Lightning 2 in three
places. The model, data module, optimiser and schedule all come from Chemformer unchanged; only
the parts Lightning renamed are replaced:

- `pl.utilities.seed.seed_everything` moved to `pl.seed_everything`.
- `validation_epoch_end` and `test_epoch_end` were removed in Lightning 2. They received the list
  of step outputs; the replacement hooks take nothing, so the steps' outputs are collected on the
  module and averaged in `on_validation_epoch_end`, exactly as `_avg_dicts` did before. Lightning 2
  refuses to run a model that still defines the old hooks, so they are removed from the class.
- `Trainer(gpus=..., plugins=...)` became `Trainer(accelerator=..., devices=...)`, and
  `precision=16` became `"16-mixed"`.

One behaviour is restored rather than copied: the upstream `ModelCheckpoint` does not pass `mode`,
which selected the *lowest* `val_molecular_accuracy` once Lightning dropped its "auto" mode. The
intent is clearly the highest accuracy, so `mode="max"` is set.

Validation decodes molecules, which costs far more than a training step, so `--limit_val_batches`
samples a fraction of the validation set for per-epoch model selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def patch_lightning_hooks(module_cls) -> None:
    """Replace the Lightning 1 epoch-end hooks with their Lightning 2 equivalents."""
    original_validation_step = module_cls.validation_step

    def on_validation_epoch_start(self):
        self._collected_val_outputs = []

    def validation_step(self, batch, batch_idx):
        outputs = original_validation_step(self, batch, batch_idx)
        self._collected_val_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        collected = getattr(self, "_collected_val_outputs", [])
        if collected:
            self._log_dict(self._avg_dicts(collected))
            collected.clear()

    module_cls.on_validation_epoch_start = on_validation_epoch_start
    module_cls.validation_step = validation_step
    module_cls.on_validation_epoch_end = on_validation_epoch_end
    for removed in ("validation_epoch_end", "test_epoch_end"):
        if removed in vars(module_cls):
            delattr(module_cls, removed)


def build_trainer(args, log_dir: Path):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    checkpoint = ModelCheckpoint(
        monitor="val_molecular_accuracy",
        mode="max",  # upstream omits this; Lightning 2 would otherwise keep the worst checkpoint
        save_top_k=1,
        save_last=False,
    )
    return pl.Trainer(
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus or 1,
        logger=TensorBoardLogger(str(log_dir), name=args.task),
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint],
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.clip_grad,
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=1,
        precision="16-mixed",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_path", required=True, help="Released checkpoint to warm start from")
    p.add_argument("--vocab_path", required=True)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--chemformer_root", required=True)
    p.add_argument(
        "--task",
        default="backward_prediction",
        choices=["backward_prediction", "forward_prediction"],
    )
    p.add_argument("--dataset", default="uspto_50")
    p.add_argument("--model_type", default="bart")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--acc_batches", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--schedule", default="cycle")
    p.add_argument("--warm_up_steps", type=int, default=8000)
    p.add_argument("--augment", default="None")
    p.add_argument("--aug_prob", type=float, default=0.5)
    p.add_argument("--train_tokens", type=int, default=None)
    p.add_argument("--num_buckets", type=int, default=24)
    p.add_argument("--limit_val_batches", type=float, default=0.25)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--num_nodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=73, help="Chemformer's own default is 73")
    p.add_argument(
        "--num_workers", type=int, default=8, help="Dataloader workers; default is all CPUs"
    )
    args = p.parse_args()

    sys.path.insert(0, args.chemformer_root)  # Chemformer's modules import each other relatively
    import molbart.util as util
    import pytorch_lightning as pl
    from molbart.decoder import DecodeSampler
    from molbart.fine_tune import load_model
    from molbart.models.pre_train import _AbsTransformerModel

    pl.seed_everything(args.seed)
    patch_lightning_hooks(_AbsTransformerModel)

    tokeniser = util.load_tokeniser(args.vocab_path, util.DEFAULT_CHEM_TOKEN_START)
    # forward_prediction reads the same rows the other way round: reactants -> product
    forward = args.task == "forward_prediction"
    dataset = util.build_dataset(args, forward=forward)
    datamodule = util.build_reaction_datamodule(args, dataset, tokeniser, forward=forward)
    # The data module defaults to one worker per visible CPU (112 here), which thrashes the box
    # when several trainings share it. The cap only affects loading speed.
    datamodule._num_workers = args.num_workers
    train_steps = util.calc_train_steps(args, datamodule)
    print(f"train steps: {train_steps}", flush=True)

    sampler = DecodeSampler(tokeniser, util.DEFAULT_MAX_SEQ_LEN)
    model = load_model(
        args, sampler, len(tokeniser), train_steps + 1, tokeniser.vocab[tokeniser.pad_token]
    )
    sampler.max_seq_len = model.max_seq_len

    trainer = build_trainer(args, Path(args.log_dir))
    trainer.fit(model, datamodule=datamodule)
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
