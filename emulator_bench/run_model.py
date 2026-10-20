"""Run one model on one split file. Zero-shot: released checkpoints only, no training."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import zlib
from pathlib import Path

from emulator_bench.common import (
    DONE_MARKER,
    FOLDS,
    FORWARD_MODELS,
    OUT_ROOT,
    RETRO_MODELS,
    atomic_write_text,
    discover_splits,
    load_hparams,
    open_dataset,
    run_dir,
)
from emulator_bench.predcache import CachedModel, cache_dir


def seeded_root_aligned_class():
    """RootAligned retro with one random stream per input, so an output does not depend on its batch."""
    from syntheseus.reaction_prediction.inference import root_aligned

    # The class name picks the default checkpoint, so it stays RootAlignedModel.
    class RootAlignedModel(root_aligned.RootAlignedModel):
        def _get_reactions(self, inputs, num_results, **kwargs):
            results = []
            for mol in inputs:
                random.seed(zlib.crc32(mol.smiles.encode()))
                results += super()._get_reactions([mol], num_results, **kwargs)
            return results

    return RootAlignedModel


def build_model(model: str, direction: str, out_root: Path, device: str):
    from syntheseus.reaction_prediction.inference.config import (
        BackwardModelClass,
        ForwardModelClass,
    )

    hp = load_hparams()["models"][model][direction]
    kwargs = dict(hp.get("model_kwargs", {}))
    # A None model_dir makes syntheseus download its default checkpoint into SYNTHESEUS_CACHE_DIR.
    model_dir = (
        out_root / "checkpoints" / model / direction
        if hp["checkpoint"] == "authors_release"
        else None
    )
    if model == "GLN":
        from emulator_bench.gln_compat import prepare

        prepare(out_root / "external/GLN")
    if direction == "retro":
        cls = (
            seeded_root_aligned_class()
            if model == "RootAligned"
            else BackwardModelClass[model].value
        )
        built = cls(model_dir=model_dir, device=device, **kwargs)
        if model in ("LocalRetro", "RetroKNN"):
            built.model.eval()  # the syntheseus wrapper leaves dropout on
        return built
    if model == "RootAligned":
        from emulator_bench.rsmiles_forward import RootAlignedForwardModel

        return RootAlignedForwardModel(model_dir=model_dir, device=device, **kwargs)
    return ForwardModelClass[model].value(
        model_dir=model_dir, device=device, is_forward=True, **kwargs
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True, choices=sorted(set(RETRO_MODELS) | set(FORWARD_MODELS))
    )
    p.add_argument("--direction", required=True, choices=["retro", "forward"])
    p.add_argument("--split_group", required=True)
    p.add_argument("--split_name", default="all")
    p.add_argument("--fold", required=True, choices=FOLDS)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument(
        "--runs_root", type=Path, default=None, help="Where run folders go (default: out_root)"
    )
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="Run the model on every row instead of reading emulator_bench.predict's cache",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="With --no_cache: cuda:0 (CUDA_VISIBLE_DEVICES picks the GPU) or cpu",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Use only the first N rows (smoke tests)"
    )
    p.add_argument(
        "--num_top_results", type=int, default=None, help="Default from default_hparams.json"
    )
    p.add_argument("--batch_size", type=int, default=None, help="Default from default_hparams.json")
    args = p.parse_args()
    # torch >= 2.6 loads checkpoints with weights_only=True. The released checkpoints are pickles of
    # library classes, so this variable restores the old behaviour. It has no effect on torch < 2.6.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault(
        "SYNTHESEUS_CACHE_DIR", str(args.out_root / "checkpoints/syntheseus_default")
    )

    from syntheseus.cli.eval_single_step import compute_metrics
    from syntheseus.reaction_prediction.data.dataset import DataFold
    from syntheseus.reaction_prediction.utils.misc import set_random_seed

    set_random_seed(0)  # as in syntheseus eval-single-step

    split = next(
        s for s in discover_splits() if s.group == args.split_group and s.name == args.split_name
    )
    out = run_dir(args.runs_root or args.out_root, args.direction, args.model, split, args.fold)
    if (out / DONE_MARKER).exists():
        print(f"Already done: {out}")
        return
    out.mkdir(parents=True, exist_ok=True)

    hp = load_hparams()
    mhp = hp["models"][args.model][args.direction]
    num_top = args.num_top_results or mhp.get(
        "num_top_results", hp["num_top_results"][args.direction]
    )
    batch_size = args.batch_size or mhp.get("batch_size", hp["batch_size"])
    dataset, counts = open_dataset(args.out_root, split, args.fold, args.model, args.direction)
    t0 = time.time()
    if args.no_cache:
        model = build_model(args.model, args.direction, args.out_root, args.device)
    else:
        model = CachedModel(
            cache_dir(args.out_root, args.model, args.direction, num_top), args.direction, num_top
        )
    t_load = time.time() - t0
    t0 = time.time()
    results = compute_metrics(
        model,
        dataset=dataset,
        num_dataset_truncation=args.limit,
        num_top_results=num_top,
        fold=DataFold(args.fold),
        batch_size=batch_size,
        include_predictions=True,
        resumable=True,
        output_dir=out,
    )
    seconds = time.time() - t0
    summary = {
        "model": args.model,
        "direction": args.direction,
        "split_group": split.group,
        "split_name": split.name,
        "fold": args.fold,
        "n": results.num_samples,
        "n_rows_in_file": counts["n_kept"] + counts["n_dropped"],
        "n_dropped": counts["n_dropped"],
        "n_multi_product": counts["n_multi_product"],
        "n_multi_tie": counts["n_multi_tie"],
        "top1": results.top_k[0],
        "top10": results.top_k[9],
        "mrr": results.mrr,
        "num_top_results": num_top,
        "batch_size": batch_size,
        "seconds_inference": seconds,
        "seconds_model_load": t_load,
        "cached": not args.no_cache,
        "device": args.device if args.no_cache else None,
        "limit": args.limit,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    atomic_write_text(out / "result.json", json.dumps(summary, indent=2))
    atomic_write_text(out / DONE_MARKER, "")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
