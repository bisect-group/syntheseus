"""Predict every unique input of the split files once and cache the result (one shard per process).

Shard i of n takes inputs i, i+n, ... of the sorted unique inputs. Finished batches are kept,
so a restarted shard continues where it stopped. A batch that runs out of GPU memory is retried one
input at a time (`predict_batch`).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from emulator_bench.common import FOLDS, FORWARD_MODELS, OUT_ROOT, RETRO_MODELS, load_hparams
from emulator_bench.predcache import cache_dir, load_cache, save_batch, unique_inputs

OOM_RETRIES = 3  # waits for memory that other processes on a shared GPU may free
OOM_WAIT_S = 60


def predict_batch(model, inputs: list, keys: list[str], num_top: int):
    """(predictions, keys skipped for lack of GPU memory) of one batch.

    A batch that runs out of GPU memory is retried one input at a time. The models give the same
    output for an input in any batch (README, "Two phases"), so the predictions do not change. An
    input that still does not fit after OOM_RETRIES waits gets no predictions and is recorded as
    skipped; the common test set then leaves it out for every model.
    """
    import torch

    try:
        return model(inputs, num_results=num_top), frozenset()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
    predictions, skipped = [], set()
    for key, x in zip(keys, inputs):
        for attempt in range(OOM_RETRIES + 1):
            try:
                predictions += model([x], num_results=num_top)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt < OOM_RETRIES:
                    time.sleep(OOM_WAIT_S)
        else:
            predictions.append([])
            skipped.add(key)
            print(f"out of GPU memory, skipped: {key}", flush=True)
    return predictions, frozenset(skipped)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True, choices=sorted(set(RETRO_MODELS) | set(FORWARD_MODELS))
    )
    p.add_argument("--direction", required=True, choices=["retro", "forward"])
    p.add_argument("--folds", nargs="+", default=["test"], choices=FOLDS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None, help="First N rows per split file (tests)")
    p.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Split group or group/name to predict; default: every split file on disk",
    )
    p.add_argument("--num_top_results", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0, help="Global and per-input random seed")
    p.add_argument(
        "--model_dir", type=Path, default=None, help="Checkpoint directory (fine-tuned weights)"
    )
    p.add_argument(
        "--tag", default="", help="Prediction cache tag, e.g. finetuned_<dataset>; with --model_dir"
    )
    args = p.parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault(
        "SYNTHESEUS_CACHE_DIR", str(args.out_root / "checkpoints/syntheseus_default")
    )

    from emulator_bench.run_model import build_model
    from syntheseus.reaction_prediction.utils import misc
    from syntheseus.reaction_prediction.utils.misc import set_random_seed

    # RootAligned retro and MHNreact start a pool of cpu_count() workers on every call (112 here).
    # Forking that many costs more than the prediction, so cap the pool size before models are built.
    multiprocessing.cpu_count = lambda: 8
    misc.cpu_count = lambda default=8: 8

    hp = load_hparams()
    mhp = hp["models"][args.model][args.direction]
    num_top = args.num_top_results or mhp.get(
        "num_top_results", hp["num_top_results"][args.direction]
    )
    batch_size = args.batch_size or mhp.get("batch_size", hp["batch_size"])
    folder = cache_dir(args.out_root, args.model, args.direction, num_top, args.seed, args.tag)

    inputs = unique_inputs(
        args.out_root, args.model, args.direction, args.folds, args.limit, args.splits
    )
    done, *_ = load_cache(folder)
    todo = [k for k in list(inputs)[args.shard :: args.nshards] if k not in done]
    print(
        f"shard {args.shard}/{args.nshards}: {len(todo)} to do, {len(inputs)} unique inputs",
        flush=True,
    )
    if not todo:
        return

    set_random_seed(args.seed)  # 0 as in syntheseus eval-single-step
    t0 = time.time()
    model = build_model(
        args.model, args.direction, args.out_root, args.device, args.seed, args.model_dir
    )
    print(f"model load {time.time() - t0:.1f} s", flush=True)
    t_start = time.time()
    for i in range(0, len(todo), batch_size):
        keys = todo[i : i + batch_size]
        t0 = time.time()
        predictions, oom = predict_batch(model, [inputs[k] for k in keys], keys, num_top)
        skipped = (frozenset(keys) & getattr(model, "skipped", frozenset())) | oom
        save_batch(folder, keys, predictions, time.time() - t0, skipped)
        n = i + len(keys)
        print(f"{n}/{len(todo)} inputs, {n / (time.time() - t_start):.2f} inputs/s", flush=True)
    print(json.dumps({"shard": args.shard, "inputs": len(todo), "seconds": time.time() - t_start}))


if __name__ == "__main__":
    main()
