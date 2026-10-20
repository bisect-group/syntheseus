"""Predict every unique input of the split files once and cache the result (one shard per process).

Shard i of n takes inputs i, i+n, ... of the sorted unique inputs. Finished batches are kept,
so a restarted shard continues where it stopped.
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
    p.add_argument("--num_top_results", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
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
    folder = cache_dir(args.out_root, args.model, args.direction, num_top)

    inputs = unique_inputs(args.out_root, args.model, args.direction, args.folds, args.limit)
    done, _ = load_cache(folder)
    todo = [k for k in list(inputs)[args.shard :: args.nshards] if k not in done]
    print(
        f"shard {args.shard}/{args.nshards}: {len(todo)} to do, {len(inputs)} unique inputs",
        flush=True,
    )
    if not todo:
        return

    set_random_seed(0)  # as in syntheseus eval-single-step
    t0 = time.time()
    model = build_model(args.model, args.direction, args.out_root, args.device)
    print(f"model load {time.time() - t0:.1f} s", flush=True)
    t_start = time.time()
    for i in range(0, len(todo), batch_size):
        keys = todo[i : i + batch_size]
        t0 = time.time()
        predictions = model([inputs[k] for k in keys], num_results=num_top)
        save_batch(folder, keys, predictions, time.time() - t0)
        n = i + len(keys)
        print(f"{n}/{len(todo)} inputs, {n / (time.time() - t_start):.2f} inputs/s", flush=True)
    print(json.dumps({"shard": args.shard, "inputs": len(todo), "seconds": time.time() - t_start}))


if __name__ == "__main__":
    main()
