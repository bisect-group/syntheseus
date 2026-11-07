"""MEGAN's training with a seed.

`bin/train.py` dispatches `train_megan` through gin and never seeds anything, so repeated runs
differ only by whatever the libraries happen to do. This wrapper performs the same two steps that
`dispatch_configurable_command` performs -- resolve the gin config, then call `train_megan` with
the save path -- after seeding Python, NumPy and torch, so a run is reproducible and different
seeds give genuinely different runs.

    $PY -m emulator_bench.megan_train --config <config.gin> --save_path <dir> --seed 1
"""

from __future__ import annotations

import argparse
import random


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from bin.train import train_megan
    from src.utils.dispatch_utils import parse_config_file, parse_config_key

    parse_config_file(parse_config_key(args.config))
    train_megan(save_path=args.save_path)


if __name__ == "__main__":
    main()
