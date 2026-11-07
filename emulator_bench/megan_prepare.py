"""MEGAN's dataset acquisition and featurisation for a curated split.

MEGAN reads `DATA_DIR/uspto_50k/raw_{train,val,test}.csv` in the same
`reactants>reagents>production` form the other wrappers use, so the curated CSVs are dropped there
and MEGAN's own `uspto_50k` dataset class reads them. Its `complete_mappings` numbers any atom that
carries no map, so the products whose new atoms are deliberately left unmapped need no special
handling here.

One property of that class is overridden: `max_n_nodes` is 100 for USPTO-50K, which is below the
size of many enzymatic products, and the featuriser allocates its node and adjacency matrices from
it. It is raised to the 200 that MEGAN's USPTO-MIT dataset uses and that `default_hparams.json`
already passes at inference, so training and inference agree.

    $PY -m emulator_bench.megan_prepare --data_dir <dir> --featurizer megan_16_bfs_randat
"""

from __future__ import annotations

import argparse

MAX_N_NODES = 200


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--featurizer", default="megan_16_bfs_randat")
    p.add_argument("--dataset_key", default="uspto_50k")
    p.add_argument("--max_n_nodes", type=int, default=MAX_N_NODES)
    p.add_argument("--skip_acquire", action="store_true")
    args = p.parse_args()

    from src.config import get_featurizer
    from src.datasets.uspto_50k import Uspto50k
    from src.get_instances import get_dataset

    original = Uspto50k.meta_info.fget

    def meta_info(self):
        return {**original(self), "max_n_nodes": args.max_n_nodes}

    Uspto50k.meta_info = property(meta_info)

    dataset = get_dataset(args.dataset_key)
    if not args.skip_acquire:
        print(f"acquiring {args.dataset_key}", flush=True)
        dataset.acquire()
    print(f"featurising with {args.featurizer}", flush=True)
    get_featurizer(args.featurizer).featurize_dataset(dataset)
    print("done", flush=True)


if __name__ == "__main__":
    main()
