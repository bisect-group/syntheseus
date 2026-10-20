"""Rebuild OUT/summary_topk.csv and OUT/summary_overlap.csv from the saved run folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from emulator_bench.common import DONE_MARKER, OUT_ROOT, atomic_write_text, load_hparams

TOPK_COLUMNS = [
    "model",
    "direction",
    "split_group",
    "split_name",
    "fold",
    "n",
    "n_dropped",
    "n_multi_product",
    "n_multi_tie",
    "top1",
    "top10",
    "mrr",
]


def topk_table(runs_root: Path) -> pd.DataFrame:
    rows = []
    for done in sorted(runs_root.glob(f"*/*/*/*/*/{DONE_MARKER}")):
        run = done.parent
        result = json.loads((run / "result.json").read_text())
        with open(run / "predictions.jsonl") as f:
            n_saved = sum(1 for _ in f)
        if n_saved != result["n"]:
            raise RuntimeError(f"{run}: {n_saved} saved predictions, expected {result['n']}")
        rows.append({k: result[k] for k in TOPK_COLUMNS})
    return pd.DataFrame(rows, columns=TOPK_COLUMNS)


def overlap_table(out_root: Path) -> pd.DataFrame:
    by_corpus = out_root / "summary_overlap_by_corpus.csv"
    if not by_corpus.exists():
        return pd.DataFrame()
    table = pd.read_csv(by_corpus)
    rows = []
    for direction, mapping in load_hparams()["overlap_corpora"].items():
        for model, corpus in mapping.items():
            part = table[table["corpus"] == corpus].copy()
            part.insert(0, "direction", direction)
            part.insert(0, "model", model)
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument(
        "--runs_root", type=Path, default=None, help="Default: out_root/<direction> folders"
    )
    args = p.parse_args()
    runs_root = args.runs_root or args.out_root
    topk = topk_table(runs_root)
    atomic_write_text(args.out_root / "summary_topk.csv", topk.to_csv(index=False))
    print(topk.to_string())
    overlap = overlap_table(args.out_root)
    if len(overlap):
        atomic_write_text(args.out_root / "summary_overlap.csv", overlap.to_csv(index=False))


if __name__ == "__main__":
    main()
