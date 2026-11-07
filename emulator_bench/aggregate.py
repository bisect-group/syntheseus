"""Rebuild OUT/summary_topk.csv, OUT/summary_topk_seeds.csv and OUT/summary_overlap.csv.

summary_topk.csv has one row per run (model, direction, split file, fold, seed). For retro every
metric is on the common test set (emulator_bench.common_set, from rescore.json):

- `top1`, `top10`, `top50`, `mrr`: the headline, stereo-agnostic exact match, one reference per
  reaction. Enzymes are stereospecific while the models are not, so stereochemistry is scored
  separately rather than folded into the headline.
- `stereo_*`: the same, counting a prediction correct only with matching stereochemistry.
- `multiref_*` and `multiref_stereo_*`: one unit per unique product, correct if any reference
  reactant set of that product is matched.
- `all_*`: stereo-aware per-reaction metrics over all curated rows, which is what syntheseus
  itself computed and what the rescore self-check verifies.

For forward runs, which have no rescore.json, `top1`/`top10`/`mrr` remain the stereo-aware values
from syntheseus. summary_topk_seeds.csv has the mean and sample std of every metric over seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from emulator_bench.common import DONE_MARKER, OUT_ROOT, TOP_KS, atomic_write_text, load_hparams
from emulator_bench.rescore import RESCORE_FILE, runs_roots

METRICS = [f"top{k}" for k in TOP_KS] + ["mrr"]
TOPK_COLUMNS = [
    "model",
    "direction",
    "split_group",
    "split_name",
    "fold",
    "seed",
    "n",
    "n_dropped",
    "n_curation_dropped",
    "n_curation_merged",
    "n_multi_product",
    "n_multi_tie",
    "n_skipped",
    "num_top_results",
    *METRICS,
]
# rescore.json variants (retro only): column prefix -> (stereo key, reference key)
RESCORE_VARIANTS = {
    "": ("no_stereo", "per_reaction"),  # headline: stereochemistry ignored
    "stereo_": ("stereo", "per_reaction"),
    "multiref_": ("no_stereo", "multi_reference"),
    "multiref_stereo_": ("stereo", "multi_reference"),
}
RESCORE_COLUMNS = (
    ["n_products", "n_all"]
    + [f"{p}{m}" for p in RESCORE_VARIANTS if p for m in METRICS]
    + [f"all_{m}" for m in METRICS]
)
RUN_KEYS = ["model", "direction", "split_group", "split_name", "fold"]


def topk_table(roots: list[Path]) -> pd.DataFrame:
    rows = []
    dones = [d for root in roots for d in sorted(root.glob(f"*/*/*/*/*/{DONE_MARKER}"))]
    for done in dones:
        run = done.parent
        result = json.loads((run / "result.json").read_text())
        with open(run / "predictions.jsonl") as f:
            n_saved = sum(1 for _ in f)
        if n_saved != result["n"]:
            raise RuntimeError(f"{run}: {n_saved} saved predictions, expected {result['n']}")
        defaults = {"n_skipped": 0, "n_curation_dropped": 0, "n_curation_merged": 0, "seed": 0}
        row = {k: (defaults | result).get(k) for k in TOPK_COLUMNS}
        # Rows removed by the model's own preprocessing filter count as misses.
        total = row["n"] + row["n_dropped"]
        for k in METRICS:
            if row[k] is not None:
                row[k] *= row["n"] / total
        row["n"] = total
        if (run / RESCORE_FILE).exists():
            extra = json.loads((run / RESCORE_FILE).read_text())
            # rescore only sees the rows the model's preprocessing kept; rows it dropped are
            # misses, so every variant is rescaled the same way the headline metrics were.
            kept_fraction = (total - row["n_dropped"]) / total
            common, full = extra["common"], extra["all"]
            row["n_all"] = row["n"]
            row["n"], row["n_products"] = common["n_reactions"], common["n_products"]
            for k, v in full["stereo"]["per_reaction"].items():
                row[f"all_{k}"] = v * kept_fraction
            for prefix, (stereo, ref) in RESCORE_VARIANTS.items():
                for k, v in common[stereo][ref].items():
                    row[f"{prefix}{k}"] = v * kept_fraction
        rows.append(row)
    return pd.DataFrame(rows, columns=TOPK_COLUMNS + RESCORE_COLUMNS)


def seeds_table(topk: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample std over seeds of every metric of each run; n must agree across seeds."""
    metrics = [c for c in topk.columns if c.split("_")[-1] in METRICS]
    rows = []
    for key, part in topk.groupby(RUN_KEYS, sort=True):
        if part["n"].nunique() != 1:
            raise RuntimeError(f"{key}: seeds were scored on different rows {sorted(part['n'])}")
        row = dict(zip(RUN_KEYS, key)) | {
            "seeds": " ".join(str(s) for s in sorted(part["seed"])),
            "n": part["n"].iloc[0],
            "n_products": part["n_products"].iloc[0],
        }
        for m in metrics:
            row[f"{m}_mean"] = part[m].mean()
            row[f"{m}_std"] = part[m].std(ddof=1) if len(part) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


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
        "--runs_root",
        type=Path,
        nargs="+",
        default=None,
        help="Default: out_root and every out_root/seed<N>",
    )
    args = p.parse_args()
    topk = topk_table(args.runs_root or runs_roots(args.out_root))
    atomic_write_text(args.out_root / "summary_topk.csv", topk.to_csv(index=False))
    seeds = seeds_table(topk)
    atomic_write_text(args.out_root / "summary_topk_seeds.csv", seeds.to_csv(index=False))
    print(seeds.to_string())
    overlap = overlap_table(args.out_root)
    if len(overlap):
        atomic_write_text(args.out_root / "summary_overlap.csv", overlap.to_csv(index=False))


if __name__ == "__main__":
    main()
