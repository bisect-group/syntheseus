"""Control: run the benchmark's own inference and scoring path on the USPTO-50K test set.

The models' published USPTO-50K numbers are the reference. If this control reproduces them, the
wrapper (model construction, top-N, matching, metrics) is correct, and a low score on the reaction
outcome splits is a property of the data, not of the pipeline.

The retro task is built the same way as for the benchmark: largest product in, the full reactant
set out, atom maps removed, canonical SMILES. USPTO-50K rows have one product and no reagents on
the right, so no curation is needed.

    $PY -m emulator_bench.checks.uspto50k_control --models LocalRetro --limit 500 --device cuda:0

Writes OUT/checks/uspto50k_control/<model>.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from emulator_bench.common import OUT_ROOT, RETRO_MODELS, TOP_KS, atomic_write_text, load_hparams

RAW_TEST = "external/uspto50k/raw_test.csv"


def retro_rows(path: Path, limit: int | None) -> list[dict]:
    """(reactants, product) of each USPTO-50K test reaction, atom maps removed."""
    import pandas as pd
    from rdkit import Chem

    def strip(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)

    df = pd.read_csv(path)
    rows = []
    for rxn in df["reactants>reagents>production"].iloc[:limit]:
        left, _, right = rxn.split(">")
        products = [p for p in strip(right).split(".") if p]
        main = max(products, key=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms())
        rows.append(
            {
                "reactants": [{"smiles": s} for s in strip(left).split(".")],
                "products": [{"smiles": main}],
            }
        )
    return rows


def run(model_name: str, rows: list[dict], out_root: Path, device: str, tmp: Path) -> dict:
    from emulator_bench.run_model import build_model
    from syntheseus.cli.eval_single_step import compute_metrics
    from syntheseus.reaction_prediction.data.dataset import DataFold, DiskReactionDataset
    from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
    from syntheseus.reaction_prediction.utils.misc import set_random_seed

    data_dir = tmp / model_name
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    dataset = DiskReactionDataset(data_dir, sample_cls=ReactionSample)

    hp = load_hparams()
    num_top = hp["num_top_results"]["retro"]
    set_random_seed(0)
    t0 = time.time()
    model = build_model(model_name, "retro", out_root, device)
    results = compute_metrics(
        model,
        dataset=dataset,
        num_dataset_truncation=None,
        num_top_results=num_top,
        fold=DataFold.TEST,
        batch_size=hp["batch_size"],
        include_predictions=False,
    )
    return {
        "model": model_name,
        "n": results.num_samples,
        "num_top_results": num_top,
        **{f"top{k}": results.top_k[k - 1] for k in TOP_KS if k <= num_top},
        "mrr": results.mrr,
        "seconds": time.time() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(RETRO_MODELS))
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    import os

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault(
        "SYNTHESEUS_CACHE_DIR", str(args.out_root / "checkpoints/syntheseus_default")
    )
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    rows = retro_rows(args.out_root / RAW_TEST, args.limit)
    out_dir = args.out_root / "checks/uspto50k_control"
    for name in args.models:
        summary = run(name, rows, args.out_root, args.device, out_dir / "data")
        atomic_write_text(out_dir / f"{name}.json", json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
