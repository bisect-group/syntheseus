"""Compare ranked predictions of two runs row by row (equivalence and environment checks).

Inputs are run folders (predictions.jsonl) or `syntheseus eval-single-step` JSON files.
Reports: rows, top-1 and top-10 list equality, equal top-k accuracy, and the max score difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCORE_KEYS = ("probability", "log_probability", "score")


def load(path: Path) -> list[list[dict]]:
    """Per row: ranked predictions as dicts."""
    if path.is_dir():
        with open(path / "predictions.jsonl") as f:
            return [json.loads(line)["predictions"] for line in f]
    data = json.loads(path.read_text())
    return data["predictions"]


def key(pred: dict) -> str:
    """Reactants and products of a prediction. The input side is equal in every row."""
    sides = []
    for side in ("reactants", "products"):
        items = pred.get(side) or []
        items = items if isinstance(items, list) else [items]
        sides.append(".".join(sorted(m["smiles"] for m in items)))
    return ">>".join(sides)


def score(pred: dict) -> float | None:
    for k in SCORE_KEYS:
        if k in pred["metadata"]:
            return pred["metadata"][k]
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    rows_a, rows_b = load(args.a), load(args.b)
    n = min(len(rows_a), len(rows_b))
    same_list = same_top1 = 0
    max_diff = 0.0
    for ra, rb in zip(rows_a[:n], rows_b[:n]):
        ka, kb = [key(x) for x in ra[: args.k]], [key(x) for x in rb[: args.k]]
        same_list += ka == kb
        same_top1 += ka[:1] == kb[:1]
        if ka == kb:
            for x, y in zip(ra[: args.k], rb[: args.k]):
                sx, sy = score(x), score(y)
                if sx is not None and sy is not None:
                    max_diff = max(max_diff, abs(sx - sy))
    print(
        json.dumps(
            {
                "rows_a": len(rows_a),
                "rows_b": len(rows_b),
                "compared": n,
                f"identical_top{args.k}_lists": f"{same_list}/{n}",
                "identical_top1": f"{same_top1}/{n}",
                "max_score_diff_on_identical_lists": max_diff,
            }
        )
    )


if __name__ == "__main__":
    main()
