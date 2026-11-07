"""Atom-map the reactions of a `retro_targets prepare` request file with RXNMapper.

Standalone (pandas, rdkit, rxnmapper), so it runs in any env that has rxnmapper:

    <python> -I emulator_bench/map_reactions.py <requests.parquet> <maps.parquet> [--device cuda]

Each request has `full` (reactants >> all products) and `reduced` (reactants >> main product).
`full` is mapped first; if RXNMapper rejects it (more than 512 tokens or a parse error), `reduced`
is tried. Fragment order is kept (canonicalize_rxns=False), so mapped fragments line up with the
request. Output columns: full, mapped_rxn, method (full, main or failed), confidence, error.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def map_one(mapper, rxn: str):
    out = mapper.get_attention_guided_atom_maps([rxn], canonicalize_rxns=False)[0]
    return out["mapped_rxn"], out["confidence"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("requests", type=Path)
    p.add_argument("maps", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    import torch
    from rdkit import RDLogger
    from rxnmapper import RXNMapper

    RDLogger.DisableLog("rdApp.*")
    torch.set_grad_enabled(False)
    mapper = RXNMapper()
    mapper.model.to(args.device)
    mapper.device = torch.device(args.device)

    req = pd.read_parquet(args.requests)
    rows = []
    t0 = time.time()
    for start in range(0, len(req), args.batch_size):
        batch = req.iloc[start : start + args.batch_size]
        try:
            outs = mapper.get_attention_guided_atom_maps(
                list(batch["full"]), canonicalize_rxns=False
            )
            rows += [
                (f, o["mapped_rxn"], "full", o["confidence"], "")
                for f, o in zip(batch["full"], outs)
            ]
        except Exception:  # a reaction of the batch failed: map them one by one
            for full, reduced in zip(batch["full"], batch["reduced"]):
                errors = []
                for method, rxn in (("full", full), ("main", reduced)):
                    try:
                        mapped, conf = map_one(mapper, rxn)
                        rows.append((full, mapped, method, conf, "; ".join(errors)))
                        break
                    except Exception as e:
                        errors.append(f"{method}: {type(e).__name__}: {str(e)[:200]}")
                else:
                    rows.append((full, "", "failed", float("nan"), "; ".join(errors)))
        done = start + len(batch)
        if done % (args.batch_size * 50) < args.batch_size:
            print(f"{done}/{len(req)} mapped, {done / (time.time() - t0):.1f} rxn/s", flush=True)
    out = pd.DataFrame(rows, columns=["full", "mapped_rxn", "method", "confidence", "error"])
    args.maps.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.maps.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(args.maps)
    print(out["method"].value_counts().to_string())


if __name__ == "__main__":
    main()
