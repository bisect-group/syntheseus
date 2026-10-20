from pathlib import Path

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
O = str(Path(__file__).resolve().parents[4] / "outputs/baselines/syntheseus")
corpus = pd.read_parquet(f"{O}/corpora/uspto50k_graph2edits.parquet")
res = pd.read_parquet(
    f"{O}/overlap/uspto50k_graph2edits/random_splits_grouped_rxn_key/all/test.parquet"
)
q = res.sample(200, random_state=0)
# plain sets of on-bit indices from the DRFP hashed (folded) encoder, per reaction, one at a time
kw = dict(n_folded_length=2048, min_radius=0, radius=3, rings=False, include_hydrogens=False)


def onbits(s):
    return set(np.flatnonzero(DrfpEncoder.encode([s], show_progress_bar=False, **kw)[0]))


cs = [onbits(f"{r}>>{p}") for r, p in zip(corpus["reactants"], corpus["products"])]


def canon(side):
    return ".".join(sorted(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in side.split(".")))


ck = {canon(r) + ">>" + canon(p) for r, p in zip(corpus["reactants"], corpus["products"])}
maxdiff = 0
bad_exact = 0
for _, row in q.iterrows():
    a = onbits(row["rxn_key"])
    best = max((len(a & b) / len(a | b) if a | b else 0.0) for b in cs)
    maxdiff = max(maxdiff, abs(best - row["max_drfp_tanimoto"]))
    l, r = row["rxn_key"].split(">>")
    bad_exact += (canon(l) + ">>" + canon(r) in ck) != row["exact_match_rxn_key"]
print(
    "rows",
    len(q),
    "max abs sim diff",
    maxdiff,
    "exact mismatches",
    bad_exact,
    "n_with_sim>=0.39",
    int((q.max_drfp_tanimoto >= 0.39).sum()),
)
