"""Download the reaction training corpora and write OUT/corpora/<id>.parquet plus manifest.json.

Only reaction training sets are used. Each parquet has columns `reactants` and `products`
(canonical SMILES, atom maps removed). Corpora with identical content are listed under one id.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from emulator_bench.common import OUT_ROOT, atomic_write_text

SOURCES = {
    "uspto50k_syntheseus": {
        "url": "https://ndownloader.figshare.com/files/45206101",
        "file": "uspto50k.zip",
        "member": "raw_train.csv",
        "format": "dai_csv",
        "note": "USPTO-50K train split of Dai et al. (GLN), as used in the syntheseus CLI test (figshare 25459573)",
    },
    "uspto50k_graph2edits": {
        "url": "https://raw.githubusercontent.com/Jamson-Zhong/Graph2Edits/b12e97391c378e46e79e617d80017e95b21bba65/data/uspto_50k/raw_train.csv",
        "file": "raw_train.csv",
        "format": "dai_csv",
        "note": "USPTO-50K train file shipped in the Graph2Edits repository",
    },
    "uspto50k_mhnreact": {
        "url": "https://figshare.com/ndownloader/articles/23960766",
        "file": "MHNreact_default_checkpoint",
        "format": "mhn_checkpoint",
        "note": "rows with split=train in data.csv.gz inside the syntheseus MHNreact default checkpoint",
    },
    "uspto_mit": {
        "url": "https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip",
        "file": "data.zip",
        "member": "data/train.txt",
        "format": "mit_txt",
        "note": "USPTO-MIT train split of Jin et al. (mixed reagents)",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_maps(smiles: str) -> str | None:
    from rdkit import Chem

    out = []
    for s in smiles.split("."):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        out.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(out))


def read_pairs(source: dict, raw: Path, out_root: Path) -> list[tuple[str, str]]:
    """(reactants, products) strings as written in the corpus."""
    fmt = source["format"]
    if fmt == "mhn_checkpoint":
        path = out_root / "checkpoints/syntheseus_default/MHNreact_backward/data.csv.gz"
        df = pd.read_csv(gzip.open(path))
        df = df[df["split"] == "train"]
        return list(zip(df["reactants_can"], df["prod_smiles"]))
    if "member" in source:
        with zipfile.ZipFile(raw) as z:
            data = z.read(source["member"]).decode()
    else:
        data = raw.read_text()
    if fmt == "dai_csv":
        col = pd.read_csv(io.StringIO(data))["reactants>reagents>production"]
        parts = col.str.split(">", expand=True)
        return list(zip(parts[0], parts[2]))
    if fmt == "mit_txt":
        pairs = []
        for line in data.splitlines():
            left, right = line.split(" ")[0].split(">>")
            pairs.append((left, right))
        return pairs
    raise ValueError(fmt)


def build(out_root: Path, corpus_id: str) -> dict:
    source = SOURCES[corpus_id]
    raw = out_root / "corpora/raw" / corpus_id / source["file"]
    if source["format"] != "mhn_checkpoint" and not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(source["url"], raw)
    pairs = read_pairs(source, raw, out_root)
    rows, failed = [], 0
    for r, p in pairs:
        r, p = strip_maps(r), strip_maps(p)
        if r is None or p is None:
            failed += 1
        else:
            rows.append((r, p))
    frame = pd.DataFrame(rows, columns=["reactants", "products"])
    path = out_root / "corpora" / f"{corpus_id}.parquet"
    frame.to_parquet(path.with_name(path.name + ".tmp"))
    path.with_name(path.name + ".tmp").replace(path)
    content = hashlib.sha256("\n".join(sorted(f"{r}>>{p}" for r, p in rows)).encode()).hexdigest()
    return {
        "id": corpus_id,
        "url": source["url"],
        "note": source["note"],
        "source_file": str(raw.relative_to(out_root)) if raw.exists() else "checkpoint data.csv.gz",
        "source_sha256": sha256_file(
            raw
            if raw.exists()
            else out_root / "checkpoints/syntheseus_default/MHNreact_backward/data.csv.gz"
        ),
        "rows_in_file": len(pairs),
        "rows": len(frame),
        "rows_unparsable": failed,
        "content_sha256": content,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--ids", nargs="*", default=list(SOURCES))
    args = p.parse_args()
    (args.out_root / "corpora").mkdir(parents=True, exist_ok=True)
    manifest = [build(args.out_root, i) for i in args.ids]
    atomic_write_text(args.out_root / "corpora/manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
