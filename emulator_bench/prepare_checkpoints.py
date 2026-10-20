"""Prepare the forward checkpoints from the authors' releases (no training).

Chemformer: link the authors' fine-tuned file as the only `*.ckpt` of the model folder.
R-SMILES: copy the OpenNMT translate options next to the checkpoint.
The original files stay in place; run `corpora.py` and the syntheseus downloads separately.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from emulator_bench.common import OUT_ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    args = p.parse_args()

    # The syntheseus Chemformer fork still reads `vocab_size`, so the authors' file is used unchanged.
    chem = args.out_root / "checkpoints/Chemformer/forward"
    target = chem / "uspto_mixed.ckpt"
    if not target.exists():
        target.symlink_to("original/last.ckpt")

    rs = args.out_root / "checkpoints/RootAligned/forward"
    shutil.copy(Path(__file__).with_name("rsmiles_forward_translate.yml"), rs / "translate.yml")


if __name__ == "__main__":
    main()
