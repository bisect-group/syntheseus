"""Root-aligned SMILES forward model (reactants to products) for syntheseus.

Paper: https://arxiv.org/abs/2203.11444
Code: https://github.com/otori-bird/retrosynthesis (preprocessing/generate_RtoP_data.py, score.py)

The input side follows `generate_RtoP_data.py` in mixed-reagent mode. Each reactant gets a random root
atom. Roots and reactant order change per augmentation. The original picks roots by atom-map number.
Without atom maps the wrapper picks a root atom index uniformly, which is the same distribution.
Ranking follows `score.py` (`compute_rank`, `score_alpha=1`).
"""

from __future__ import annotations

import argparse
import random
import warnings
import zlib
from collections import defaultdict
from typing import List, Sequence

import yaml
from rdkit import Chem

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import Reaction
from syntheseus.reaction_prediction.inference.root_aligned import RootAlignedModel
from syntheseus.reaction_prediction.inference_base import ExternalForwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_unique_file_in_dir,
    process_raw_smiles_outputs_forwards,
)

AUGMENTATION_SEED = 33  # default of `-seed` in generate_RtoP_data.py


def rooted_smiles(smiles: str, root: int) -> str:
    """Canonical SMILES rooted at atom `root` (-1 for plain canonical)."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=True)


def aligned_sources(reactants: List[str], augmentation: int, rng: random.Random) -> List[str]:
    """Augmented source strings, following the root and order logic of `multi_process` in the original."""
    atom_ids = [list(range(Chem.MolFromSmiles(r).GetNumAtoms())) for r in reactants]
    max_times = 1
    for ids in atom_ids:
        max_times *= len(ids)
    times = min(augmentation, max_times)
    roots = [[-1] * len(reactants)]
    j = 0
    while j < times:
        roots.append([rng.sample(ids, 1)[0] for ids in atom_ids])
        if roots[-1] in roots[:-1]:
            roots.pop()
        else:
            j += 1
    if j < augmentation:
        roots.extend(rng.choices(roots, k=augmentation - times))
        times = augmentation

    sources = []
    for k in range(times):
        pairs = list(zip(reactants, roots[k]))
        rng.shuffle(pairs)
        sources.append(".".join(rooted_smiles(smi, root) for smi, root in pairs))
    return sources


class RootAlignedForwardModel(ExternalForwardReactionModel):
    def __init__(self, *args, num_augmentations: int = 5, **kwargs) -> None:
        """Initializes the forward R-SMILES wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the checkpoint as the only `*.pt` file
        - `model_dir` contains the OpenNMT translate options as the only `*.yml` file
        """
        super().__init__(*args, **kwargs)
        import torch

        with open(get_unique_file_in_dir(self.model_dir, pattern="*.yml")) as f:
            opt = argparse.Namespace(**yaml.safe_load(f))
        opt.models = [str(get_unique_file_in_dir(self.model_dir, pattern="*.pt"))]
        opt.output = "/dev/null"
        opt.gpu = -1 if self.device == "cpu" else torch.device(self.device).index
        opt.synthon = False

        from onmt.translate.translator import build_translator
        from onmt.utils.parse import ArgumentParser
        from root_aligned import score

        score.opt = opt
        ArgumentParser.validate_translate_opts(opt)
        self.translator = build_translator(opt, report_score=False)
        self.num_augmentations = num_augmentations
        self.beam_size = opt.beam_size
        self.batch_size = opt.batch_size
        self.batch_type = opt.batch_type

    def get_parameters(self):
        return self.translator.model.parameters()

    def _get_reactions(
        self, inputs: List[Bag[Molecule]], num_results: int
    ) -> List[Sequence[Reaction]]:
        from root_aligned.score import canonicalize_smiles_clear_map, compute_rank, smi_tokenizer

        sources: List[str] = []
        for bag in inputs:
            reactants = [m.smiles for m in bag]
            rng = random.Random(
                AUGMENTATION_SEED + zlib.crc32(".".join(sorted(reactants)).encode())
            )
            sources.extend(aligned_sources(reactants, self.num_augmentations, rng))

        batch = [bytes(smi_tokenizer(s) + "\n", "utf-8") for s in sources]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="__floordiv__ is deprecated")
            _, translations = self.translator.translate(
                src=batch,
                src_feats=defaultdict(list),
                tgt=None,
                batch_size=self.batch_size,
                batch_type=self.batch_type,
                attn_debug=False,
                align_debug=False,
            )

        lines = [t.replace(" ", "") for hyps in translations for t in hyps]
        canonical = [canonicalize_smiles_clear_map(line) for line in lines]
        per_input = self.num_augmentations * self.beam_size

        outputs = []
        for i, bag in enumerate(inputs):
            chunk = canonical[i * per_input : (i + 1) * per_input]
            predictions = [
                chunk[a * self.beam_size : (a + 1) * self.beam_size]
                for a in range(self.num_augmentations)
            ]
            rank, _ = compute_rank(predictions)
            ranked = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:num_results]
            scores = [score for _, score in ranked]
            outputs.append(
                process_raw_smiles_outputs_forwards(
                    bag,
                    [item[0] for item, _ in ranked],
                    RootAlignedModel.build_prediction_kwargs_from_scores(
                        scores, self.num_augmentations, self.beam_size, None
                    ),
                )
            )
        return outputs
