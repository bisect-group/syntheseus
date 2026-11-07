"""Forward (reactants to product) MEGAN wrapper for syntheseus.

Paper: https://arxiv.org/abs/2006.15426
Code: https://github.com/molecule-one/megan

MEGAN models a reaction as a sequence of graph edits applied to an input molecule graph, and the
released code supports both directions: `src/feat/featurize.py` swaps which side is the source
(`forward` reads `substrates` as the source and `product` as the target, retro the reverse) and
reorders the action priorities, since forward synthesis leads with bond *additions* where
retrosynthesis leads with bond *deletions*. `src/config.py` registers `megan_for_8_dfs_cano`, the
featuriser the authors used for USPTO-MIT forward prediction: eight generation steps, canonical DFS
action order. Only the featuriser and the data differ; the network and the beam search are shared.

syntheseus ships only a backward wrapper (`ExternalBackwardReactionModel`), whose sole
direction-specific parts are its base class and `process_raw_smiles_outputs_backwards`. This wrapper
reuses the released loading path unchanged by holding a backward wrapper as `_inner` purely as a
loader for the weights, vocabulary and action masks, and re-implements only the input and output
handling:

- the input is a `Bag[Molecule]` of reactants rather than one product, so the bag is joined into a
  single multi-component graph (`A.B.C`), which is the form MEGAN's featuriser reads from the
  `substrates` column;
- the output is scored as a `Reaction` with the bag on the reactant side.

Oversize inputs and beam searches that outgrow the action mask are handled as in the retro path
(`run_model.megan_class`): they are recorded in `skipped` and yield no predictions rather than
killing the batch. Reactant bags are larger than the products the retro path sees, so this matters
more here.
"""

from __future__ import annotations

import fcntl
import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

from rdkit import Chem

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import Reaction
from syntheseus.reaction_prediction.inference_base import ExternalForwardReactionModel
from syntheseus.reaction_prediction.utils.inference import process_raw_smiles_outputs_forwards

# Number of generation steps of `megan_for_8_dfs_cano`, the authors' forward featuriser.
FORWARD_GEN_STEPS = 8


def combined_input(bag: Bag[Molecule]) -> str:
    """The reactant bag as one multi-component SMILES, the form MEGAN's `substrates` column holds.

    The parts are sorted so the same bag always produces the same graph, which keeps the prediction
    cache valid (a model must give one answer per input regardless of how it was batched).
    """
    return ".".join(sorted(mol.smiles for mol in bag))


class MEGANForwardModel(ExternalForwardReactionModel):
    def __init__(
        self,
        *args,
        n_max_atoms: int = 200,
        max_gen_steps: int = FORWARD_GEN_STEPS,
        beam_batch_size: int = 10,
        **kwargs,
    ) -> None:
        """Initializes the forward MEGAN wrapper.

        Assumed format of the model directory, as for the backward wrapper:
        - `model_dir` contains the config as the only `*.gin` file, whose
          `train_megan.featurizer_key` is the forward featuriser
        - `model_dir/model_best.pt` is the model checkpoint
        - `model_dir/{featurizer_key}` contains files needed to build MEGAN's featurizer
        """
        super().__init__(*args, **kwargs)

        from syntheseus.reaction_prediction.inference import megan

        # MEGAN deletes shared log files as it loads, so concurrent shards unlink each other's and
        # raise FileNotFoundError. Building one model at a time removes the race; loading is short.
        lock = Path(tempfile.gettempdir()) / "emulator_bench_megan_load.lock"
        with open(lock, "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                self._inner = megan.MEGANModel(
                    model_dir=self.model_dir,
                    device=self.device,
                    n_max_atoms=n_max_atoms,
                    max_gen_steps=max_gen_steps,
                    beam_batch_size=beam_batch_size,
                )
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

        self.n_max_atoms = n_max_atoms
        self.max_gen_steps = max_gen_steps
        self.beam_batch_size = beam_batch_size
        self.skipped: set[str] = set()  # inputs that produced no predictions

    def get_parameters(self):
        return self._inner.get_parameters()

    def _to_batch(self, inputs: list[Bag[Molecule]]) -> list[Optional[Chem.Mol]]:
        """Each reactant bag as one `rdkit` molecule, or None where MEGAN cannot take it."""
        joined = [Molecule(combined_input(bag)) for bag in inputs]
        batch = self._inner._mols_to_batch(joined)
        for i, (mol, rdmol) in enumerate(zip(joined, batch)):
            # The action mask has n_max_atoms + 1 rows; heavy atoms are the nodes.
            if rdmol is not None and rdmol.GetNumAtoms() > self.n_max_atoms:
                self.skipped.add(mol.smiles)
                batch[i] = None
            elif rdmol is None:
                self.skipped.add(mol.smiles)
        return batch

    def _beam(self, mols: list[Chem.Mol], num_results: int) -> list[list[dict[str, Any]]]:
        import torch
        from src.model.beam_search import beam_search

        if not mols:
            return []
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*DtypeTensor constructors are no longer recommended.*"
            )
            return beam_search(
                [self._inner.model],
                mols,
                rdkit_cache=self._inner.rdkit_cache,
                max_steps=self.max_gen_steps,
                beam_size=num_results,
                batch_size=self.beam_batch_size,
                base_action_masks=self._inner.base_action_masks,
                max_atoms=self.n_max_atoms,
                reaction_types=None,
                action_vocab=self._inner.action_vocab,
            )

    def _main_product(self, smiles: str, reactant_smiles: list[str]) -> str:
        """The main product of an edited graph, by the rule that defined the curated targets.

        MEGAN rewrites the whole input graph, so `final_smi_unmapped` is every component of the
        result: the product *and* whatever was split off, typically water. The benchmark's forward
        target is the single main product (`retro_targets.choose_main`: never a cofactor or a
        spectator, then the largest remaining). Scoring MEGAN's whole bag against that one molecule
        marks a chemically correct answer wrong, so the same rule picks the main product here. The
        sequence models need no such step because they emit the target molecule directly.
        """
        parts = smiles.split(".")
        if len(parts) == 1:
            return smiles
        from emulator_bench.retro_targets import choose_main, load_cofactors, standardize

        # MEGAN can emit molecules that parse but are not valid chemistry (the released wrapper
        # notes the same for C+ atoms), and RDKit then raises rather than returning None, e.g.
        # KekulizeException while standardising. Any such prediction keeps its whole bag and is
        # scored as it is, which fails the comparison, as a broken prediction should.
        try:
            products = [standardize(p) for p in parts]
            reactants = [standardize(r) for r in reactant_smiles]
            if any(p is None for p in products) or any(r is None for r in reactants):
                return smiles
            index, _, _ = choose_main(products, reactants, load_cofactors())
        except Exception:
            return smiles
        # Every component a cofactor (index None) means there is no main product to name.
        return parts[index] if index is not None else smiles

    def _get_reactions(
        self, inputs: list[Bag[Molecule]], num_results: int
    ) -> list[Sequence[Reaction]]:
        batch = self._to_batch(inputs)
        valid_idxs = [i for i, mol in enumerate(batch) if mol is not None]
        valid = [batch[i] for i in valid_idxs]

        # Beam search adds atoms, so a path that grows past the mask size crashes its whole batch.
        # Retry one by one; an input that still crashes gets no predictions.
        try:
            results = self._beam(valid, num_results)
        except RuntimeError as e:
            if "must match the size of tensor" not in str(e):
                raise
            results = []
            for idx, mol in zip(valid_idxs, valid):
                try:
                    results += self._beam([mol], num_results)
                except RuntimeError as inner_error:
                    if "must match the size of tensor" not in str(inner_error):
                        raise
                    self.skipped.add(combined_input(inputs[idx]))
                    results.append([])

        assert len(valid_idxs) == len(results)
        all_outputs: list[list[dict[str, Any]]] = [[] for _ in batch]
        for idx, raw in zip(valid_idxs, results):
            all_outputs[idx] = raw

        results_out = []
        for input, raw_outputs in zip(inputs, all_outputs):
            reactant_smiles = [mol.smiles for mol in input]
            results_out.append(
                process_raw_smiles_outputs_forwards(
                    input=input,
                    output_list=[
                        self._main_product(prediction["final_smi_unmapped"], reactant_smiles)
                        for prediction in raw_outputs
                    ],
                    metadata_list=[
                        {"probability": prediction["prob"]} for prediction in raw_outputs
                    ],
                )
            )
        return results_out
