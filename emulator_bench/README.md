# Syntheseus zero-shot wrapper (reaction outcome)

This folder runs released reaction-prediction checkpoints on the Leak-CURBER reaction outcome splits.
It does no training and no tuning. Beam search is deterministic, so there are no seeds.

Run all commands from the syntheseus submodule root. `commands.txt` has the exact commands.

## What it does

- Retro: Chemformer, GLN, Graph2Edits, LocalRetro, MEGAN, MHNreact, RetroKNN, RootAligned.
  Each uses the syntheseus default checkpoint (USPTO-50K).
- Forward: Chemformer and R-SMILES (RootAligned), with the authors' USPTO-MIT mixed checkpoints.
- Input: the `test` file of each split folder under
  `data/processed/datasets/reaction_outcome_dataset/` (random 4,299 + time 3,511 + DRFP 4,298 = 12,108 rows,
  11,683 unique). Zero-shot inference only, so `--folds` defaults to `test`. `--folds train val test` also works.
- The product of each row is its largest product (most heavy atoms). The reactant set is the target.
- Metrics: top-1, top-10 and MRR (MRR@10) with syntheseus `compute_metrics`, top-10 for every model. Per-row predictions are saved.
- Dedup cache: each unique model input is predicted once, then expanded to every split row (see Two phases).
- Overlap audit: DRFP Tanimoto and exact reaction match of each split file against each model's
  training corpus, in the form of `tab:pretraining_audit`.

## Files

- `launch.py`: runs both phases for all models x directions x splits x folds. Flags: `--gpus --procs_per_gpu --workers
  --eval_procs --models --directions --splits --folds --limit --dry_run`.
- `predict.py`, `predcache.py`: one shard of the prediction cache; cache key, storage and the cached model.
- `run_model.py`: one run per split file and fold. Writes `predictions.jsonl`, `result.json` and a `DONE` marker.
  `--no_cache` runs the model on every row instead.
- `common.py`: split discovery, largest-product rule, JSONL cache.
- `rsmiles_forward.py`, `rsmiles_forward_translate.yml`: R-SMILES forward model.
- `gln_compat.py`: runs GLN in the modern env without its compiled extensions.
- `overlap.py`, `corpora.py`: overlap audit and corpus download.
- `aggregate.py`: writes `summary_topk.csv` and `summary_overlap.csv`.
- `compare_runs.py`: compares two runs (ranked lists and scores).
- `default_hparams.json`: every inference setting with its source.
- `environment.yml`: the pinned env. `environment_modern_freeze.txt`: the modern env.
- `tests/`, `checks/`: unit tests and the scripts used for the equivalence checks.

## Two phases

1. Predict (`predict.py`, GPU). The unique inputs of all test files are sorted. Retro input is the largest product;
   forward input is the reactant set (canonical SMILES). Each model x direction runs `--workers` shard processes.
   Shard i takes inputs i, i+n, i+2n, ... and predicts them in batches of 16 (`batch_size`).
   Predictions are saved per batch in `OUT/cache/predictions_v2/<direction>/<model>/top<N>_<hash>/`.
   The hash covers the checkpoint and the model settings. A restarted shard skips finished batches.
2. Evaluate (`run_model.py`, CPU). Each split file and fold reads the cache and runs syntheseus `compute_metrics`.
   Metrics and `predictions.jsonl` are the same as a run without the cache, except the `timing` field.

A model must give the same output for an input in any batch, or the cache would change results.
Checked on 16 inputs (batch, reversed, single, split): GLN (15 of 16 in a repeat of the same batch, GPU float noise),
Graph2Edits, MEGAN, MHNreact, Chemformer and RootAligned forward are batch-independent.
The syntheseus wrappers of LocalRetro and RetroKNN leave dropout on, and RootAligned retro draws roots from the global
`random` state. So `build_model` sets `model.eval()` for LocalRetro and RetroKNN, and seeds RootAligned retro per input
(CRC32 of the input SMILES, one input per call). Sharded and cached runs then equal a plain run on 200 rows
(LocalRetro, RootAligned, Chemformer: 200 of 200 identical top-10 lists; max score difference 0, 0 and 1.6e-6).

`predict.py` caps the process pools to 8 workers (`multiprocessing.cpu_count` and syntheseus `cpu_count`).
RootAligned retro and MHNreact start a pool of `cpu_count()` workers on every call, and `cpu_count()` is 112 here.
Forking that many workers cost most of the time. The cap changes the time only: 54 unique inputs gave the same top-10 lists
and scores with and without it (see `REPORT.md`).

`launch.py` sets `PYTHONHASHSEED=0` for every child process, so set and dict order does not change between workers.

## Output layout

`OUT = outputs/baselines/syntheseus`

- Runs: `OUT/<direction>/<model>/<split_group>/<split_name>/<fold>/` (`split_name` is `all` for flat groups).
- Checkpoints: `OUT/checkpoints/`. Corpora: `OUT/corpora/` with `manifest.json`. Caches: `OUT/cache/`.
- Summaries: `OUT/summary_topk.csv`, `OUT/summary_overlap.csv`, `OUT/summary_overlap_by_corpus.csv`.

## Environment

Use the conda env `syntheseus_modern` for every model, MHNreact and GLN included
(python 3.11, torch 2.11 cu128, rdkit 2024.9.6, pandas below 3; full list in `environment_modern_freeze.txt`).
The pinned env `syntheseus` (`environment.yml`, torch 2.2.2) is a reference only. Its real RDKit is 2022.9.5.

Pinned against modern env on 32 rows (ranked top-10 lists): LocalRetro, RetroKNN, RootAligned (both directions) and
Chemformer forward are identical. Chemformer retro 30/32, MEGAN 29/32, Graph2Edits 27/32, with equal top-1. They differ only at
tied or near-tied candidates. MHNreact is 8/32: the pinned env gives radical species such as `C([O])=O` from templates,
the modern env gives the closed-shell form (RDKit and rdchiral version). The modern env is the choice for MHNreact.

## Notes

- The product of each row is its largest product (most heavy atoms; ties: longer canonical SMILES, then lexicographic).
  The retro target is the full reactant set. The forward target is that one product. About 90% of rows are multi-product.
  `result.json` has `n_multi_product` and `n_multi_tie`.
- `num_top_results` is 10 for retro and forward. Chemformer and RootAligned beam size is 10.
- GLN runs without Docker: `gln_compat.py` replaces the compiled `extlib` and `torch_scatter`/`torch_sparse`.
  Against the Docker reference on 232 rows: 227 identical top-10 lists, 231 identical top-1, equal accuracy on 200
  USPTO-50K rows. Setup: clone kmaziarz/GLN at 5437fa7a into `OUT/external/GLN` and run `make` in `gln/mods/mol_gnn/mg_clib`.
- Chemformer forward uses the authors' `fined-tuned/uspto_mixed` checkpoint unchanged. On 1000 USPTO-MIT test rows
  top-1 is 0.906 on single-product rows (published about 0.91). Its beams repeat one molecule, so top-10 is close to top-1.
- R-SMILES forward (`rsmiles_forward.py`) follows the original RtoP settings (beam 10, n_best 10, 4096 tokens, max_length 1000,
  5 augmentations, `score.py` ranking). Without atom maps the roots are drawn at random from a per-input seed.
- Code changes outside `emulator_bench/`: `syntheseus/reaction_prediction/models/retro_knn.py` (`torch_scatter` replaced by
  `scatter_add_`) and `syntheseus/reaction_prediction/utils/misc.py` (`dictify` accepts numpy scalars).
- Overlap audit (`overlap.py`): all folds of all split folders against each corpus. DRFP settings follow the Leak-CURBER splitter.
  Per-row results are in `OUT/overlap/`; tables in `OUT/summary_overlap_by_corpus.csv` and `OUT/summary_overlap.csv`.
