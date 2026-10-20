# Syntheseus zero-shot wrapper: report

Reaction outcome is zero-shot here. No model is trained, tuned or seeded. All commands run from the submodule root
`/home/adhil/github/EMULaToR/baselines/syntheseus`. `$PY` is `/home/adhil/.conda/envs/syntheseus_modern/bin/python`.

## 1. Method summary

- Syntheseus is a library of single-step reaction models with one interface.
- Retro models: Chemformer, GLN, Graph2Edits, LocalRetro, MEGAN, MHNreact, RetroKNN, RootAligned.
  Each uses the released USPTO-50K checkpoint.
- Forward models: Chemformer and R-SMILES (RootAligned), with the authors' USPTO-MIT mixed checkpoints.
- Each model gives a ranked list of candidates for an input. Retro input is the product. Forward input is the reactant set.
- The wrapper predicts the test files of the Leak-CURBER reaction outcome splits and scores top-1, top-10 and MRR@10.
- The wrapper also audits the overlap between each test file and each model's training corpus.

## 2. Original default settings

Every value and its source is in `emulator_bench/default_hparams.json`.

| Setting | Value | Source |
|---|---|---|
| Training, epochs, lr, optimizer | none (released checkpoints) | not applicable |
| Top-N results | 10, retro and forward | user decision; R-SMILES RtoP translate config and Chemformer evaluation for forward |
| Batch size | 16 | syntheseus `eval-single-step` default (`BaseEvalConfig.batch_size`) |
| Random seed | `set_random_seed(0)`; beam search is deterministic | `eval-single-step` |
| Chemformer beam size | 10 (= `num_results`) | `ChemformerModel` |
| Graph2Edits | `max_edit_steps` 9 | syntheseus default |
| MEGAN | `n_max_atoms` 200, `max_gen_steps` 16, `beam_batch_size` 10 | syntheseus default |
| MHNreact | `use_FPF` true, 1000 additional templates | syntheseus default |
| RootAligned retro | 20 augmentations, temperature 2.0, beam 10, 2048 tokens, max length 500 | checkpoint yml |
| RootAligned forward | 5 augmentations, beam 10, n_best 10, 4096 tokens, max length 1000, score alpha 1 | `RtoP-MIT-mixed-aug5-*.yml`, `score.py` |
| GLN | `schneider50k`, template set `default` | syntheseus default |
| Metrics | top-k and MRR, exact match on syntheseus `Reaction` equality, `skip_repeats=True` | syntheseus `TopKMetricsAccumulator` |

The publication PDFs were not read for this report. The settings come from the released code and configs.

## 3. What the wrapper does

- `launch.py` runs two phases for every model x direction x split folder x fold. Default fold is `test`.
- Predict phase (`predict.py`, GPU): the unique inputs of all test files are sorted and split into `--workers` shards.
  A shard predicts in batches of 16 and saves each batch in `OUT/cache/predictions_v2/...`.
  The cache folder name has a hash of the checkpoint and model settings. A restarted shard skips finished batches.
- Evaluate phase (`run_model.py`, CPU): each split file reads the cache and runs syntheseus `compute_metrics`.
  It writes `predictions.jsonl`, `result.json` and a `DONE` marker. `--no_cache` runs the model on every row.
- The product of a row is its largest product (most heavy atoms; ties: longer canonical SMILES, then lexicographic).
  The retro target is the full reactant set. The forward target is that one product.
- Input is 12,108 test rows (random 4,299, time 3,511, DRFP 4,298), 11,683 unique.
- Determinism measures in the code:
  - `build_model` calls `model.eval()` for LocalRetro and RetroKNN. The syntheseus wrapper leaves dropout on.
  - `build_model` seeds RootAligned retro per input (`random.seed` of the CRC32 of the input SMILES, one input per call).
    The output of an input then does not depend on its batch.
  - R-SMILES forward draws roots from a per-input seed: 33 plus the CRC32 of the sorted reactants.
  - `predict.py` and `run_model.py` call `set_random_seed(0)`.
  - `launch.py` sets `PYTHONHASHSEED=0` for every child process.
- Speed-up: `predict.py` caps `multiprocessing.cpu_count` and syntheseus `misc.cpu_count` to 8.
  RootAligned retro and MHNreact start a pool of `cpu_count()` workers on every call, and `cpu_count()` is 112 here.
  Measured: RootAligned retro 217 s with the cap and 597 s without. MHNreact 21 s with the cap and 131 s without.
  54 of 54 outputs were identical with and without the cap.
- GLN runs without Docker. `gln_compat.py` replaces the compiled `extlib`, `torch_scatter` and `torch_sparse`.
- Environment: conda env `syntheseus_modern` for every model. The pinned env `syntheseus` is a reference only.

## 4. Files added, removed or changed

- Added: the whole `emulator_bench/` folder (`launch.py`, `predict.py`, `predcache.py`, `run_model.py`, `common.py`,
  `corpora.py`, `overlap.py`, `aggregate.py`, `compare_runs.py`, `prepare_checkpoints.py`, `gln_compat.py`,
  `rsmiles_forward.py`, `rsmiles_forward_translate.yml`, `default_hparams.json`, env files, `tests/`, `checks/`,
  `README.md`, `commands.txt`, `REPORT.md`). The folder is not yet committed.
- Changed outside `emulator_bench/`, two files:
  - `syntheseus/reaction_prediction/models/retro_knn.py`: `torch_scatter.scatter` replaced by `output.scatter_add_`.
  - `syntheseus/reaction_prediction/utils/misc.py`: `dictify` accepts numpy scalars (`np.generic`).
- Removed: nothing.

## 5. Bugs fixed

- LocalRetro and RetroKNN predicted with dropout on in the syntheseus wrapper. The wrapper sets `eval()`.
- RootAligned retro drew roots from the global `random` state, so output depended on batch and order. It is now seeded per input.
- `dictify` failed on numpy scalars. Fixed in `misc.py`.
- RetroKNN needed `torch_scatter`. Replaced by `scatter_add_`.
- Pools of 112 workers made RootAligned retro and MHNreact slow. Capped at 8 (see section 6).

## 6. Deviations, and why behaviour is unchanged

- Pool cap of 8 workers: changes time only. 54 of 54 outputs were identical with and without it (section 3).
- `retro_knn.py` `scatter_add_`: same sum as `torch_scatter.scatter` with the default sum reduce. No separate measurement is recorded.
- Modern env instead of pinned env: see section 9 for the measured differences.
- GLN without Docker: see section 9.
- Prediction cache and shards: each input is predicted once, then expanded to all rows. Valid only if output does not depend on batch.
  See section 9.
- Per-input seed for RootAligned retro and R-SMILES forward: needed for batch independence. Syntheseus itself seeds once globally.
  The seed changes which random roots are drawn, so single outputs can differ from a plain syntheseus run.
- Top-N is 10 for every model by user decision.

## 7. Exact commands

The full list is in `emulator_bench/commands.txt`. Main steps:

```
PY=/home/adhil/.conda/envs/syntheseus_modern/bin/python
$PY -m emulator_bench.corpora
$PY -m emulator_bench.prepare_checkpoints
taskset -c 0-15 $PY -u -m emulator_bench.overlap
$PY -m emulator_bench.launch --gpus 0 --workers 4 --procs_per_gpu 4 --dry_run
$PY -m emulator_bench.launch --gpus 0 --workers 4 --procs_per_gpu 4
$PY -m emulator_bench.aggregate
```

There are no seeds. The run covers all models x directions x split folders on the test fold.

## 8. Smoke test result

`commands.txt` has a 32-row smoke command (`--limit 32`). This report has no recorded result of a smoke run.
The 32-row comparisons in section 9 used the same limit.

## 9. Equivalence verdict

Verdict: EQUIVALENT-EXCEPT-DOCUMENTED. Numbers below come from the checks recorded in `README.md`.
Nothing is trained, so the check compares outputs of the wrapper with the original code or other envs.

- Batch independence on 16 inputs (batch, reversed, single, split): Graph2Edits, MEGAN, MHNreact, Chemformer and
  RootAligned forward are batch-independent. GLN gave 15 of 16 identical in a repeat of the same batch (GPU float noise).
  LocalRetro, RetroKNN and RootAligned retro were not independent until the `eval()` and per-input seed fixes.
- Sharded and cached runs against a plain run on 200 rows (LocalRetro, RootAligned, Chemformer): 200 of 200 identical top-10 lists.
  Max score difference 0, 0 and 1.6e-6.
- Pinned env against modern env on 32 rows, ranked top-10 lists:
  - Identical: LocalRetro, RetroKNN, RootAligned (both directions), Chemformer forward.
  - Chemformer retro 30 of 32, MEGAN 29 of 32, Graph2Edits 27 of 32. Top-1 is equal. Differences are at tied or near-tied candidates.
  - MHNreact 8 of 32. The pinned env gives radical species such as `C([O])=O`. The modern env gives the closed-shell form
    (RDKit and rdchiral versions). The modern env is used.
- GLN without Docker against the Docker reference on 232 rows: 227 identical top-10 lists, 231 identical top-1.
  Equal accuracy on 200 USPTO-50K rows.
- Chemformer forward on 1000 USPTO-MIT test rows: top-1 0.906 on single-product rows (published about 0.91).
- Pool cap: 54 of 54 outputs identical.
- Not checked against the original code: Graph2Edits, MEGAN, MHNreact, RetroKNN, GLN direct runs through the syntheseus CLI.
  Only LocalRetro has a recorded `eval-single-step` command in `commands.txt`.

## 10. Open issues

- No recorded smoke-test result (section 8).
- The equivalence numbers in section 9 are copied from `README.md`. This report has no logs for them.
  The 12-section layout in the brief lists 10 sections. This report follows those 10.
- The syntheseus `eval-single-step` result for LocalRetro is not recorded here.
- The pool-cap timings (217 s, 597 s, 21 s, 131 s) come from the earlier agent. The input count and GPU are not recorded.
- The `scatter_add_` change in RetroKNN has no recorded output comparison.
- The paper and publication PDFs were not compared with the defaults (section 2).
- `PYTHONHASHSEED=0` was added to `launch.py` after the current zero-shot run started. That run used the old environment.
  Its workers had random hash seeds. The effect on outputs is not measured.
- Chemformer forward beams repeat one molecule, so top-10 is close to top-1.
- About 90% of rows are multi-product. The forward task keeps only the largest product.
- `emulator_bench/` is not committed.
