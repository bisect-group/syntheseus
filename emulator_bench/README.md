# Syntheseus zero-shot wrapper (reaction outcome)

This folder runs released reaction-prediction checkpoints on the Leak-CURBER reaction outcome splits.
It does no training and no tuning. Beam search is deterministic, so there are no seeds.

Run all commands from the syntheseus submodule root. `commands.txt` has the exact commands.

## What it does

- Retro: Chemformer, GLN, Graph2Edits, LocalRetro, MEGAN, MHNreact, RetroKNN, RootAligned.
  Each uses the syntheseus default checkpoint (USPTO-50K).
- Forward: Chemformer and R-SMILES (RootAligned), with the authors' USPTO-MIT mixed checkpoints.
- Input: the `test` file of each split folder under `data/processed/[datasets/]reaction_outcome_dataset/`
  (random 4,299 + time 3,511 + DRFP 4,298 = 12,108 rows, 11,683 unique). Zero-shot inference only, so `--folds`
  defaults to `test`.
- Retro reads curated split files (`retro_targets.py`, in `reaction_outcome_dataset_retro/` next to the dataset):
  one main product per row that is not a cofactor, and only the reactants that give heavy atoms to it (RXNMapper
  atom mapping). No-op rows and rows whose products are all cofactors are dropped, duplicates merged
  (random 4,022, time 3,341, DRFP 4,258 rows). See "Retro targets" below.
- Forward: the product of each row is its largest product (most heavy atoms). The reactant set is the input.
- Metrics: retro top-1, top-10, top-50 and MRR (MRR@50) from 50 predictions; forward top-1, top-10 and MRR@10.
  syntheseus `compute_metrics`; per-row predictions are saved. The **headline retro metric ignores
  stereochemistry**: enzymes are stereospecific while USPTO-trained models largely are not, so stereo is
  reported as its own `stereo_*` columns rather than folded into the headline. `rescore.py` computes both,
  plus the multi-reference variants.
- Common test set (`common_set.py`): every retro model is scored on the same rows. A curated row is left out
  for all models if any model cannot take it (MEGAN above 200 heavy atoms or skipped at run time in any seed,
  Chemformer input or target above 512 tokens, RootAligned target above 500 tokens).
- Seeds: `--seed N` (launch, predict, run_model) sets the global seed and RootAligned's per-input seed
  (CRC32 of `N:SMILES`; seed 0 keeps the plain CRC32). Seed N > 0 caches to `top<N>_<hash>_seed<N>` and runs to
  `OUT/seed<N>/`. `summary_topk_seeds.csv` has the mean and std over seeds.
- Dedup cache: each unique model input is predicted once, then expanded to every split row (see Two phases).
- Overlap audit: DRFP Tanimoto and exact reaction match of each split file against each model's
  training corpus, in the form of `tab:pretraining_audit`.

## Files

- `launch.py`: runs both phases for all models x directions x splits x folds. Flags: `--gpus --procs_per_gpu --workers
  --eval_procs --models --directions --splits --folds --limit --dry_run`.
- `predict.py`, `predcache.py`: one shard of the prediction cache; cache key, storage and the cached model.
- `run_model.py`: one run per split file and fold. Writes `predictions.jsonl`, `result.json` and a `DONE` marker.
  `--no_cache` runs the model on every row instead.
- `common.py`: split discovery, largest-product rule (forward), JSONL cache.
- `retro_targets.py`, `map_reactions.py`, `cofactors.json`: curated retro split files (main product, atom-mapped
  reactant target, cleanup) and the curation report.
- `rescore.py`: stereo-agnostic and multi-reference retro metrics from `predictions.jsonl` (`rescore.json` per run),
  on all curated rows and on the common test set.
- `common_set.py`: the common retro test set (`<fold>_common.parquet`, `<fold>_common_excluded.parquet`).
- `rsmiles_forward.py`, `rsmiles_forward_translate.yml`: R-SMILES forward model.
- `megan_forward.py`: forward MEGAN, over the released code's own forward mode (syntheseus ships no such wrapper).
- `gln_compat.py`: runs GLN in the modern env without its compiled extensions.
- `overlap.py`, `corpora.py`: overlap audit and corpus download.
- `aggregate.py`: writes `summary_topk.csv` and `summary_overlap.csv`.
- `compare_runs.py`: compares two runs (ranked lists and scores).
- `default_hparams.json`: every inference setting with its source.
- `environment.yml`: the pinned env. `environment_modern_freeze.txt`: the modern env.
- `tests/`, `checks/`: unit tests and the scripts used for the equivalence checks.

## Two phases

1. Predict (`predict.py`, GPU). The unique inputs of all test files are sorted. Retro input is the curated main
   product; forward input is the reactant set (canonical SMILES). Each model x direction runs `--workers` shard processes.
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

A batch that runs out of GPU memory (shared GPUs, large enzymatic molecules) is retried one input at a time
(`predict.predict_batch`). An input that still does not fit after 3 one-minute waits is recorded as skipped and
leaves the common test set. On the shared A10s this happened to 2 molecules (GLN in some seeds, RetroKNN in one),
both also above MEGAN's 200-atom and Chemformer's 512-token limits. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and at most 2
processes per GPU for LocalRetro and RetroKNN (up to 12 GB and 6 GB per process).

Seed variation (random split, ranked top-50 lists that differ from seed 0): Graph2Edits and MHNreact 0 of 4,022,
Chemformer 1, LocalRetro 8-9, MEGAN 51-53, RetroKNN 49-61 (GPU float noise at tied scores), GLN 388-400, and
RootAligned about 3,830 (the seed changes its augmentation roots).

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

- Forward: the product of each row is its largest product (most heavy atoms; ties: longer canonical SMILES, then
  lexicographic). The forward target is that one product. About 90% of rows are multi-product.
  `result.json` has `n_multi_product` and `n_multi_tie`.
- `num_top_results` is 50 for retro and 10 for forward. Beam-search models use beam size = `num_top_results`;
  RootAligned retro keeps its checkpoint's beam 10 x 20 augmentations. Top-10 of a top-50 run is not the same as a
  top-10 run, because the beam is wider.
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

## Retro targets

The split files are enzymatic reactions, so the largest product is often a cofactor (NAD(P)+, CoA, ADP, SAH: 28% of
random, 37% of time and 4% of DRFP test rows) and the raw `reactants` column holds water, charges and cofactors.
`retro_targets.py` builds the retro task from `rxn_key` instead:

- Molecules are neutralized, imidic acids are written as amides, SMILES are canonical with stereo.
- Main product: the largest product that is not a cofactor (`cofactors.json`: listed cofactors, inorganic species,
  single atoms) and not a spectator (found unchanged among the reactants, e.g. heme). Co-products of carriers
  (acyl-CoA, acyl-ACP, C1-THF, NDP-sugars, when the free carrier is a reactant) and of amino-group shuttles
  (Glu/2-oxoglutarate, Ala/pyruvate, succinate) are used only when nothing else is left.
- Target: the reactants that give heavy atoms to the main product in an RXNMapper mapping of the reaction (water from
  `rxn_smiles` added back). Water, O2, H2O2, metal ions and hydride/electron carriers (NAD(P)H, FADH2, FMNH2, dyes) are
  never targets. A reactant that gives one heavy atom stays only if it is a real one-atom donor (SAM methyl, NH3,
  halide, sulfide); a lone oxygen is a mapping ambiguity. Reactions too long for RXNMapper (about 1%) keep their
  reactants that are not cofactors.
- Dropped: rows whose products are all cofactors, no-op rows (nothing new besides spectators and cofactors), rows with
  an empty target. Rows of one split file with the same (product, reactants) are merged.

Steps (the mapper runs in any env with rxnmapper):

    $PY -m emulator_bench.retro_targets prepare
    <python with rxnmapper> -I emulator_bench/map_reactions.py <requests.parquet> <maps.parquet>
    $PY -m emulator_bench.retro_targets build

Output: `<fold>.parquet` and `<fold>_dropped.parquet` per split folder, `manifest.json` and `curation_report.md`
(counts, rules used, random examples). `rescore.py` then gives, per run, top-k and MRR for exact (stereo) and
stereo-agnostic match, per reaction and per unique product with every reference reactant set of that product.

## Fine-tuning

The zero-shot numbers measure released USPTO checkpoints on enzymatic reactions, and they are low because
the chemistry is out of domain (see `REPORT.md`). Fine-tuning trains those same checkpoints on the curated
**train** split, selects on **val**, and scores the unchanged **test** set, so a fine-tuned number is
directly comparable with its zero-shot row.

- `finetune_data.py`: per-model training data from the curated files. The writers differ because the models
  do. `--model Chemformer --skip_maps` writes a pickle of RDKit molecules and a `set` column and needs no
  atom maps, so it keeps every curated row. `--model LocalRetro` and `--model RootAligned` write
  `raw_{train,val,test}.csv` in the `reactants>reagents>production` form with the atom maps the template and
  root-alignment code needs; `mapped_subreaction` rebuilds the mapping of the *curated task* (target
  reactants to main product) out of the RXNMapper mapping of the full reaction. RootAligned additionally
  needs every product atom mapped, so its writer numbers the atoms whose source the curation removed
  (`fully_map_products`); without this, 28% of reactions are rejected, and systematically the oxygenations.
- `finetune_localretro.py`, `finetune_chemformer.py`, `finetune_rootaligned.py`, `finetune_megan.py`: one
  driver per model. Each warm starts from the released checkpoint, trains with the model's own published
  hyper-parameters, uses val for early stopping or best-checkpoint selection, and exports a syntheseus
  model directory, so the usual predict/evaluate path serves the result unchanged.
- `finetune_eval.sh <model> <group> <name> [shards] [direction] [seed]`: evaluation through the same
  predict/run_model/rescore/aggregate path as the zero-shot runs. Predictions are cached under a tag, so
  fine-tuned and zero-shot predictions never mix.
- `parallel_worker.sh <gpu>`: one worker per GPU, each claiming the next unfinished cell with an atomic
  `mkdir`. A cell is skipped when its run directory already has a `DONE`, so the whole matrix is resumable.

**How much transfers.** A sequence model transfers whole: Chemformer and RootAligned load every released
weight. A template or action model cannot, because the size of its output layer is set by a vocabulary
extracted from the training data, and the enzymatic vocabulary is not the USPTO one: LocalRetro copies 28 of
32 tensors and reinitialises the template heads, MEGAN copies 172 of 174. The extracted enzymatic template
set is itself a finding: 26,145 LocalRetro classes against USPTO-50K's 672, 80% of them seen once.

**Directions.** `--direction forward` trains the reverse task on the same curated rows, so forward is the
exact inverse of retro rather than a different dataset. syntheseus ships forward wrappers for Chemformer
and RootAligned only; `megan_forward.py` adds a third (below), and LocalRetro and the template models stay
retro-only. The released USPTO-MIT forward checkpoints are distributed by hand and are not in this checkout,
so a forward run warm starts from the released *backward* checkpoint: same architecture and tokeniser, only
the direction is new.

**MEGAN forward** is not a new model but the released code's own forward mode, which syntheseus does not
expose. `src/config.py` registers `megan_for_8_dfs_cano`, the featuriser the authors used for USPTO-MIT:
`forward=True` makes `src/feat/featurize.py` read `substrates` as the source and `product` as the target,
and reorders the edit priorities to lead with bond *additions* where retrosynthesis leads with deletions.
The network and the beam search are shared, so only the two ends needed work:

- `finetune_megan.py --direction forward` swaps that featuriser key into the gin config and writes to a
  separate tree, because the two directions' action vocabularies differ (retro 87 actions, forward 59) and
  must not share a featurised dataset. Retro keeps its original paths, so finished retro runs stay valid.
  The warm start transfers 172 of 174 tensors, as for retro.
- `megan_forward.py` holds the released backward wrapper as an inner loader for the weights, vocabulary and
  action masks, and re-implements only the input and output handling: syntheseus' wrapper is
  direction-specific just in its base class and output processor. A reactant `Bag` is joined into one
  multi-component graph (`A.B.C`, the form the `substrates` column takes), sorted so one input always gives
  one graph and the prediction cache stays valid. Oversize-input skipping and the beam-overflow retry carry
  over from the retro path and matter more here, since reactant bags are larger than the products retro sees.

The published 8-step cap costs nothing on this data: forward needs 3.5 edits on average against retro's 6.4,
so featurisation keeps 86% of training reactions (57,644/67,350 on the random split) where retro keeps 69%.
Assembling a product from known substrates is a shorter edit path than decomposing it. There is no released
forward MEGAN checkpoint, so this cell is fine-tuned only and has no zero-shot baseline;
`default_hparams.json` marks it `"checkpoint": "finetuned_only"` and `launch.py` skips such cells when
building zero-shot jobs.

**Seeds.** `--seed N` on each driver sets the training seed and writes to `<dataset>_seed<N>`; seed 0 keeps
the original paths, so adding seeds never disturbs finished work. The seed reaches the places that actually
vary a run: Chemformer's `seed_everything`, OpenNMT's `seed` *and* RootAligned's augmentation roots, and
MEGAN through `megan_train.py`, which seeds Python, NumPy and torch because `bin/train.py` seeds nothing.

**Upstream compatibility.** The released training code predates this environment, and the fixes live in
wrappers rather than in the vendored packages: `localretro_preprocess.py` (numpy 2 writes `np.int64(3)` into
the CSVs that LocalRetro reads back with `eval`, and pandas 2 removed `DataFrame.append`),
`chemformer_train.py` (Lightning 2 removed `validation_epoch_end` and changed `Trainer(gpus=...)`; its
`ModelCheckpoint` also omits `mode`, which under Lightning 2 would keep the *worst* checkpoint),
`megan_prepare.py` (MEGAN's USPTO-50K class caps molecules at 100 atoms and its featuriser stores the
reaction class in a sparse matrix, so the class must be an integer).

**Measured cost** on one A10, random split, per seed: Chemformer retro 3.3 h train + 5.8 h eval, Chemformer
forward 4.9 h + 3.3 h, RootAligned retro 10.5 h + 1.8 h, RootAligned forward 6.8 h + 0.4 h, MEGAN retro
34.4 h + 0.6 h. Fine-tuned retro inference is much slower than zero-shot because the template space is
larger, so `finetune_eval.sh` sharding matters.
