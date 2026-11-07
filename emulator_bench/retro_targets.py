"""Curated retro split files: one main product per reaction and the reactants that form it.

The split files hold enzymatic reactions. `rxn_key` is the cleaned reaction (water and H+ removed,
tautomers fixed); `rxn_smiles` is the original one. For retrosynthesis each row becomes
main product -> reactants:

- Every molecule of `rxn_key` is neutralized (RDKit Uncharger), imidic acids are written as
  amides (`N=C(O)` -> `NC(=O)`, left over in some peptides), and SMILES are canonical with stereo.
- The main product is the largest product (most heavy atoms, then longer canonical SMILES, then
  lexicographic order) that is not a cofactor (`cofactors.json`) and not a spectator (a molecule
  found unchanged among the reactants, e.g. the heme of P450 rows). Co-products of a carrier or of
  an amino-group shuttle pair (e.g. L-glutamate when 2-oxoglutarate is a reactant) are the main
  product only when no other product is left. Rows whose products are all cofactors are dropped,
  and so are rows whose other products are all spectators (no_op).
- The reactant target is the set of reactants that give at least one heavy atom to the main
  product, read from an RXNMapper atom mapping of the reaction (water from `rxn_smiles` added back
  so that hydrolysis oxygens map to water), minus the `never_target` species and metal ions. A
  reactant that gives a single heavy atom stays only if it is a real one-atom donor (see
  `one_atom_donor`): a lone oxygen, or a lone atom from a cofactor other than SAM, is a mapping
  ambiguity. If the
  mapping fails, the target is every reactant that is not a cofactor or a spectator.
- Rows of one split file that give the same (product, reactants) pair are merged into one.

The curated files go to RETRO_DIR with the layout of DATA_DIR: `<fold>.parquet` (benchmark rows),
`<fold>_dropped.parquet` (removed rows and why), plus `manifest.json` and `curation_report.md`.
Run from baselines/syntheseus:

    $PY -m emulator_bench.retro_targets prepare     # writes the reactions to atom-map
    <python with rxnmapper> -I emulator_bench/map_reactions.py <requests> <maps>
    $PY -m emulator_bench.retro_targets build       # curated files and report
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from emulator_bench.common import (
    DATA_DIR,
    OUT_ROOT,
    RETRO_DIR,
    SplitFile,
    atomic_write_text,
    discover_splits,
    retro_file,
)

COFACTORS_PATH = Path(__file__).with_name("cofactors.json")
RETRO_VERSION = 1  # bump when the curation rules change
WATER_KEY = "XLYOFNOQVPJJNP"
NONMETALS = frozenset("H B C N O F Si P S Cl As Se Br Te I".split())
MAPPER = Path(__file__).with_name("map_reactions.py")


@dataclass(frozen=True)
class Species:
    smiles: str  # neutralized canonical SMILES with stereo
    key: str  # InChIKey first block: connectivity, ignores protonation and stereo
    n_heavy: int
    has_carbon: bool
    metal_ion: bool  # a single metal atom


@lru_cache(maxsize=1)
def _uncharger():
    from rdkit.Chem.MolStandardize import rdMolStandardize

    return rdMolStandardize.Uncharger()


@lru_cache(maxsize=1)
def _imidic_acid():
    from rdkit import Chem

    return Chem.MolFromSmarts("[NX2;!a;+0]=[CX3;!a]-[OX2H1;+0]")


def _amide_form(mol):
    """Rewrite every imidic acid N=C-OH as its amide tautomer NH-C=O (the input if that fails)."""
    from rdkit import Chem

    out = mol
    try:
        while match := out.GetSubstructMatch(_imidic_acid()):
            n, c, o = match
            rw = Chem.RWMol(out)
            bond = rw.GetBondBetweenAtoms(n, c)
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            rw.GetBondBetweenAtoms(c, o).SetBondType(Chem.BondType.DOUBLE)
            for i in (n, o):
                rw.GetAtomWithIdx(i).SetNumExplicitHs(0)
                rw.GetAtomWithIdx(i).SetNoImplicit(False)
            Chem.SanitizeMol(rw)
            out = rw.GetMol()
    except Exception:
        return mol
    if out is mol:
        return mol
    # Round trip, so that leftover bond directions of the old N=C bond are dropped.
    return Chem.MolFromSmiles(Chem.MolToSmiles(out)) or mol


@lru_cache(maxsize=None)
def standardize(smiles: str) -> Species | None:
    """Neutralized canonical form of one molecule, or None if RDKit cannot parse it."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _amide_form(mol)
    uncharged = _uncharger().uncharge(mol)
    std = Chem.MolToSmiles(uncharged)
    # The Uncharger leaves a few anions unparsable (PF6-, cyclopentadienide); keep those charged.
    if Chem.MolFromSmiles(std) is None:
        std = Chem.MolToSmiles(mol)
    else:
        mol = uncharged
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    key = Chem.MolToInchiKey(mol).split("-")[0]
    if not key:  # InChI fails on a few exotic species; the stereo-free SMILES stands in
        flat = Chem.Mol(mol)
        Chem.RemoveStereochemistry(flat)
        key = f"smiles:{Chem.MolToSmiles(flat)}"
    return Species(
        smiles=std,
        key=key,
        n_heavy=len(heavy),
        has_carbon=any(a.GetAtomicNum() == 6 for a in heavy),
        metal_ion=len(heavy) == 1 and heavy[0].GetSymbol() not in NONMETALS,
    )


@dataclass(frozen=True)
class Cofactors:
    currency: dict[str, str]  # key -> name
    never_target: dict[str, str]  # key -> name
    carriers: tuple[tuple[str, str, frozenset[str]], ...]  # (name, core SMILES, free-form keys)
    pairs: dict[str, tuple[str, frozenset[str]]]  # product key -> (name, partner keys)
    one_atom_donors: frozenset[str]  # cofactor names that may give a single heavy atom
    digest: str


def load_cofactors(path: Path = COFACTORS_PATH) -> Cofactors:
    text = path.read_text()
    raw = json.loads(text)

    def key(smiles: str) -> str:
        sp = standardize(smiles)
        if sp is None:
            raise ValueError(f"{path}: cannot parse {smiles}")
        return sp.key

    currency = {key(e["smiles"]): e["name"] for e in raw["currency"]}
    by_name = {e["name"]: key(e["smiles"]) for e in raw["currency"]}
    carriers = tuple(
        (e["name"], e["core"], frozenset(by_name[f] for f in e["free"])) for e in raw["carriers"]
    )
    pairs = {
        key(e["smiles"]): (e["name"], frozenset(key(p["smiles"]) for p in e["partners"]))
        for e in raw["pairs"]
    }
    never = {key(e["smiles"]): e["name"] for e in raw["never_target"]}
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    donors = frozenset(raw["one_atom_donors"])
    assert donors <= set(by_name), donors - set(by_name)
    return Cofactors(currency, never, carriers, pairs, donors, digest)


def is_currency(sp: Species, cof: Cofactors) -> bool:
    """Never the main product: listed cofactors, inorganic species and single atoms."""
    return sp.key in cof.currency or not sp.has_carbon or sp.n_heavy <= 1


def is_never_target(sp: Species, cof: Cofactors) -> bool:
    return sp.key in cof.never_target or sp.n_heavy == 0 or sp.metal_ion


@lru_cache(maxsize=None)
def _has_core(smiles: str, core: str) -> bool:
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmiles(core))


def product_tier(
    sp: Species, reactant_keys: frozenset[str], reactant_smiles: frozenset[str], cof: Cofactors
) -> tuple[int, str]:
    """0: candidate main product, 1: co-product (carrier or pair), 2: cofactor or spectator.

    Returns (tier, reason).
    """
    if is_currency(sp, cof):
        return 2, "cofactor"
    if sp.smiles in reactant_smiles:
        return 2, "spectator"
    if sp.key in cof.pairs:
        name, partners = cof.pairs[sp.key]
        if partners & reactant_keys:
            return 1, f"pair:{name}"
    for name, core, free in cof.carriers:
        if free & reactant_keys and _has_core(sp.smiles, core):
            return 1, f"carrier:{name}"
    return 0, ""


def choose_main(
    products: list[Species], reactants: list[Species], cof: Cofactors
) -> tuple[int | None, str, bool]:
    """(index of the main product or None, rule, whether the size rule hit a tie)."""
    rkeys = frozenset(r.key for r in reactants)
    rsmiles = frozenset(r.smiles for r in reactants)
    tiers = [product_tier(p, rkeys, rsmiles, cof) for p in products]

    def rank(i: int):
        p = products[i]
        return (tiers[i][0], -p.n_heavy, -len(p.smiles), p.smiles)

    order = sorted(range(len(products)), key=rank)
    best = order[0]
    if tiers[best][0] == 2:
        spectator = any(t == (2, "spectator") for t in tiers)
        return None, "no_op" if spectator else "only_cofactor_products", False
    tie = any(
        rank(i)[:3] == rank(best)[:3] and products[i].smiles != products[best].smiles
        for i in order[1:]
    )
    return best, "largest_non_cofactor" if tiers[best][0] == 0 else tiers[best][1], tie


@dataclass(frozen=True)
class Prepared:
    reactants: tuple[Species, ...]
    products: tuple[Species, ...]
    main: int  # index into products
    rule: str
    tie: bool
    full: str  # reaction to map: reactants (+ water) >> products (+ water)
    reduced: str  # reactants (+ water) >> main product, for reactions too long for the mapper
    map_reactants: tuple[int | None, ...]  # reactant index of each mapped fragment (None: water)
    main_fragment: int  # fragment index of the main product on the right of `full`


def _side(smiles: str) -> list[str]:
    return [s for s in smiles.split(".") if s]


def _n_water(species: list[Species]) -> int:
    return sum(s.key == WATER_KEY for s in species)


def prepare_row(rxn_key: str, rxn_smiles: str, cof: Cofactors) -> Prepared | str:
    """Standardized molecules, main product and the reactions to map; or the reason to drop."""
    left, right = rxn_key.split(">>")
    reactants = [standardize(s) for s in _side(left)]
    products = [standardize(s) for s in _side(right)]
    if any(s is None for s in reactants + products):
        return "parse_error"
    if not reactants or not products:
        return "empty_side"
    main, rule, tie = choose_main(products, reactants, cof)
    if main is None:
        return rule

    # Water that rxn_key removed, so that every product atom has its true source when mapping.
    orig_left, orig_right = rxn_smiles.split(">>")
    orig = [[standardize(s) for s in _side(x)] for x in (orig_left, orig_right)]
    extra = [
        max(0, _n_water([s for s in o if s is not None]) - _n_water(k))
        for o, k in zip(orig, (reactants, products))
    ]
    r_idx = [i for i, r in enumerate(reactants) if r.n_heavy > 0]
    p_idx = [i for i, p in enumerate(products) if p.n_heavy > 0]
    lhs = ".".join([reactants[i].smiles for i in r_idx] + ["O"] * extra[0])
    rhs = ".".join([products[i].smiles for i in p_idx] + ["O"] * extra[1])
    return Prepared(
        reactants=tuple(reactants),
        products=tuple(products),
        main=main,
        rule=rule,
        tie=tie,
        full=f"{lhs}>>{rhs}",
        reduced=f"{lhs}>>{products[main].smiles}",
        map_reactants=tuple(r_idx) + (None,) * extra[0],
        main_fragment=p_idx.index(main),
    )


def mapped_contributions(mapped_rxn: str, main_fragment: int) -> list[list[str]] | None:
    """For each fragment on the left, the elements of its heavy atoms mapped into product fragment
    `main_fragment`. None if the mapped reaction cannot be read."""
    from rdkit import Chem

    def heavy(fragment: str) -> list[tuple[int, str]]:
        mol = Chem.MolFromSmiles(fragment, sanitize=False)
        if mol is None:
            raise ValueError(fragment)
        return [(a.GetAtomMapNum(), a.GetSymbol()) for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

    left, right = mapped_rxn.split(">>")
    try:
        main_maps = {m for m, _ in heavy(right.split(".")[main_fragment])} - {0}
        return [[e for m, e in heavy(f) if m in main_maps] for f in left.split(".")]
    except (ValueError, IndexError):
        return None


def one_atom_donor(sp: Species, element: str, cof: Cofactors) -> bool:
    """Whether a reactant that gives a single heavy atom to the main product stays in the target.

    A lone oxygen is a mapping ambiguity (the product oxygen could come from water, O2 or the
    reactant itself), and so is a lone atom from a carbon-containing cofactor, except the listed
    one-atom donors (the methyl of SAM). Inorganic donors (N of NH3, halides, S) stay.
    """
    if element == "O":
        return False
    if sp.key in cof.currency and sp.has_carbon:
        return cof.currency[sp.key] in cof.one_atom_donors
    return True


@dataclass(frozen=True)
class Target:
    reactants: tuple[str, ...]  # sorted unique SMILES
    removed: tuple[str, ...]  # reactants (unique SMILES) not in the target
    method: str  # rxnmapper_full, rxnmapper_main or fallback_*
    confidence: float  # RXNMapper confidence, NaN without a mapping
    one_atom_kept: tuple[str, ...]  # target reactants that give a single heavy atom
    one_atom_dropped: tuple[str, ...]  # reactants left out because they give a single heavy atom


def target_of(prep: Prepared, mapping: dict | None, cof: Cofactors) -> Target:
    """The reactant target from the atom mapping, or the cofactor fallback without one."""
    keep: list[int] = []
    kept_one, dropped_one = [], []
    method, confidence = "fallback_no_mapping", float("nan")
    if mapping is not None and mapping["method"] in ("full", "main"):
        fragment = prep.main_fragment if mapping["method"] == "full" else 0
        contrib = mapped_contributions(mapping["mapped_rxn"], fragment)
        if contrib is not None and len(contrib) == len(prep.map_reactants):
            method, confidence = f"rxnmapper_{mapping['method']}", float(mapping["confidence"])
            for i, atoms in zip(prep.map_reactants, contrib):
                if i is None or not atoms or is_never_target(prep.reactants[i], cof):
                    continue
                r = prep.reactants[i]
                if len(atoms) == 1:
                    if not one_atom_donor(r, atoms[0], cof):
                        dropped_one.append(r.smiles)
                        continue
                    kept_one.append(r.smiles)
                keep.append(i)
            if not keep:
                method = "fallback_empty_mapping"
        else:
            method = "fallback_bad_mapping"
    if not keep:
        spectators = {p.smiles for p in prep.products}
        usable = [
            i
            for i, r in enumerate(prep.reactants)
            if not is_never_target(r, cof) and r.smiles not in spectators
        ]
        keep = [i for i in usable if not is_currency(prep.reactants[i], cof)] or usable
    target = tuple(sorted({prep.reactants[i].smiles for i in keep}))
    removed = tuple(sorted({r.smiles for r in prep.reactants} - set(target)))
    return Target(
        target,
        removed,
        method,
        confidence,
        tuple(sorted(set(kept_one) & set(target))),
        tuple(sorted(set(dropped_one) - set(target))),
    )


def split_files(folds) -> list[tuple[SplitFile, str]]:
    return [(s, f) for s in discover_splits() for f in folds]


def prepare_all(folds, cof: Cofactors) -> dict[tuple[str, str, str], dict[str, Prepared | str]]:
    """Prepared rows per (group, name, fold), keyed by rxn_key."""
    out = {}
    for split, fold in split_files(folds):
        df = pd.read_parquet(split.fold_path(fold), columns=["rxn_key", "rxn_smiles"])
        out[(split.group, split.name, fold)] = {
            str(k): prepare_row(str(k), s, cof) for k, s in zip(df["rxn_key"], df["rxn_smiles"])
        }
    return out


def requests_of(prepared) -> pd.DataFrame:
    rows = {
        (p.full, p.reduced)
        for per_file in prepared.values()
        for p in per_file.values()
        if isinstance(p, Prepared)
    }
    return pd.DataFrame(sorted(rows), columns=["full", "reduced"])


def map_paths(out_root: Path, requests: pd.DataFrame) -> tuple[Path, Path]:
    digest = hashlib.sha256("\n".join(requests["full"]).encode()).hexdigest()[:16]
    folder = out_root / "cache/atom_maps"
    return folder / f"requests_{digest}.parquet", folder / f"maps_{digest}.parquet"


def curate_file(
    split: SplitFile, fold: str, prepared: dict[str, Prepared | str], maps: dict, cof: Cofactors
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(benchmark rows, dropped rows) of one split file."""
    src = pd.read_parquet(split.fold_path(fold), columns=["rxn_key", "rxn_smiles", "uniprot_date"])
    rows, dropped = [], []
    for k, smiles, date in zip(src["rxn_key"].astype(str), src["rxn_smiles"], src["uniprot_date"]):
        prep = prepared[k]
        if isinstance(prep, str):
            dropped.append({"rxn_key": k, "rxn_smiles": smiles, "reason": prep, "detail": ""})
            continue
        target = target_of(prep, maps.get(prep.full), cof)
        product = prep.products[prep.main].smiles
        if product in target.reactants:
            dropped.append(
                {"rxn_key": k, "rxn_smiles": smiles, "reason": "no_op", "detail": product}
            )
            continue
        if not target.reactants:
            dropped.append(
                {"rxn_key": k, "rxn_smiles": smiles, "reason": "empty_target", "detail": product}
            )
            continue
        minor = [p.smiles for i, p in enumerate(prep.products) if i != prep.main]
        rows.append(
            {
                "rxn_key": k,
                "rxn_smiles": smiles,
                "uniprot_date": date,
                "product": product,
                "reactants": ".".join(target.reactants),
                "minor_products": ".".join(minor),
                "removed_reactants": ".".join(target.removed),
                "main_rule": prep.rule,
                "tie": prep.tie,
                "map_method": target.method,
                "map_confidence": target.confidence,
                "one_atom_kept": ".".join(target.one_atom_kept),
                "one_atom_dropped": ".".join(target.one_atom_dropped),
            }
        )
    df = pd.DataFrame(rows)
    # Merge rows that give the same (product, reactants) pair; the first row is kept.
    pair = df["product"] + ">>" + df["reactants"]
    df["merged_rxn_keys"] = pair.map(df.groupby(pair)["rxn_key"].agg(list))
    df["n_merged"] = df["merged_rxn_keys"].map(len)
    first = ~pair.duplicated()
    for k, smiles, keys in zip(df["rxn_key"][~first], df["rxn_smiles"][~first], pair[~first]):
        dropped.append({"rxn_key": k, "rxn_smiles": smiles, "reason": "duplicate", "detail": keys})
    df = df[first].reset_index(drop=True)
    df["n_references"] = df["product"].map(df["product"].value_counts())
    return df, pd.DataFrame(dropped, columns=["rxn_key", "rxn_smiles", "reason", "detail"])


def report(results: dict, cof: Cofactors, seed: int = 0, n_examples: int = 25) -> str:
    lines = [
        "# Curated retro split files",
        "",
        f"Rules: emulator_bench/retro_targets.py (RETRO_VERSION {RETRO_VERSION}); "
        f"cofactors.json sha256 {cof.digest}.",
        "",
        "## Counts",
        "",
        "| split | fold | rows in | parse_error | only_cofactor_products | no_op | empty_target "
        "| duplicate | rows out | unique products | products with >1 reference |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (group, name, fold), (df, dropped) in results.items():
        reasons = Counter(dropped["reason"])
        n_in = len(df) + len(dropped)
        multi = int((df.drop_duplicates("product")["n_references"] > 1).sum())
        lines.append(
            f"| {group}/{name} | {fold} | {n_in} | {reasons['parse_error'] + reasons['empty_side']} "
            f"| {reasons['only_cofactor_products']} | {reasons['no_op']} | {reasons['empty_target']} "
            f"| {reasons['duplicate']} | {len(df)} | {df['product'].nunique()} | {multi} |"
        )
    for (group, name, fold), (df, dropped) in results.items():
        lines += ["", f"## {group}/{name} {fold}", ""]
        lines.append(f"- main product rule: {dict(Counter(df['main_rule']).most_common())}")
        lines.append(f"- size-rule ties: {int(df['tie'].sum())}")
        lines.append(f"- target from: {dict(Counter(df['map_method']).most_common())}")
        conf = df["map_confidence"].dropna()
        if len(conf):
            q = np.quantile(conf, [0.1, 0.5, 0.9])
            lines.append(
                f"- RXNMapper confidence: 10% {q[0]:.3f}, median {q[1]:.3f}, 90% {q[2]:.3f}"
            )
        n_target = df["reactants"].str.count(r"\.") + 1
        lines.append(f"- reactants per target: {dict(sorted(Counter(n_target).items()))}")

        def names(column: str) -> dict:
            found = Counter(
                cof.currency.get(standardize(s).key, s)
                for v in df[column]
                if v
                for s in v.split(".")
            )
            return dict(found.most_common(15))

        kept_cof = Counter(
            cof.currency[standardize(s).key]
            for t in df["reactants"]
            for s in t.split(".")
            if standardize(s).key in cof.currency
        )
        lines.append(f"- listed cofactors kept in targets (atom donors): {dict(kept_cof)}")
        lines.append(f"- one-heavy-atom donors kept: {names('one_atom_kept')}")
        lines.append(f"- one-heavy-atom contributions left out: {names('one_atom_dropped')}")
        lines += ["", "Most frequent main products:", ""]
        for smiles, c in df["product"].value_counts().head(10).items():
            lines.append(f"- {c} × `{smiles}`")
        lines += ["", f"{n_examples} random rows (seed {seed}):", ""]
        sample = df.sample(min(n_examples, len(df)), random_state=seed)
        for r in sample.itertuples():
            lines.append(f"- rxn_key `{r.rxn_key}`")
            lines.append(f"  - retro: `{r.product}` >> `{r.reactants}` ({r.map_method})")
            if r.removed_reactants:
                lines.append(f"  - removed reactants: `{r.removed_reactants}`")
            if r.minor_products:
                lines.append(f"  - minor products: `{r.minor_products}`")
        drops = dropped[dropped["reason"] != "duplicate"]
        if len(drops):
            lines += ["", "Dropped rows (up to 10 per reason):", ""]
            for reason, part in drops.groupby("reason"):
                for r in part.head(10).itertuples():
                    lines.append(f"- {reason}: `{r.rxn_key}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("step", choices=["prepare", "build"])
    p.add_argument("--folds", nargs="+", default=["test"])
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    cof = load_cofactors()
    prepared = prepare_all(args.folds, cof)
    requests = requests_of(prepared)
    req_path, maps_path = map_paths(args.out_root, requests)

    if args.step == "prepare":
        if not req_path.exists():
            req_path.parent.mkdir(parents=True, exist_ok=True)
            requests.to_parquet(req_path)
        print(f"{len(requests)} reactions to map. Next:")
        print(f"  <python with rxnmapper> -I {MAPPER} {req_path} {maps_path}")
        print(f"  python -m emulator_bench.retro_targets build --folds {' '.join(args.folds)}")
        return

    if not maps_path.exists():
        raise SystemExit(f"Missing {maps_path}. Run the `prepare` step and the mapper first.")
    maps = pd.read_parquet(maps_path).set_index("full").to_dict("index")
    results = {}
    for split, fold in split_files(args.folds):
        df, dropped = curate_file(split, fold, prepared[(split.group, split.name, fold)], maps, cof)
        assert not any(is_currency(standardize(s), cof) for s in df["product"])
        path = retro_file(split, fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        dropped.to_parquet(path.with_name(f"{fold}_dropped.parquet"), index=False)
        results[(split.group, split.name, fold)] = (df, dropped)
        print(f"{path}: {len(df)} rows, {len(dropped)} dropped")
    manifest = {
        "retro_version": RETRO_VERSION,
        "cofactors_sha256": cof.digest,
        "source": str(DATA_DIR),
        "atom_maps": str(maps_path),
        "files": {
            f"{g}/{n}/{f}": {"rows": len(df), "dropped": dict(Counter(d["reason"]))}
            for (g, n, f), (df, d) in results.items()
        },
    }
    atomic_write_text(RETRO_DIR / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    atomic_write_text(RETRO_DIR / "curation_report.md", report(results, cof))
    print(f"Report: {RETRO_DIR / 'curation_report.md'}")


if __name__ == "__main__":
    main()
