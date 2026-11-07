import math

import pytest

from emulator_bench.retro_targets import (
    Prepared,
    choose_main,
    load_cofactors,
    mapped_contributions,
    prepare_row,
    standardize,
    target_of,
)

NADPH = "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)C=CC1"
NADP = "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)c1"
ATP = "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O"
ADP = "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)O)C(O)C1O"
UDP = "O=c1ccn(C2OC(COP(=O)(O)OP(=O)(O)O)C(O)C2O)c(=O)[nH]1"
UDP_GLC = "O=c1ccn(C2OC(COP(=O)(O)OP(=O)(O)OC3OC(CO)C(O)C(O)C3O)C(O)C2O)c(=O)[nH]1"
COA = "CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCS"
ACCOA = COA[:-1] + "SC(C)=O"
OG = "O=C(O)CCC(=O)C(=O)O"
GLU = "NC(CCC(=O)O)C(=O)O"


@pytest.fixture(scope="module")
def cof():
    return load_cofactors()


def species(smiles: str):
    return [standardize(s) for s in smiles.split(".")]


def main_of(rxn: str, cof):
    left, right = rxn.split(">>")
    products = species(right)
    i, rule, _ = choose_main(products, species(left), cof)
    return (None if i is None else products[i].smiles), rule


def test_standardize_neutralizes_and_keeps_stereo():
    assert standardize("CC(=O)[O-]").smiles == "CC(=O)O"
    assert standardize("C[C@H](N)C(=O)O").smiles == "C[C@H](N)C(=O)O"
    # charge variants of one cofactor share the key
    assert standardize(NADP.replace("OP(=O)(O)OP", "OP(=O)([O-])OP")).key == standardize(NADP).key


def test_main_product_skips_cofactors(cof):
    # ketone reduction: NADP+ is larger than the alcohol but is a cofactor
    assert main_of(f"CC(=O)c1ccccc1.{NADPH}>>CC(O)c1ccccc1.{NADP}", cof) == (
        "CC(O)c1ccccc1",
        "largest_non_cofactor",
    )
    # kinase: ADP is larger than the phosphate ester
    assert main_of(f"OCC(O)CO.{ATP}>>O=P(O)(O)OCC(O)CO.{ADP}", cof)[0] == "O=P(O)(O)OCC(O)CO"


def test_coproducts_lose_to_the_substrate_product(cof):
    # transaminase: L-glutamate is the larger product, but 2-oxoglutarate is a reactant
    assert main_of(f"NCC(=O)O.{OG}>>O=CC(=O)O.{GLU}", cof) == ("O=CC(=O)O", "largest_non_cofactor")
    # omega-transaminase with pyruvate as amino acceptor: L-alanine is the co-product
    assert main_of("CC(C)N.CC(=O)C(=O)O>>CC(C)=O.CC(N)C(=O)O", cof)[0] == "CC(C)=O"
    # deacetylation by CoA: acetyl-CoA is a carrier co-product
    assert main_of(f"CC(=O)OCc1ccccc1.{COA}>>OCc1ccccc1.{ACCOA}", cof)[0] == "OCc1ccccc1"
    # glycosyltransferase run backwards: UDP-glucose is a carrier co-product
    assert main_of(f"OC1OC(CO)C(O)C(O)C1Oc1ccccc1.{UDP}>>Oc1ccccc1.{UDP_GLC}", cof)[0] == (
        "Oc1ccccc1"
    )


def test_coproduct_is_main_when_nothing_else_is_left(cof):
    # glutamate dehydrogenase: L-glutamate is the only carbon product
    product, rule = main_of(f"{OG}.N.{NADPH}>>{GLU}.{NADP}", cof)
    assert product == GLU and rule == "pair:L-glutamate"


def test_only_cofactor_products_are_dropped(cof):
    assert main_of(f"{ADP}.{ADP}>>{ATP}.Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O", cof) == (
        None,
        "only_cofactor_products",
    )
    assert prepare_row(f"{NADPH}.O=O>>{NADP}.OO", f"{NADPH}.O=O>>{NADP}.OO", cof) == (
        "only_cofactor_products"
    )


def test_prepare_adds_back_water(cof):
    prep = prepare_row("CC(=O)OC>>CC(=O)O.CO", "CC(=O)OC.O>>CC(=O)[O-].CO.[H+]", cof)
    assert isinstance(prep, Prepared)
    assert prep.full == "COC(C)=O.O>>CC(=O)O.CO"
    assert prep.reduced == "COC(C)=O.O>>CC(=O)O"
    assert prep.map_reactants == (0, None)
    assert prep.main_fragment == 0


def test_mapped_contributions():
    mapped = (
        "[CH3:1][C:2](=[O:3])[CH3:4].[NH2:5][CH3:6]>>[CH3:1][CH:2]([OH:3])[CH3:4].[NH2:5][CH3:6]"
    )
    assert mapped_contributions(mapped, 0) == [["C", "C", "O", "C"], []]
    assert mapped_contributions(mapped, 1) == [[], ["N", "C"]]
    assert mapped_contributions("C>>C", 3) is None


def _prep(rxn_key: str, rxn_smiles: str, cof) -> Prepared:
    prep = prepare_row(rxn_key, rxn_smiles, cof)
    assert isinstance(prep, Prepared)
    return prep


def test_target_drops_hydride_donor_and_oxygen(cof):
    # monooxygenase: substrate + O2 + NADPH >> phenol + water + NADP+ (O2 donates the O)
    prep = _prep(
        f"c1ccccc1.O=O.{NADPH}>>Oc1ccccc1.{NADP}", f"c1ccccc1.O=O.{NADPH}>>Oc1ccccc1.O.{NADP}", cof
    )
    assert prep.full.split(">>")[1].split(".")[0] == "Oc1ccccc1"
    # hand mapping: benzene carbons and one O2 oxygen go to phenol
    lhs = "[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[O:1]=[O:8]." + NADPH
    mapped = f"{lhs}>>[OH:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.{NADP}.[OH2:8]"
    t = target_of(prep, {"method": "full", "mapped_rxn": mapped, "confidence": 0.9}, cof)
    assert t.reactants == ("c1ccccc1",)
    assert t.method == "rxnmapper_full" and t.confidence == 0.9
    assert set(t.removed) == {"O=O", standardize(NADPH).smiles}


def test_target_keeps_group_donor(cof):
    # SAM methylation: the methyl carbon comes from SAM, so SAM stays in the target
    sam = "C[S+](CCC(N)C(=O)O)CC1OC(n2cnc3c(N)ncnc32)C(O)C1O"
    sah = "Nc1ncnc2c1ncn2C1OC(CSCCC(N)C(=O)O)C(O)C1O"
    prep = _prep(f"Oc1ccccc1.{sam}>>COc1ccccc1.{sah}", f"Oc1ccccc1.{sam}>>COc1ccccc1.{sah}", cof)
    sam_mapped = "[CH3:1][S+](CCC(N)C(=O)O)CC1OC(n2cnc3c(N)ncnc32)C(O)C1O"
    mapped = (
        f"[OH:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1.{sam_mapped}"
        f">>[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1.{sah}"
    )
    t = target_of(prep, {"method": "full", "mapped_rxn": mapped, "confidence": 0.5}, cof)
    assert t.reactants == tuple(sorted({"Oc1ccccc1", standardize(sam).smiles}))
    assert t.one_atom_kept == (standardize(sam).smiles,)


def test_target_fallback_without_mapping(cof):
    prep = _prep(
        f"CC(=O)c1ccccc1.{NADPH}>>CC(O)c1ccccc1.{NADP}",
        f"CC(=O)c1ccccc1.{NADPH}>>CC(O)c1ccccc1.{NADP}",
        cof,
    )
    t = target_of(prep, None, cof)
    assert t.reactants == ("CC(=O)c1ccccc1",)
    assert t.method == "fallback_no_mapping" and math.isnan(t.confidence)
    bad = target_of(prep, {"method": "full", "mapped_rxn": "C.C.C>>C", "confidence": 1.0}, cof)
    assert bad.method == "fallback_bad_mapping" and bad.reactants == ("CC(=O)c1ccccc1",)


def test_imidic_acids_become_amides():
    assert standardize("CC(O)=NC").smiles == "CNC(C)=O"
    assert standardize("NC(CCC(=N)O)C(=O)O").smiles == standardize("NC(CCC(N)=O)C(=O)O").smiles
    assert standardize("N[C@@H](C)C(O)=N[C@@H](C)C(=O)O").smiles == "C[C@H](N)C(=O)N[C@@H](C)C(=O)O"
    assert standardize("Oc1ccccn1").smiles == "Oc1ccccn1"  # aromatic: unchanged


def test_spectators_are_not_the_main_product(cof):
    heme = "C=CC1=C(C)c2cc3[nH]c(cc4[nH]c(cc5nc(cc1n2)C(=C)C=5CCC(C)=O)c(CCC(C)=O)c4C)c(C)c3=CC"
    rxn = f"CCCCCCCC.{heme}.[Fe+7].O=O>>CCCCCCCCO.{heme}.[Fe+7]"
    assert main_of(rxn, cof) == ("CCCCCCCCO", "largest_non_cofactor")
    assert main_of("CCO>>CCO", cof) == (None, "no_op")
    # without a mapping the spectator stays out of the target too
    t = target_of(_prep(rxn, rxn, cof), None, cof)
    assert t.reactants == ("CCCCCCCC",)


def test_lone_oxygen_is_left_out(cof):
    # 2-oxoglutarate dioxygenase where the mapper credits the new OH oxygen to 2-oxoglutarate
    rxn = f"c1ccccc1.{OG}.O=O>>Oc1ccccc1.O=C(O)CCC(=O)O.O=C=O"
    prep = _prep(rxn, rxn, cof)
    og = "[OH:1]C(=O)CCC(=O)C(=O)O"
    mapped = f"[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.{og}.O=O>>[OH:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.O=C(O)CCC(=O)O.O=C=O"
    t = target_of(prep, {"method": "full", "mapped_rxn": mapped, "confidence": 0.3}, cof)
    assert t.reactants == ("c1ccccc1",)
    assert t.one_atom_dropped == (standardize(OG).smiles,)
    # hydride carriers never stay in the target, whatever the mapping says
    fmnh2 = "Cc1cc2c(cc1C)N(CC(O)C(O)C(O)COP(=O)(O)O)c1[nH]c(=O)[nH]c(=O)c1N2"
    fmn = "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(CC(O)C(O)C(O)COP(=O)(O)O)c2cc1C"
    rxn = f"c1ccccc1.{fmnh2}.O=O>>Oc1ccccc1.{fmn}"
    prep = _prep(rxn, rxn, cof)
    fm = standardize(fmnh2).smiles.replace("Cc1", "[CH3:1]c1", 1).replace("cc1C", "cc1[CH3:8]", 1)
    mapped = f"[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.{fm}.O=O>>[CH3:1][CH2:8]c1ccccc1.{fmn}"
    t = target_of(prep, {"method": "full", "mapped_rxn": mapped, "confidence": 0.3}, cof)
    assert t.reactants == ("c1ccccc1",) and t.method == "fallback_empty_mapping"
    # an inorganic one-atom donor stays: chlorination with chloride
    rxn = "c1ccccc1.Cl.OO>>Clc1ccccc1"
    prep = _prep(rxn, rxn, cof)
    mapped = "[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[ClH:1].OO>>[Cl:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
    t = target_of(prep, {"method": "full", "mapped_rxn": mapped, "confidence": 0.3}, cof)
    assert t.reactants == ("Cl", "c1ccccc1")
