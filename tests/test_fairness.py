import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fairness as fm  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "employes_exemple.csv"
FEATS = ["age", "poste", "departement", "niveau_diplome", "anciennete_ans", "performance",
         "nb_personnes_managees", "ville", "temps_partiel", "interruption_carriere_mois"]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def prep(df):
    return fm.prepare(df, fm.binarize(df["augmentation_forte"], None), "genre", "F", "H", FEATS,
                      fm.binarize(df["augmentation_forte_equitable"], None))


@pytest.fixture(scope="module")
def results(prep):
    preds = {k: fm.fit_predict(k, prep) for k in fm.MODEL_KINDS}
    return {k: fm.fairness_metrics(prep.y_te, p, prep.a_te, "F", "H", prep.truth_te) for k, p in preds.items()}


# --------------------------------------------------------------------------- briques
def test_binarize_binaire_et_seuil():
    assert fm.binarize(pd.Series([0, 1, 1]), None).tolist() == [0, 1, 1]
    assert fm.binarize(pd.Series([2.0, 5.0, 7.5]), 5.0).tolist() == [0, 1, 1]
    with pytest.raises(ValueError):
        fm.binarize(pd.Series([0.0, 3.0]), None)


def test_prepare_le_genre_n_est_pas_dans_les_variables(prep, df):
    assert prep.X_tr.shape[1] + 1 == prep.Xa_tr.shape[1]
    assert set(np.unique(prep.Xa_tr[:, -1])) <= {0.0, 1.0}
    assert len(prep.X_tr) + len(prep.X_te) == len(df)
    assert set(prep.idx_te).isdisjoint(set(range(len(df))) - set(prep.idx_te))
    # les deux groupes et les deux classes sont présents dans le test
    for g in ("F", "H"):
        assert 0 < prep.y_te[prep.a_te == g].mean() < 1


def test_metriques_sur_un_cas_a_la_main():
    a = np.array(["F"] * 4 + ["H"] * 4)
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    p = np.array([1, 0, 0, 0, 1, 1, 1, 0])   # F : sélection 1/4, TPR 1/2 · H : sélection 3/4, TPR 1
    m = fm.fairness_metrics(y, p, a, "F", "H")
    assert m["sr_a"] == pytest.approx(0.25) and m["sr_b"] == pytest.approx(0.75)
    assert m["dp_diff"] == pytest.approx(-0.5) and m["dp_ratio"] == pytest.approx(1 / 3)
    assert m["tpr_a"] == pytest.approx(0.5) and m["tpr_b"] == pytest.approx(1.0)
    assert m["tpr_diff"] == pytest.approx(-0.5)
    assert m["accuracy"] == pytest.approx(6 / 8)    # concordance sauf pour les personnes n°2 et n°7


def test_proxy_audit_detecte_les_proxys(df):
    auc, tab = fm.proxy_audit(df, FEATS, "genre", "F")
    assert auc > 0.7
    assert {"temps_partiel", "interruption_carriere_mois"} <= set(tab.variable.head(6))


# --------------------------------------------------------------------------- le scénario complet
def test_historique_et_modele_naif_sont_biaises(prep, results):
    hist = [prep.y_te[prep.a_te == g].mean() for g in ("F", "H")]
    assert min(hist) / max(hist) < 0.5
    assert results["naif"]["dp_ratio"] < 0.5          # le modèle naïf reproduit, voire amplifie, le biais


def test_retirer_le_genre_ne_suffit_pas(results):
    assert results["sans_genre"]["dp_ratio"] < 0.8    # le biais passe par les proxys


def test_parite_demographique_corrige_le_biais(results):
    assert results["eg_dp"]["dp_ratio"] > 0.8
    assert abs(results["eg_dp"]["dp_diff"]) < abs(results["sans_genre"]["dp_diff"])
    assert abs(results["to_dp"]["dp_diff"]) < 0.06


def test_egalite_des_chances_preserve_un_historique_biaise(results):
    assert results["eg_eo"]["dp_ratio"] < 0.8         # la contrainte porte sur des étiquettes biaisées


def test_le_modele_equitable_est_plus_proche_de_la_verite(results):
    assert results["eg_dp"]["accuracy_truth"] > results["naif"]["accuracy_truth"]
    assert abs(results["eg_dp"]["tpr_diff_truth"]) < abs(results["naif"]["tpr_diff_truth"])


def test_test_contrefactuel_du_modele_naif(prep):
    assert fm.counterfactual_flip_rate(prep) > 0.05


def test_contrainte_plus_stricte_reduit_l_ecart(prep):
    tr = fm.tradeoff(prep, "eg_dp", [0.30, 0.02])
    assert abs(tr.dp_diff.iloc[1]) < abs(tr.dp_diff.iloc[0])
