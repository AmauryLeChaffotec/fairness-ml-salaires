"""Fairness en apprentissage automatique pour les décisions RH (Fairlearn).

Scénario : l'historique des décisions (ex. augmentation forte oui/non) est biaisé. Un modèle entraîné dessus
reproduit le biais. Ce module entraîne six modèles et mesure leur équité :

  naif         avec le genre comme variable                         -> reproduit le biais
  sans_genre   « équité par ignorance » : sans le genre             -> le biais passe par les proxys
  eg_dp        Fairlearn ExponentiatedGradient + parité démographique   (in-processing)
  eg_eo        Fairlearn ExponentiatedGradient + égalité des chances    (in-processing)
  to_dp        Fairlearn ThresholdOptimizer + parité démographique      (post-traitement)
  to_eo        Fairlearn ThresholdOptimizer + égalité des chances       (post-traitement)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
    MetricFrame,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient, TruePositiveRateParity
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_KINDS = {
    "naif": "Naïf (avec le genre)",
    "sans_genre": "Sans le genre",
    "eg_dp": "Fairlearn · parité démographique",
    "eg_eo": "Fairlearn · égalité des chances",
    "to_dp": "Post-traitement · parité démographique",
    "to_eo": "Post-traitement · égalité des chances",
}

BASES = {
    "Régression logistique": lambda: LogisticRegression(max_iter=2000),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=120, max_depth=3, random_state=0),
}


# --------------------------------------------------------------------------- préparation
@dataclass
class Prepared:
    X_tr: np.ndarray          # variables objectives (sans le genre)
    X_te: np.ndarray
    Xa_tr: np.ndarray         # variables objectives + indicateur de genre (dernière colonne, 0/1)
    Xa_te: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    a_tr: np.ndarray          # groupe de chaque personne (chaîne)
    a_te: np.ndarray
    truth_te: np.ndarray | None
    idx_te: np.ndarray        # position des lignes de test dans le DataFrame d'origine
    group_a: str
    group_b: str


def binarize(series: pd.Series, threshold: float | None) -> np.ndarray:
    """Cible binaire : colonne déjà binaire, ou valeur ≥ seuil."""
    s = pd.to_numeric(series, errors="coerce")
    if threshold is None:
        if set(s.dropna().unique()) - {0, 1}:
            raise ValueError("La cible n'est pas binaire : indiquez un seuil.")
        return s.to_numpy(float).astype(int)
    return (s >= threshold).to_numpy().astype(int)


def prepare(df: pd.DataFrame, y: np.ndarray, sens_col: str, group_a: str, group_b: str, features,
            truth: np.ndarray | None = None, test_size: float = 0.3, seed: int = 42) -> Prepared:
    a = df[sens_col].astype(str).to_numpy()
    strat = np.char.add(a.astype(str), y.astype(str))
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed, stratify=strat)

    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    pre = ColumnTransformer(
        [("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    X_tr = pre.fit_transform(df.iloc[idx_tr][features])
    X_te = pre.transform(df.iloc[idx_te][features])
    flag = lambda idx: (a[idx] == group_a).astype(float)[:, None]
    return Prepared(
        X_tr, X_te, np.hstack([X_tr, flag(idx_tr)]), np.hstack([X_te, flag(idx_te)]),
        y[idx_tr], y[idx_te], a[idx_tr], a[idx_te], None if truth is None else truth[idx_te], idx_te,
        group_a, group_b)


# --------------------------------------------------------------------------- modèles
def fit_predict(kind: str, prep: Prepared, base: str = "Régression logistique", bound: float = 0.02,
                seed: int = 0) -> np.ndarray:
    """Entraîne un modèle sur l'échantillon d'entraînement et renvoie ses décisions (0/1) sur le test."""
    make = BASES[base]
    if kind == "naif":
        return make().fit(prep.Xa_tr, prep.y_tr).predict(prep.Xa_te)
    if kind == "sans_genre":
        return make().fit(prep.X_tr, prep.y_tr).predict(prep.X_te)
    if kind in ("eg_dp", "eg_eo"):
        constraint = DemographicParity(difference_bound=bound) if kind == "eg_dp" else TruePositiveRateParity(difference_bound=bound)
        mitigator = ExponentiatedGradient(make(), constraints=constraint, max_iter=25 if base == "Gradient boosting" else 50)
        mitigator.fit(prep.X_tr, prep.y_tr, sensitive_features=prep.a_tr)
        return np.asarray(mitigator.predict(prep.X_te, random_state=seed)).astype(int)
    if kind in ("to_dp", "to_eo"):
        fitted = make().fit(prep.X_tr, prep.y_tr)
        opt = ThresholdOptimizer(
            estimator=fitted, prefit=True, predict_method="predict_proba", objective="accuracy_score",
            constraints="demographic_parity" if kind == "to_dp" else "true_positive_rate_parity")
        opt.fit(prep.X_tr, prep.y_tr, sensitive_features=prep.a_tr)
        return np.asarray(opt.predict(prep.X_te, sensitive_features=prep.a_te, random_state=seed)).astype(int)
    raise ValueError(f"Modèle inconnu : {kind}")


def counterfactual_flip_rate(prep: Prepared, base: str = "Régression logistique") -> float:
    """Part des décisions du modèle naïf qui changent si on inverse uniquement le genre."""
    model = BASES[base]().fit(prep.Xa_tr, prep.y_tr)
    flipped = prep.Xa_te.copy()
    flipped[:, -1] = 1.0 - flipped[:, -1]
    return float((model.predict(prep.Xa_te) != model.predict(flipped)).mean())


# --------------------------------------------------------------------------- métriques
def fairness_metrics(y_true, y_pred, a, group_a: str, group_b: str, truth=None) -> dict:
    """Équité et précision d'un jeu de décisions. Les écarts sont signés : groupe A − groupe B."""
    y_true, y_pred, a = np.asarray(y_true), np.asarray(y_pred), np.asarray(a)
    mf = MetricFrame(metrics={"sr": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
                     y_true=y_true, y_pred=y_pred, sensitive_features=a)
    by = mf.by_group
    sr_a, sr_b = float(by.loc[group_a, "sr"]), float(by.loc[group_b, "sr"])
    out = {
        "sr_a": sr_a, "sr_b": sr_b, "dp_diff": sr_a - sr_b,
        "dp_ratio": min(sr_a, sr_b) / max(sr_a, sr_b) if max(sr_a, sr_b) > 0 else np.nan,
        "tpr_a": float(by.loc[group_a, "tpr"]), "tpr_b": float(by.loc[group_b, "tpr"]),
        "tpr_diff": float(by.loc[group_a, "tpr"] - by.loc[group_b, "tpr"]),
        "eo_diff": float(equalized_odds_difference(y_true, y_pred, sensitive_features=a)),
        "accuracy": float((y_true == y_pred).mean()),
        "taux_positif": float(y_pred.mean()),
    }
    if truth is not None:
        truth = np.asarray(truth)
        mt = MetricFrame(metrics={"tpr": true_positive_rate, "fpr": false_positive_rate},
                         y_true=truth, y_pred=y_pred, sensitive_features=a).by_group
        out.update({
            "accuracy_truth": float((truth == y_pred).mean()),
            "tpr_diff_truth": float(mt.loc[group_a, "tpr"] - mt.loc[group_b, "tpr"]),
            "eo_diff_truth": float(equalized_odds_difference(truth, y_pred, sensitive_features=a)),
        })
    return out


def tradeoff(prep: Prepared, kind: str, bounds, base: str = "Régression logistique", seed: int = 0) -> pd.DataFrame:
    """Compromis équité / précision : un modèle par niveau de contrainte (plus la borne est basse, plus c'est strict)."""
    rows = []
    for b in bounds:
        pred = fit_predict(kind, prep, base, bound=float(b), seed=seed)
        m = fairness_metrics(prep.y_te, pred, prep.a_te, prep.group_a, prep.group_b, prep.truth_te)
        rows.append({"borne": float(b), **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- audit des proxys
def proxy_audit(df: pd.DataFrame, features, sens_col: str, group_a: str, seed: int = 0):
    """Dans quelle mesure les variables « objectives » trahissent-elles le genre ?

    Renvoie (AUC d'un modèle qui prédit le genre sans le connaître, tableau d'information mutuelle normalisée)."""
    y = (df[sens_col].astype(str) == group_a).to_numpy().astype(int)
    cols, discrete = {}, []
    for f in features:
        s = df[f]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            cols[f] = s.to_numpy(float); discrete.append(False)
        else:
            cols[f] = pd.factorize(s.astype(str))[0].astype(float); discrete.append(True)
    X = pd.DataFrame(cols)
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=seed)
    p = y.mean()
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p)) if 0 < p < 1 else np.nan
    table = pd.DataFrame({"variable": list(features), "association": mi / h}).sort_values("association", ascending=False)

    pre = ColumnTransformer([
        ("num", StandardScaler(), [f for f in features if pd.api.types.is_numeric_dtype(df[f])]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         [f for f in features if not pd.api.types.is_numeric_dtype(df[f])])])
    Xe = pre.fit_transform(df[list(features)])
    xtr, xte, ytr, yte = train_test_split(Xe, y, test_size=0.3, random_state=seed, stratify=y)
    auc = roc_auc_score(yte, HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(xtr, ytr).predict_proba(xte)[:, 1])
    return float(auc), table.reset_index(drop=True)
