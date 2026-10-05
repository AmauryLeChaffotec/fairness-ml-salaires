"""Fairness en apprentissage automatique pour les décisions RH (Fairlearn).

Lancer :  streamlit run app.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import fairness as fm

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Fairness ML · Décisions RH", page_icon="🧭", layout="wide")

EXAMPLE = Path(__file__).parent / "data" / "employes_exemple.csv"
C_A, C_B, C_OK, C_GREY = "#E07A1F", "#2A7F9E", "#2A7F9E", "#6B7280"
GENRE_HINTS = ("genre", "sexe", "gender", "sex")
ID_HINTS = ("id", "matricule", "employe", "employee")
TARGET_HINTS = ("augmentation", "promotion", "decision", "décision", "bonus")
TRUTH_HINTS = ("equitable", "équitable", "verite", "vérité", "truth", "sans_biais")
MERIT_HINTS = ("performance", "evaluation", "évaluation", "note", "merite", "mérite")


@st.cache_data(show_spinner=False)
def load_csv(source) -> pd.DataFrame:
    return pd.read_csv(source, sep=None, engine="python")


def first_match(columns, hints, default=0):
    return next((i for i, c in enumerate(columns) if any(h in c.lower() for h in hints)), default)


def merit_levels(s: pd.Series):
    """Niveaux de mérite : quintiles pour une variable continue, valeurs sinon. Renvoie (étiquettes, ordre)."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
        q = pd.qcut(s, 5, duplicates="drop")
        return q.astype(str), [str(c) for c in q.cat.categories]
    labels = s.astype(str)
    order = [str(v) for v in sorted(s.unique())] if pd.api.types.is_numeric_dtype(s) else sorted(labels.unique())
    return labels, order


def pct(v, digits=1):
    return "n/d" if v is None or pd.isna(v) else f"{v * 100:.{digits}f} %"


def pts(v):
    return "n/d" if v is None or pd.isna(v) else f"{v * 100:+.1f} pts"


@st.cache_data(show_spinner="Entraînement des six modèles…")
def run_models(df, y, truth, sens, a, b, feats, base, bound, test_size):
    prep = fm.prepare(df, y, sens, a, b, list(feats), truth, test_size)
    preds = {k: fm.fit_predict(k, prep, base, bound) for k in fm.MODEL_KINDS}
    metrics = {k: fm.fairness_metrics(prep.y_te, p, prep.a_te, a, b, prep.truth_te) for k, p in preds.items()}
    flip = fm.counterfactual_flip_rate(prep, base)
    return prep, preds, metrics, flip


@st.cache_data(show_spinner="Calcul du compromis équité / précision…")
def run_tradeoff(df, y, truth, sens, a, b, feats, base, kind, test_size):
    prep = fm.prepare(df, y, sens, a, b, list(feats), truth, test_size)
    return fm.tradeoff(prep, kind, [0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01], base)


@st.cache_data(show_spinner=False)
def run_proxy(df, feats, sens, a):
    return fm.proxy_audit(df, list(feats), sens, a)


# --------------------------------------------------------------------------- barre latérale
st.sidebar.title("🧭 Fairness ML")
st.sidebar.caption("Fairlearn · décisions RH")

upload = st.sidebar.file_uploader("Historique RH (CSV)", type=["csv"])
if upload is not None:
    data = load_csv(upload)
elif EXAMPLE.exists():
    data = load_csv(EXAMPLE)
    st.sidebar.info("Fichier d'exemple chargé (données fictives).")
else:
    st.error("Aucun fichier d'exemple trouvé. Lancez `python generate_data.py` ou importez un CSV.")
    st.stop()

cols = list(data.columns)
num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]
bin_cols = [c for c in num_cols if set(data[c].dropna().unique()) <= {0, 1}]

st.sidebar.subheader("Colonnes")
target = st.sidebar.selectbox("Décision historique (cible)", num_cols,
                              index=first_match(num_cols, TARGET_HINTS, len(num_cols) - 1),
                              help="Ex. augmentation forte (0/1), ou un montant que l'on binarise avec un seuil.")
thr = None
if not set(data[target].dropna().unique()) <= {0, 1}:
    lo, hi = float(data[target].min()), float(data[target].max())
    thr = st.sidebar.slider("Seuil : décision positive si ≥", lo, hi, float(data[target].median()))
sens = st.sidebar.selectbox("Attribut sensible (genre)", cols, index=first_match(cols, GENRE_HINTS))
values = sorted(data[sens].dropna().astype(str).unique())
if len(values) < 2:
    st.error("La colonne sensible doit contenir au moins deux valeurs.")
    st.stop()
default_a = next((i for i, v in enumerate(values) if v.lower() in ("f", "femme", "femmes", "female", "w")), 0)
group_a = st.sidebar.selectbox("Groupe étudié (A)", values, index=default_a)
group_b = st.sidebar.selectbox("Groupe de référence (B)", [v for v in values if v != group_a])
truth_options = ["(aucune)"] + [c for c in bin_cols if c != target]
truth_col = st.sidebar.selectbox("Vérité terrain équitable (facultatif)", truth_options,
                                 index=first_match(truth_options[1:], TRUTH_HINTS, -1) + 1,
                                 help="La décision qu'on aurait prise sans biais. Elle n'existe que dans des données synthétiques ; "
                                      "elle permet de mesurer la précision par rapport à autre chose qu'un historique biaisé.")
id_col = st.sidebar.selectbox("Identifiant", ["(index)"] + cols, index=first_match(cols, ID_HINTS, -1) + 1)

excluded = {target, sens, id_col} | ({truth_col} if truth_col != "(aucune)" else set())
candidates = [c for c in cols if c not in excluded and c not in ("augmentation_forte_equitable",)]
features = st.sidebar.multiselect("Variables explicatives", candidates, default=candidates,
                                  help="Le genre est exclu. Attention aux proxys (temps partiel, interruptions de carrière…).")
if not features:
    st.warning("Sélectionnez au moins une variable explicative.")
    st.stop()

st.sidebar.subheader("Modèles")
base = st.sidebar.radio("Modèle de base", list(fm.BASES), help="Le gradient boosting est plus souple mais nettement plus lent à contraindre (environ une minute pour les six modèles).")
bound = st.sidebar.slider("Écart de parité toléré (points)", 0, 20, 2, 1,
                          help="Borne de la contrainte Fairlearn. Plus elle est basse, plus la contrainte est stricte.") / 100
test_size = st.sidebar.slider("Part de test (%)", 20, 40, 30, 5) / 100

# --------------------------------------------------------------------------- préparation
need = [target, sens] + features + ([truth_col] if truth_col != "(aucune)" else [])
df = data.dropna(subset=need)
df = df[df[sens].astype(str).isin([group_a, group_b])].reset_index(drop=True)
try:
    y_all = fm.binarize(df[target], thr)
except ValueError as e:
    st.error(str(e))
    st.stop()
truth_all = None if truth_col == "(aucune)" else fm.binarize(df[truth_col], None)
grp = df[sens].astype(str).to_numpy()
for g in (group_a, group_b):
    if len(df[grp == g]) < 50 or y_all[grp == g].sum() < 20 or (1 - y_all[grp == g]).sum() < 20:
        st.error(f"Le groupe {g} doit compter au moins 50 personnes dont 20 décisions positives et 20 négatives.")
        st.stop()

prep, preds, metrics, flip_rate = run_models(df, y_all, truth_all, sens, group_a, group_b, tuple(features), base, bound, test_size)
has_truth = prep.truth_te is not None
hist_sr = {g: float(y_all[grp == g].mean()) for g in (group_a, group_b)}
hist_ratio = min(hist_sr.values()) / max(hist_sr.values())

st.title("Fairness ML : décisions RH sans reproduire le biais")
st.caption(
    f"{len(df):,} personnes ({(grp == group_a).sum()} {group_a} · {(grp == group_b).sum()} {group_b}) · "
    f"{len(features)} variables · modèle de base : {base} · test : {len(prep.y_te)} personnes".replace(",", " "))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Biais historique", "🤖 Six modèles", "⚖️ Compromis équité / précision", "🔍 Analyse fine", "📐 Méthode & limites"])

# --------------------------------------------------------------------------- 1. biais historique
with tab1:
    auc, proxy_tab = run_proxy(df, tuple(features), sens, group_a)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Décisions positives · {group_a}", pct(hist_sr[group_a]))
    c2.metric(f"Décisions positives · {group_b}", pct(hist_sr[group_b]))
    c3.metric("Ratio de sélection", f"{hist_ratio:.2f}", "règle des 80 % : " + ("respectée" if hist_ratio >= 0.8 else "non respectée"),
              delta_color="normal" if hist_ratio >= 0.8 else "inverse")
    c4.metric("Genre prédictible sans le connaître", f"AUC {auc:.2f}", "0.5 = aucun proxy · 1 = proxy parfait", delta_color="off")

    l, r = st.columns(2)
    with l:
        st.subheader("Taux de décisions positives")
        fig = go.Figure(go.Bar(x=[group_a, group_b], y=[hist_sr[group_a] * 100, hist_sr[group_b] * 100],
                               marker_color=[C_A, C_B], text=[pct(hist_sr[group_a]), pct(hist_sr[group_b])], textposition="outside"))
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="%", yaxis_range=[0, max(hist_sr.values()) * 130])
        st.plotly_chart(fig, width="stretch")
    with r:
        st.subheader("À mérite comparable")
        merit_options = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 12]
        merit = st.selectbox("Variable de mérite", merit_options, index=first_match(merit_options, MERIT_HINTS, 0))
        level, order = merit_levels(df[merit])
        tmp = pd.DataFrame({"niveau": level, "groupe": grp, "y": y_all}).groupby(["niveau", "groupe"], as_index=False).agg(
            taux=("y", "mean"), n=("y", "size"))
        fig = px.line(tmp, x="niveau", y="taux", color="groupe", markers=True, category_orders={"niveau": order},
                      color_discrete_map={group_a: C_A, group_b: C_B}, hover_data=["n"],
                      labels={"niveau": merit, "taux": "Décisions positives", "groupe": "Groupe"})
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_tickformat=".0%")
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=order)
        st.plotly_chart(fig, width="stretch")
    st.caption("Si la courbe d'un groupe reste sous l'autre pour un même niveau de mérite, l'écart ne s'explique pas par la performance.")

    st.subheader("Proxys : ce qui trahit le genre")
    st.caption("Association normalisée (information mutuelle / entropie du genre) entre chaque variable et le genre.")
    fig = go.Figure(go.Bar(y=proxy_tab.variable[::-1], x=proxy_tab.association[::-1], orientation="h", marker_color=C_GREY))
    fig.update_layout(height=max(260, 34 * len(proxy_tab)), margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Association avec le genre")
    st.plotly_chart(fig, width="stretch")
    if auc >= 0.7:
        st.warning(f"Un modèle retrouve le genre avec une AUC de {auc:.2f} sans jamais le voir. Retirer simplement la colonne "
                   "genre ne suffit donc pas : l'information passe par les autres variables.")

# --------------------------------------------------------------------------- 2. six modèles
with tab2:
    names = fm.MODEL_KINDS
    rows = []
    for k, label in names.items():
        m = metrics[k]
        row = {"Modèle": label, f"Sél. {group_a}": m["sr_a"], f"Sél. {group_b}": m["sr_b"],
               "Écart parité": m["dp_diff"], "Ratio": m["dp_ratio"], "Écart ég. chances": m["tpr_diff"],
               "Préc. historique": m["accuracy"]}
        if has_truth:
            row["Préc. vérité"] = m["accuracy_truth"]
            row["Écart ég. ch. vérité"] = m["tpr_diff_truth"]
        rows.append(row)
    tbl = pd.DataFrame(rows)
    shown = tbl.copy()
    for c in tbl.columns[1:]:
        shown[c] = tbl[c].map(lambda v, c=c: "n/d" if pd.isna(v) else (f"{v:.2f}" if c == "Ratio" else (pts(v) if c.startswith("Écart") else pct(v))))
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("Sél. = taux de décisions positives · Écart = groupe A − groupe B · Ratio = taux le plus bas / taux le plus haut · "
               "Préc. = part de décisions identiques à l'historique, ou à la vérité terrain équitable lorsqu'elle existe.")

    l, r = st.columns(2)
    labels = ["Historique"] + list(names.values())
    short = ["Historique", "Naïf", "Sans genre", "FL parité", "FL ég. chances", "Post parité", "Post ég. chances"]
    with l:
        st.subheader("Taux de sélection par groupe")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=short, y=[hist_sr[group_a] * 100] + [metrics[k]["sr_a"] * 100 for k in names], name=group_a, marker_color=C_A))
        fig.add_trace(go.Bar(x=short, y=[hist_sr[group_b] * 100] + [metrics[k]["sr_b"] * 100 for k in names], name=group_b, marker_color=C_B))
        fig.update_layout(barmode="group", height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="% de décisions positives",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    with r:
        st.subheader("Ratio de sélection (règle des 80 %)")
        ratios = [hist_ratio] + [metrics[k]["dp_ratio"] for k in names]
        fig = go.Figure(go.Bar(x=short, y=ratios, marker_color=[C_OK if v >= 0.8 else C_A for v in ratios],
                               text=[f"{v:.2f}" for v in ratios], textposition="outside"))
        fig.add_hline(y=0.8, line_dash="dash", line_color="gray", annotation_text="seuil 0,8")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_range=[0, 1.12], yaxis_title="min / max des taux")
        st.plotly_chart(fig, width="stretch")

    m_naif, m_sg, m_dp, m_eo = metrics["naif"], metrics["sans_genre"], metrics["eg_dp"], metrics["eg_eo"]
    st.markdown(
        f"- **Naïf** : ratio de sélection **{m_naif['dp_ratio']:.2f}** contre {hist_ratio:.2f} dans l'historique. "
        f"Le modèle reproduit le biais, et **{flip_rate * 100:.0f} %** de ses décisions changent si l'on inverse uniquement le genre.\n"
        f"- **Sans le genre** : ratio **{m_sg['dp_ratio']:.2f}**. Retirer la colonne ne suffit pas, les proxys portent le biais.\n"
        f"- **Parité démographique (Fairlearn)** : ratio **{m_dp['dp_ratio']:.2f}** pour une précision de {pct(m_dp['accuracy'])} "
        f"par rapport à l'historique" + (f" et de **{pct(m_dp['accuracy_truth'])} par rapport à la vérité terrain**" if has_truth else "") + ".\n"
        f"- **Égalité des chances** : écart de parité de **{pts(m_eo['dp_diff'])}** malgré une contrainte satisfaite. "
        "Elle se mesure par rapport aux décisions passées : si elles sont biaisées, elle préserve le biais.")
    if has_truth:
        st.info("Par rapport à l'historique, un modèle équitable paraît moins précis, car on le compare à des décisions biaisées. "
                "Par rapport à la vérité terrain (les décisions sans biais), c'est l'inverse : voir la colonne « Précision / vérité terrain ».")

# --------------------------------------------------------------------------- 3. compromis
with tab3:
    kind = st.radio("Contrainte", ["eg_dp", "eg_eo"], format_func=lambda k: names[k], horizontal=True)
    tr = run_tradeoff(df, y_all, truth_all, sens, group_a, group_b, tuple(features), base, kind, test_size)
    l, r = st.columns(2)
    with l:
        st.subheader("Écart obtenu selon la contrainte")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr.borne * 100, y=tr.dp_diff.abs() * 100, mode="lines+markers", name="|Écart de parité|", line=dict(color=C_A)))
        fig.add_trace(go.Scatter(x=tr.borne * 100, y=tr.tpr_diff.abs() * 100, mode="lines+markers", name="|Écart égalité des chances|", line=dict(color=C_B)))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Borne de la contrainte (points)",
                          yaxis_title="Écart obtenu (points)", legend=dict(orientation="h", y=1.1), xaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with r:
        st.subheader("Précision selon la contrainte")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr.borne * 100, y=tr.accuracy * 100, mode="lines+markers", name="Par rapport à l'historique", line=dict(color=C_GREY)))
        if has_truth:
            fig.add_trace(go.Scatter(x=tr.borne * 100, y=tr.accuracy_truth * 100, mode="lines+markers", name="Par rapport à la vérité terrain", line=dict(color=C_OK)))
        fig.add_hline(y=metrics["sans_genre"]["accuracy"] * 100, line_dash="dot", line_color="gray", annotation_text="modèle sans genre")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Borne de la contrainte (points)",
                          yaxis_title="Précision (%)", legend=dict(orientation="h", y=1.1), xaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    show = tr[["borne", "sr_a", "sr_b", "dp_diff", "dp_ratio", "tpr_diff", "accuracy"] + (["accuracy_truth"] if has_truth else [])].copy()
    show.columns = ["Borne", f"Sélection {group_a}", f"Sélection {group_b}", "Écart de parité", "Ratio", "Écart ég. chances", "Précision / historique"] + (
        ["Précision / vérité"] if has_truth else [])
    for c in show.columns:
        show[c] = show[c].map(lambda v, c=c: f"{v:.2f}" if c == "Ratio" else (pts(v) if c.startswith("Écart") else pct(v)))
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Lecture : plus la borne est basse, plus la contrainte est stricte. Le coût en précision se mesure par rapport à l'historique ; "
               "avec une vérité terrain, on voit si la contrainte rapproche vraiment le modèle des décisions équitables.")

# --------------------------------------------------------------------------- 4. analyse fine
with tab4:
    choice = st.selectbox("Modèle à examiner", list(names), index=2, format_func=names.get)
    pred = preds[choice]
    merit_options = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 12]
    merit2 = st.selectbox("Variable de mérite", merit_options, index=first_match(merit_options, MERIT_HINTS, 0), key="merit2")
    level, order = merit_levels(df[merit2])
    level = level.iloc[prep.idx_te].reset_index(drop=True)
    tmp = pd.concat([
        pd.DataFrame({"niveau": level, "groupe": prep.a_te, "taux": prep.y_te, "source": "Historique"}),
        pd.DataFrame({"niveau": level, "groupe": prep.a_te, "taux": pred, "source": names[choice]})])
    agg = tmp.groupby(["niveau", "groupe", "source"], as_index=False).taux.mean()
    agg["serie"] = agg.groupe + " · " + agg.source
    fig = px.line(agg, x="niveau", y="taux", color="serie", markers=True, category_orders={"niveau": order},
                  labels={"niveau": merit2, "taux": "Décisions positives", "serie": ""})
    fig.update_traces(line=dict(dash="solid"))
    for trace in fig.data:
        if "Historique" in trace.name:
            trace.update(line=dict(dash="dot"))
        trace.update(line_color=C_A if trace.name.startswith(group_a) else C_B)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=order)
    st.plotly_chart(fig, width="stretch")
    st.caption("Pointillé : décisions historiques. Trait plein : décisions du modèle choisi, sur l'échantillon de test.")

    c1, c2 = st.columns(2)
    c1.metric("Décisions du modèle naïf qui changent si on inverse le genre", f"{flip_rate * 100:.1f} %",
              "test contrefactuel", delta_color="off")
    c2.metric("Parité du modèle choisi", f"{metrics[choice]['dp_ratio']:.2f}", f"écart {pts(metrics[choice]['dp_diff'])}", delta_color="off")
    st.caption("Le test contrefactuel ne s'applique qu'à un modèle qui utilise le genre. Un modèle « sans genre » le réussit toujours "
               "(0 % de décisions changées) tout en restant biaisé via les proxys : ce test ne suffit donc jamais seul.")

    ids = df[id_col].astype(str).to_numpy() if id_col != "(index)" else df.index.astype(str).to_numpy()
    out = pd.DataFrame({"id": ids[prep.idx_te], "groupe": prep.a_te, "decision_historique": prep.y_te, "decision_modele": pred})
    if has_truth:
        out["decision_equitable"] = prep.truth_te
    st.download_button("Télécharger les décisions du modèle (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="decisions_modele.csv", mime="text/csv")

# --------------------------------------------------------------------------- 5. méthode
with tab5:
    st.markdown(
        "**Le problème.** Un modèle entraîné sur des décisions passées biaisées apprend ce biais, y compris sans voir le "
        "genre, via des variables corrélées (temps partiel, interruptions de carrière, poste…).\n\n"
        "**Deux critères d'équité de groupe**")
    st.latex(r"\text{Parité démographique : } P(\hat Y=1\mid A=a)=P(\hat Y=1\mid A=b)")
    st.latex(r"\text{Égalité des chances : } P(\hat Y=1\mid Y=1,A=a)=P(\hat Y=1\mid Y=1,A=b)")
    st.markdown(
        "**Deux familles d'algorithmes Fairlearn**\n"
        "- **ExponentiatedGradient (in-processing)** : réentraîne le modèle en pénalisant mathématiquement le lien résiduel "
        "entre prédiction et genre. Le genre n'est utilisé qu'à l'entraînement, pas à la décision.\n"
        "- **ThresholdOptimizer (post-traitement)** : ajuste un seuil par groupe sur un modèle existant. "
        "Il exige de connaître le genre au moment de la décision, ce qui peut poser un problème juridique de traitement différencié.\n\n"
        "**Règle des 80 %.** Un ratio de sélection (taux le plus bas / taux le plus haut) inférieur à 0,8 est un signal d'impact disproportionné.")
    st.warning(
        "**Limites importantes.**\n"
        "- **L'égalité des chances suppose que les étiquettes sont justes.** Avec un historique biaisé, elle préserve le biais ; "
        "la parité démographique est alors plus adaptée, mais elle impose l'égalité des taux quel que soit le mérite.\n"
        "- **La précision mesurée sur l'historique pénalise un modèle équitable**, puisque l'historique est biaisé.\n"
        "- Ces critères sont **des statistiques de groupe** : ils ne garantissent pas l'équité de chaque décision individuelle.\n"
        "- Le choix d'un critère d'équité est **normatif**, pas technique : à décider avec vos équipes RH et juridiques.\n"
        "- Le modèle doit **assister** une décision humaine, pas la remplacer.")
