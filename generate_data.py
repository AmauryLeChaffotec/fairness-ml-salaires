"""Génère un historique RH synthétique avec un biais d'augmentation : data/employes_exemple.csv

Cible historique : `augmentation_forte` (1 = augmentation ≥ 5 % lors du dernier cycle).
Le mérite réel dépend de la performance, de l'ancienneté, du poste et du diplôme. On y injecte VOLONTAIREMENT :
  - un biais direct : à mérite égal, les femmes sont moins souvent augmentées ;
  - un biais indirect : le temps partiel et les interruptions de carrière (plus fréquents chez les femmes)
    sont pénalisés alors qu'ils ne mesurent aucun mérite. Ce sont des « proxys » du genre.
La colonne `augmentation_forte_equitable` est la décision qu'on aurait prise SANS ces biais. Elle n'existe que dans
des données synthétiques (c'est la vérité terrain, utile pour la démonstration).
Les données sont fictives : elles servent uniquement à démontrer l'application.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 21
N = 4000

# poste -> (niveau hiérarchique, département, part des femmes parmi les recrutés, poids de fréquence)
POSTES = {
    "Assistant RH":   (0, "RH",         0.80, 8),
    "Comptable":      (1, "Finance",    0.65, 9),
    "Commercial":     (2, "Ventes",     0.40, 14),
    "Développeur":    (3, "IT",         0.25, 18),
    "Data Analyst":   (3, "IT",         0.40, 8),
    "Chef de projet": (5, "IT",         0.40, 10),
    "Responsable":    (6, "Management", 0.35, 7),
    "Directeur":      (7, "Direction",  0.25, 3),
}
DIPLOMES = ["Bac", "Bac+2", "Bac+3", "Bac+5", "Doctorat"]
VILLES = ["Paris", "Lyon", "Nantes", "Lille", "Toulouse"]


def main() -> None:
    rng = np.random.default_rng(SEED)

    noms = list(POSTES)
    poids = np.array([POSTES[p][3] for p in noms], dtype=float)
    poste = rng.choice(noms, size=N, p=poids / poids.sum())
    niveau = np.array([POSTES[p][0] for p in poste])

    femme = rng.random(N) < np.array([POSTES[p][2] for p in poste])
    genre = np.where(femme, "F", "H")

    diplome_idx = rng.choice(len(DIPLOMES), size=N, p=[0.10, 0.20, 0.25, 0.38, 0.07])
    ville = rng.choice(VILLES, size=N, p=[0.35, 0.20, 0.15, 0.15, 0.15])
    anciennete = np.clip(rng.gamma(shape=2.2, scale=3.5, size=N), 0, 35).round(1)
    age = np.clip(23 + anciennete + rng.normal(4, 4, N), 21, 64).round().astype(int)
    performance = rng.choice([1, 2, 3, 4, 5], size=N, p=[0.04, 0.16, 0.45, 0.28, 0.07])
    equipe = np.where(niveau >= 5, rng.integers(2, 25, N), 0)

    # proxys du genre : sans lien avec le mérite
    temps_partiel = (rng.random(N) < np.where(femme, 0.22, 0.04)).astype(int)
    a_interruption = rng.random(N) < np.where(femme, 0.40, 0.08)
    interruption = np.where(a_interruption, rng.gamma(2.0, 6.0, N), 0).round().astype(int)

    merite = (0.85 * (performance - 3) + 0.05 * np.minimum(anciennete, 15) + 0.22 * niveau
              + 0.15 * diplome_idx + rng.normal(0, 0.55, N))
    seuil = np.quantile(merite, 0.70)                       # 30 % d'augmentations fortes sans biais
    equitable = merite > seuil

    score_historique = merite - 0.55 * femme - 0.45 * temps_partiel - 0.012 * interruption
    historique = score_historique > np.quantile(score_historique, 0.70)

    df = pd.DataFrame({
        "id_employe": [f"E{i:04d}" for i in range(1, N + 1)],
        "genre": genre,
        "age": age,
        "poste": poste,
        "departement": [POSTES[p][1] for p in poste],
        "niveau_diplome": [DIPLOMES[i] for i in diplome_idx],
        "anciennete_ans": anciennete,
        "performance": performance,
        "nb_personnes_managees": equipe,
        "ville": ville,
        "temps_partiel": temps_partiel,
        "interruption_carriere_mois": interruption,
        "augmentation_forte": historique.astype(int),
        "augmentation_forte_equitable": equitable.astype(int),
    })

    out = Path(__file__).parent / "data" / "employes_exemple.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} lignes écrites dans {out}")
    print(df.groupby("genre")[["augmentation_forte", "augmentation_forte_equitable"]].mean().round(3))


if __name__ == "__main__":
    main()
