"""
oasis1_motion_categories.py
----------------------------
Adresse la question de Sylvain : "déterminer catégories de mouvement (type artefacts)"

Approche : clustering k-means sur les scores Agitation + variables morphométriques
pour identifier si des groupes naturels émergent au-delà du simple gradient de score.
Produit aussi une analyse des sujets outliers (même score, épaisseur très différente).

Usage :
    python oasis1_motion_categories.py \
        --scores   datasets/oasis1/results_mni/oasis1_scores_mni.csv \
        --fs_dir   /project/hippocampus/common/datasets/OASIS1_BIDS/processed_freesurfer7.4.1 \
        --participants /project/hippocampus/common/datasets/OASIS1_BIDS/raw_data_bids/participants.tsv \
        --out_dir  datasets/oasis1/results_mni \
        --id_format oasis1
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def parse_aparc_stats(filepath):
    regions = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                regions[parts[0]] = float(parts[4])
            except ValueError:
                continue
    return regions


def load_aparc(fs_dir, subjects):
    fs_dir = Path(fs_dir)
    rows = []
    for sub in subjects:
        row = {"sub": sub}
        ok = True
        for hemi in ["lh", "rh"]:
            path = fs_dir / sub / "stats" / f"{hemi}.aparc.stats"
            if not path.exists():
                ok = False
                break
            for region, thick in parse_aparc_stats(path).items():
                row[f"{hemi}_{region}"] = thick
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)


def plot_clusters(df, out_path, label_a="PC1", label_b="PC2"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PCA coloré par cluster
    sc = axes[0].scatter(df[label_a], df[label_b],
                         c=df["cluster"], cmap="tab10", alpha=0.7, s=30)
    axes[0].set_xlabel(label_a)
    axes[0].set_ylabel(label_b)
    axes[0].set_title("Clusters (k-means) dans l'espace PCA")
    plt.colorbar(sc, ax=axes[0])

    # PCA coloré par score de mouvement
    sc2 = axes[1].scatter(df[label_a], df[label_b],
                          c=df["motion"], cmap="RdYlBu_r", alpha=0.7, s=30)
    axes[1].set_xlabel(label_a)
    axes[1].set_ylabel(label_b)
    axes[1].set_title("Score de mouvement dans l'espace PCA")
    plt.colorbar(sc2, ax=axes[1], label="Score Agitation (mm)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Figure clusters : {out_path}")


def identify_outliers(df, out_path):
    """
    Sujets avec résidu élevé : même score de mouvement, épaisseur très différente.
    Ce sont les cas "résistants" et "sensibles" au mouvement.
    """
    slope, intercept, *_ = stats.linregress(df["motion"], df["mean_thickness"])
    df = df.copy()
    df["predicted_thickness"] = slope * df["motion"] + intercept
    df["residual"] = df["mean_thickness"] - df["predicted_thickness"]
    df["residual_z"] = stats.zscore(df["residual"])

    threshold = 2.0
    df["outlier_type"] = "normal"
    df.loc[df["residual_z"] > threshold,  "outlier_type"] = "resistant"   # épaisseur > attendue
    df.loc[df["residual_z"] < -threshold, "outlier_type"] = "sensitive"   # épaisseur < attendue

    print(f"\n  Sujets résistants (épaisseur > attendue) : {(df['outlier_type']=='resistant').sum()}")
    print(f"  Sujets sensibles  (épaisseur < attendue) : {(df['outlier_type']=='sensitive').sum()}")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"normal": "#91bfdb", "resistant": "#1a9641", "sensitive": "#d73027"}
    for otype, grp in df.groupby("outlier_type"):
        ax.scatter(grp["motion"], grp["mean_thickness"],
                   c=colors[otype], label=otype, alpha=0.7, s=40)
    x_range = np.linspace(df["motion"].min(), df["motion"].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, "k--", lw=1.5, label="régression")
    ax.set_xlabel("Score Agitation (mm)")
    ax.set_ylabel("Épaisseur corticale moyenne (mm)")
    ax.set_title("Sujets résistants vs sensibles au mouvement\nOASIS1 — scores MNI")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Figure outliers : {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores",       required=True)
    parser.add_argument("--fs_dir",       required=True)
    parser.add_argument("--participants", required=True)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--id_format",    default=None)
    parser.add_argument("--n_clusters",   type=int, default=3)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Scores
    scores = pd.read_csv(args.scores)
    if "sub" not in scores.columns:
        scores = scores.iloc[:, 1:3]
        scores.columns = ["sub", "motion"]

    # Démographie
    demo = pd.read_csv(args.participants, sep="\t")
    if args.id_format == "oasis1":
        demo["sub"] = (demo["participant_id"].astype(str)
                       .str.extract(r"sub-OASIS1(\d+)", expand=False)
                       .apply(lambda x: f"sub-{x}" if pd.notna(x) else np.nan))
    else:
        demo["sub"] = demo["participant_id"].astype(str)
    age_col = next(c for c in demo.columns if "age" in c.lower())
    demo["age"] = pd.to_numeric(demo[age_col], errors="coerce")
    sex_col = next(c for c in demo.columns if c.lower() in ["sex", "gender"])
    demo["sex_num"] = (demo[sex_col].astype(str).str.upper() == "M").astype(int)

    merged = scores.merge(demo[["sub", "age", "sex_num"]], on="sub", how="inner")

    # APARC
    print("Lecture aparc.stats...")
    aparc = load_aparc(args.fs_dir, merged["sub"].tolist())
    region_cols = [c for c in aparc.columns if c != "sub"]

    full = merged.merge(aparc, on="sub", how="inner")
    full["mean_thickness"] = full[region_cols].mean(axis=1)
    print(f"  {len(full)} sujets")

    # ---- Clustering ----
    print(f"\nClustering k-means (k={args.n_clusters})...")
    features = full[["motion", "mean_thickness", "age"]].dropna()
    idx = features.index
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    full.loc[idx, "cluster"] = km.fit_predict(X)

    # PCA pour visualisation
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    full.loc[idx, "PC1"] = coords[:, 0]
    full.loc[idx, "PC2"] = coords[:, 1]

    print(f"  Variance expliquée PC1+PC2 : {pca.explained_variance_ratio_[:2].sum():.1%}")
    print("\n  Caractéristiques par cluster :")
    print(full.groupby("cluster")[["motion", "mean_thickness", "age"]].median().round(3).to_string())

    plot_clusters(full.dropna(subset=["PC1", "cluster"]),
                  out / "oasis1_clusters.png")

    # ---- Outliers ----
    print("\nIdentification des sujets résistants/sensibles...")
    full_with_outliers = identify_outliers(full, out / "oasis1_outliers.png")
    full_with_outliers[["sub", "motion", "mean_thickness", "residual", "residual_z",
                         "outlier_type", "cluster"]].to_csv(
        out / "oasis1_subject_categories.csv", index=False
    )

    print(f"\nTerminé. Fichiers dans {out}")


if __name__ == "__main__":
    main()
