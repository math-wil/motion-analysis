"""
oasis1_motion_variance.py
--------------------------
Adresse la question de Sylvain :
"Pour un même score de mouvement, différents niveaux de dégradation de l'épaisseur corticale"

Pour chaque bin de score Agitation, calcule :
- Distribution de l'épaisseur corticale (médiane, IQR, variance)
- Sujets "résistants" vs "sensibles" au mouvement pour un même score
- Régions APARC qui montrent le plus de variabilité intra-bin

Usage :
    python oasis1_motion_variance.py \
        --scores   datasets/oasis1/results_mni/oasis1_scores_mni.csv \
        --glm_csv  datasets/oasis1/results_mni/oasis1_glm_aparc_mni.csv \
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


def load_aparc_mean(fs_dir, subjects):
    """Charge épaisseur moyenne globale (lh+rh) pour chaque sujet."""
    fs_dir = Path(fs_dir)
    rows = []
    for sub in subjects:
        lh = fs_dir / sub / "stats" / "lh.aparc.stats"
        rh = fs_dir / sub / "stats" / "rh.aparc.stats"
        if not lh.exists() or not rh.exists():
            continue
        lh_r = parse_aparc_stats(lh)
        rh_r = parse_aparc_stats(rh)
        all_thick = list(lh_r.values()) + list(rh_r.values())
        rows.append({"sub": sub, "mean_thickness": np.mean(all_thick)})
        # Ajoute aussi par région pour l'analyse de variance
        row = {"sub": sub, "mean_thickness": np.mean(all_thick)}
        for region, v in lh_r.items():
            row[f"lh_{region}"] = v
        for region, v in rh_r.items():
            row[f"rh_{region}"] = v
        rows[-1] = row
    return pd.DataFrame(rows)


def assign_bins(motion_series, n_bins=5):
    """Découpe les scores en n_bins égaux, retourne labels et limites."""
    bins = pd.qcut(motion_series, q=n_bins, duplicates="drop")
    return bins


def plot_thickness_per_bin(df, out_path):
    """Boxplot épaisseur corticale par bin de mouvement."""
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = [grp["mean_thickness"].values for _, grp in df.groupby("motion_bin", observed=True)]
    labels = [str(k) for k in df.groupby("motion_bin", observed=True).groups.keys()]
    ax.boxplot(groups, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="#4575b4", alpha=0.6))
    ax.set_xlabel("Bin de score Agitation (mouvement croissant →)")
    ax.set_ylabel("Épaisseur corticale moyenne (mm)")
    ax.set_title("Distribution de l'épaisseur corticale par niveau de mouvement\nOASIS1 — scores MNI")
    plt.xticks(rotation=30, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Figure bins : {out_path}")


def plot_variance_per_bin(df, out_path):
    """Variance intra-bin de l'épaisseur — montre que même score = dégradation variable."""
    summary = df.groupby("motion_bin", observed=True)["mean_thickness"].agg(
        ["median", "std", "count"]
    ).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(summary))
    ax.bar(x, summary["std"], color="#d73027", alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b) for b in summary["motion_bin"]], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Écart-type épaisseur intra-bin (mm)")
    ax.set_title("Variabilité de l'épaisseur corticale pour un même niveau de mouvement\nOASIS1 — scores MNI")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Figure variance : {out_path}")


def find_variable_regions(df, region_cols, out_path, top_n=15):
    """
    Pour chaque région APARC, calcule la corrélation partielle mouvement~épaisseur
    et la variance résiduelle (= variabilité non expliquée par le mouvement).
    Identifie les régions les plus variables indépendamment du mouvement.
    """
    results = []
    for col in region_cols:
        sub = df[["motion", col]].dropna()
        if len(sub) < 30:
            continue
        r, p = stats.spearmanr(sub["motion"], sub[col])
        # Résidus après régression linéaire mouvement → épaisseur
        slope, intercept, *_ = stats.linregress(sub["motion"], sub[col])
        residuals = sub[col] - (slope * sub["motion"] + intercept)
        results.append({
            "region": col,
            "spearman_r": r,
            "p": p,
            "residual_std": residuals.std(),
        })
    res = pd.DataFrame(results).sort_values("residual_std", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top régions par corrélation avec mouvement
    top_corr = res.nsmallest(top_n, "spearman_r")
    axes[0].barh(top_corr["region"], top_corr["spearman_r"], color="#4575b4", alpha=0.8)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title(f"Top {top_n} régions — corrélation mouvement~épaisseur")
    axes[0].set_xlabel("Spearman r")

    # Top régions par variance résiduelle (= variabilité indépendante du mouvement)
    top_var = res.head(top_n)
    axes[1].barh(top_var["region"], top_var["residual_std"], color="#d73027", alpha=0.8)
    axes[1].set_title(f"Top {top_n} régions — variance résiduelle\n(variabilité non expliquée par le mouvement)")
    axes[1].set_xlabel("Écart-type résiduel (mm)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Figure régions variables : {out_path}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores",       required=True)
    parser.add_argument("--fs_dir",       required=True)
    parser.add_argument("--participants", required=True)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--id_format",    default=None)
    parser.add_argument("--n_bins",       type=int, default=5)
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
    demo["sex"] = demo[sex_col]

    merged = scores.merge(demo[["sub", "age", "sex"]], on="sub", how="inner")

    # APARC
    print("Lecture aparc.stats...")
    aparc = load_aparc_mean(args.fs_dir, merged["sub"].tolist())
    region_cols = [c for c in aparc.columns if c not in ["sub", "mean_thickness"]]

    full = merged.merge(aparc, on="sub", how="inner")
    print(f"  {len(full)} sujets")

    # Bins de mouvement
    full["motion_bin"] = assign_bins(full["motion"], n_bins=args.n_bins)

    # Stats par bin
    bin_stats = full.groupby("motion_bin", observed=True).agg(
        n=("motion", "count"),
        motion_median=("motion", "median"),
        motion_min=("motion", "min"),
        motion_max=("motion", "max"),
        thickness_median=("mean_thickness", "median"),
        thickness_std=("mean_thickness", "std"),
        thickness_min=("mean_thickness", "min"),
        thickness_max=("mean_thickness", "max"),
    )
    print("\nStats par bin de mouvement :")
    print(bin_stats.to_string())
    bin_stats.to_csv(out / "oasis1_motion_bins.csv")

    # Figures
    plot_thickness_per_bin(full, out / "oasis1_thickness_per_bin.png")
    plot_variance_per_bin(full, out / "oasis1_variance_per_bin.png")
    res = find_variable_regions(full, region_cols, out / "oasis1_variable_regions.png")
    res.to_csv(out / "oasis1_region_variance.csv", index=False)

    print(f"\nTerminé. Fichiers dans {out}")


if __name__ == "__main__":
    main()
