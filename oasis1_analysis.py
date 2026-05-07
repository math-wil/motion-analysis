import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# ── 1. Chemins ────────────────────────────────────────────────────────────────
MOTION_CSV = "/home/av62870@ens.ad.etsmtl.ca/Documents/oasis1_motion_scores.csv"
FREESURFER_DIR = "/project/hippocampus/common/datasets/OASIS1_BIDS/processed_freesurfer7.4.1"
OUTPUT_DIR = "/home/av62870@ens.ad.etsmtl.ca/Documents/oasis1-analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 2. Charger les scores de mouvement ────────────────────────────────────────
motion_df = pd.read_csv(MOTION_CSV, index_col=0)
motion_df = motion_df[["sub", "motion"]].copy()
print(f"Scores de mouvement : {len(motion_df)} sujets")

# ── 3. Lire l'épaisseur corticale moyenne depuis lh.aparc.stats ───────────────
def get_mean_thickness(stats_file):
    """Extrait l'épaisseur corticale moyenne globale depuis un fichier aparc.stats."""
    with open(stats_file, "r") as f:
        for line in f:
            if "MeanThickness" in line and "Measure" in line:
                # Ligne : # Measure Cortex, MeanThickness, Mean Thickness, 2.34941, mm
                parts = line.strip().split(",")
                return float(parts[-2].strip())
    return None

thickness_data = []
for sub in os.listdir(FREESURFER_DIR):
    if not sub.startswith("sub-"):
        continue
    lh_stats = os.path.join(FREESURFER_DIR, sub, "stats", "lh.aparc.stats")
    if not os.path.exists(lh_stats):
        continue
    thickness = get_mean_thickness(lh_stats)
    if thickness is not None:
        thickness_data.append({"sub": sub, "lh_mean_thickness": thickness})

thickness_df = pd.DataFrame(thickness_data)
print(f"Épaisseurs corticales : {len(thickness_df)} sujets")

# ── 4. Fusionner sur la colonne "sub" ─────────────────────────────────────────
merged_df = pd.merge(motion_df, thickness_df, on="sub", how="inner")
print(f"Sujets après fusion : {len(merged_df)}")

# ── 5. Corrélation de Spearman ────────────────────────────────────────────────
rho, pval = stats.spearmanr(merged_df["motion"], merged_df["lh_mean_thickness"])
print(f"\nCorrélation de Spearman : rho = {rho:.3f}, p = {pval:.2e}")

# ── 6. Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(merged_df["motion"], merged_df["lh_mean_thickness"],
           alpha=0.5, s=20, color="steelblue")

m, b = np.polyfit(merged_df["motion"], merged_df["lh_mean_thickness"], 1)
x_line = [merged_df["motion"].min(), merged_df["motion"].max()]
y_line = [m * x + b for x in x_line]
ax.plot(x_line, y_line, color="orange", linewidth=2)

ax.set_xlabel("Predicted Motion Score (mm)")
ax.set_ylabel("Mean Left Hemisphere Cortical Thickness (mm)")
ax.set_title(f"OASIS1 — Motion vs Cortical Thickness\nSpearman rho = {rho:.3f}, p = {pval:.2e}")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "motion_vs_thickness_oasis1.png"), dpi=150)
print(f"Figure sauvegardée dans {OUTPUT_DIR}")
plt.close()