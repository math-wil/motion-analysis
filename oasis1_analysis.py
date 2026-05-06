import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# ── 1. Chemins ────────────────────────────────────────────────────────────────
MOTION_CSV = "/home/av62870@ens.ad.etsmtl.ca/Documents/oasis1_motion_scores.csv"
FREESURFER_DIR = "/project/hippocampus/common/datasets/OASIS1_BIDS/processed_freesurfer7.4.1"
OUTPUT_DIR = "/home/av62870@ens.ad.etsmtl.ca/Documents/oasis1_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 2. Charger les scores de mouvement ────────────────────────────────────────
motion_df = pd.read_csv(MOTION_CSV)
# Garder seulement les colonnes utiles et renommer
motion_df = motion_df[["sub", "motion"]].copy()
print(f"Scores de mouvement chargés : {len(motion_df)} sujets")

# ── 3. Lire l'épaisseur corticale depuis lh.aparc.stats ──────────────────────
def read_aparc_stats(stats_file):
    """Lit un fichier lh.aparc.stats ou rh.aparc.stats et retourne
    l'épaisseur corticale moyenne globale de l'hémisphère."""
    mean_thickness = None
    with open(stats_file, "r") as f:
        for line in f:
            # La ligne qui contient l'épaisseur moyenne globale
            if "MeanThickness" in line and "Measure" in line:
                # Exemple : # Measure Cortex, MeanThickness, Mean Thickness, 2.34941, mm
                parts = line.strip().split(",")
                mean_thickness = float(parts[-2].strip())
                break
    return mean_thickness

thickness_data = []
for sub_folder in os.listdir(FREESURFER_DIR):
    if not sub_folder.startswith("sub-"):
        continue
    
    # Chemin vers lh.aparc.stats
    lh_stats = os.path.join(FREESURFER_DIR, sub_folder, "stats", "lh.aparc.stats")
    
    if not os.path.exists(lh_stats):
        continue
    
    thickness = read_aparc_stats(lh_stats)
    if thickness is not None:
        # Extraire l'identifiant BIDS depuis le nom du dossier FreeSurfer
        # Le dossier s'appelle sub-OASISXXXXsub-OASISXXXX → on prend juste sub-XXXX
        sub_id = sub_folder.split("sub-OASIS")[-1]
        sub_id = f"sub-{sub_id.split('sub-')[0]}" if "sub-" in sub_id else sub_folder
        thickness_data.append({"sub_fs": sub_folder, "lh_mean_thickness": thickness})

thickness_df = pd.DataFrame(thickness_data)
print(f"Épaisseurs corticales chargées : {len(thickness_df)} sujets")

# ── 4. Fusionner les deux DataFrames ─────────────────────────────────────────
# On fait la jointure sur le numéro de sujet
# motion_df a "sub-0001", FreeSurfer a "sub-0001sub-OASIS10001" ou similaire
# On extrait le numéro commun

def extract_sub_number(sub_str):
    """Extrait le numéro de sujet depuis un identifiant BIDS."""
    import re
    match = re.search(r"sub-0*(\d+)", sub_str)
    if match:
        return int(match.group(1))
    return None

motion_df["sub_num"] = motion_df["sub"].apply(extract_sub_number)
thickness_df["sub_num"] = thickness_df["sub_fs"].apply(extract_sub_number)

merged_df = pd.merge(motion_df, thickness_df, on="sub_num", how="inner")
print(f"Sujets après fusion : {len(merged_df)}")
print(merged_df.head())

# ── 5. Corrélation de Spearman ────────────────────────────────────────────────
rho, pval = stats.spearmanr(merged_df["motion"], merged_df["lh_mean_thickness"])
print(f"\nCorrélation de Spearman : rho = {rho:.3f}, p = {pval:.2e}")

# ── 6. Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(merged_df["motion"], merged_df["lh_mean_thickness"], 
           alpha=0.5, s=20, color="steelblue")

# Ligne de régression
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