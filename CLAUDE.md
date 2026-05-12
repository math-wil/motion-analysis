# motion-analysis

## TÂCHE EN COURS (session laptop 12 mai)
Ajouter les fichiers manquants au repo et committer :
1. Copier `pipeline_agitation.py` dans `pipelines/`
2. Copier `pipeline_glm_aparc.py` dans `pipelines/`
3. Remplacer `datasets/oasis1/scripts/oasis1_glm_mean.py` par la version corrigée (BASE_DIR mis à jour)
4. Vérifier que `datasets/ds001907/` existe avec les sous-dossiers scripts/, results/, figures/, metadata/
5. Commit : "add pipelines + ds001907 structure" et push

---

## Projet
Mathilde Wilfart, M.Sc. recherche, labo Neuro-iX, ÉTS Montréal.
Directeur : Sylvain Bouix.
Sujet : correction des artefacts de mouvement en IRM cérébrale structurelle T1w.
But : qualifier le mouvement (Agitation) + corriger (JDAC) + valider sur épaisseur corticale APARC.

## Environnements

**Laptop Windows (maintenant) — code uniquement, pas de données**
```
Repo : C:\Users\Mathilde\Documents\GitHub\motion-analysis
```

**PC du labo (demain) — exécution des analyses**
```
User      : av62870@ens.ad.etsmtl.ca@ETS053232L
Conda env : cortical-motion
Repo      : ~/Documents/motion-analysis

OASIS1 BIDS raw    : /project/hippocampus/common/datasets/OASIS1_BIDS/raw_data_bids
OASIS1 preprocMNI  : /project/hippocampus/common/datasets/OASIS1_BIDS/preprocessed_mni
OASIS1 FreeSurfer  : /project/hippocampus/common/datasets/OASIS1_BIDS/processed_freesurfer7.4.1
ds001907           : pas encore téléchargé
```

## Structure du repo
```
motion-analysis/
├── pipelines/
│   ├── pipeline_agitation.py    wrapper agitation CLI
│   ├── pipeline_glm_aparc.py    GLM APARC générique multi-datasets
│   └── pipeline_jdac.py         à créer (source sur PC labo)
├── datasets/
│   ├── oasis1/
│   │   ├── scripts/
│   │   │   ├── oasis1_agitation.py
│   │   │   ├── oasis1_glm_mean.py   (BASE_DIR = ~/Documents/motion-analysis/datasets/oasis1)
│   │   │   └── oasis1_glm_aparc.py
│   │   ├── results_raw/
│   │   │   ├── oasis1_scores_raw.csv
│   │   │   ├── oasis1_glm_aparc.csv
│   │   │   └── oasis1_glm_mean.txt
│   │   ├── results_mni/
│   │   │   ├── oasis1_scores_mni.csv       (en local sur PC labo, pas encore pushé)
│   │   │   ├── oasis1_jdac_subjects.csv
│   │   │   └── oasis1_glm_aparc_mni.csv    (à générer)
│   │   ├── figures/
│   │   └── metadata/
│   └── ds001907/
│       ├── scripts/
│       ├── results/
│       ├── figures/
│       └── metadata/
└── CLAUDE.md
```

## Commandes types (PC du labo)
```bash
# Agitation sur preprocessed_mni
python pipelines/pipeline_agitation.py \
    --bids_root /project/hippocampus/common/datasets/OASIS1_BIDS/preprocessed_mni \
    --out       datasets/oasis1/results_mni/oasis1_scores_mni.csv \
    --pattern   "{sub}/{sub}_t1w_mni.nii.gz"

# GLM APARC OASIS1
python pipelines/pipeline_glm_aparc.py \
    --dataset      oasis1 \
    --scores       datasets/oasis1/results_mni/oasis1_scores_mni.csv \
    --fs_dir       /project/hippocampus/common/datasets/OASIS1_BIDS/processed_freesurfer7.4.1 \
    --participants /project/hippocampus/common/datasets/OASIS1_BIDS/raw_data_bids/participants.tsv \
    --bids_root    /project/hippocampus/common/datasets/OASIS1_BIDS/preprocessed_mni \
    --out_dir      datasets/oasis1/results_mni \
    --id_format    oasis1

# Télécharger ds001907 (sur Hippocampus)
cd /project/hippocampus/common/datasets
datalad install https://github.com/OpenNeuroDatasets/ds001907
```

## Prochaines étapes (semaine 11-15 mai)
1. Pusher oasis1_scores_mni.csv depuis le labo
2. Relancer GLM APARC sur scores MNI
3. Télécharger ds001907 + Agitation + GLM APARC
4. Créer pipeline_jdac.py (fichier source jdac_infer.py sur le labo)
5. Comparer ΔAIC OASIS1 vs ds001907
