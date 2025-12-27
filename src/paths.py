"""
Configuration des chemins de données pour le projet.

Ce fichier centralise tous les chemins pour faciliter le passage entre
Kaggle et local.
"""

import os

# Détection automatique de l'environnement
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    # Configuration Kaggle
    BASE_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection"
    SERIES_DIR = os.path.join(BASE_DIR, "series")
    TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
    TRAIN_LOCALIZERS_CSV = os.path.join(BASE_DIR, "train_localizers.csv")

    # Outputs Kaggle
    OUTPUT_DIR = "/kaggle/working"

else:
    # Configuration locale
    # Détermine le répertoire racine du projet
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Dossier data local
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    SERIES_DIR = os.path.join(DATA_DIR, "series")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TRAIN_LOCALIZERS_CSV = os.path.join(DATA_DIR, "train_localizers.csv")

    # Outputs locaux
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

    # Créer les dossiers s'ils n'existent pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chemins de sortie communs
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Créer tous les dossiers de sortie
for dir_path in [PROCESSED_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# Affichage de la configuration (pour debug)
def print_config():
    """Affiche la configuration actuelle des chemins."""
    print("=" * 60)
    print(f"Environment: {'KAGGLE' if IS_KAGGLE else 'LOCAL'}")
    print("=" * 60)
    print(f"SERIES_DIR: {SERIES_DIR}")
    print(f"TRAIN_CSV: {TRAIN_CSV}")
    print(f"TRAIN_LOCALIZERS_CSV: {TRAIN_LOCALIZERS_CSV}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
    print("=" * 60)


# Pour compatibilité avec les anciens notebooks
series_path = SERIES_DIR
