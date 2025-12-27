# Notebooks - Aneurysm Detection

Ce dossier contient les notebooks Jupyter pour l'analyse et le développement du projet de détection d'anévrismes.

## Vue d'ensemble

Les notebooks sont organisés par étapes du workflow de machine learning, de l'exploration des données jusqu'à l'inférence.

## Liste des notebooks

### [01_exploration_donnees.ipynb](01_exploration_donnees.ipynb)
**Objectif** : Exploration et nettoyage des données DICOM

**Contenu** :
- Chargement des datasets (train.csv, train_localizers.csv)
- Analyse des modalités (CTA, MRA, MRI T1post, MRI T2)
- Détection et filtrage des séries défectueuses
- Statistiques sur les dimensions des volumes
- Visualisation des distributions

**Sorties** : df_cleaned.csv avec les séries valides

---

### [02_dataset_creation.ipynb](02_dataset_creation.ipynb)
**Objectif** : Création des datasets de cubes 3D

**Contenu** :
- Extraction de cubes positifs (contenant anévrismes)
- Extraction de cubes négatifs (sans anévrismes)
- Création de dictionnaires par modalité
- Sauvegarde au format .npz

**Sorties** : CTA_dataset.npz, MRA_dataset.npz, etc.

---

### [03_entrainement_modele.ipynb](03_entrainement_modele.ipynb)
**Objectif** : Entraînement du modèle U-Net 3D

**Contenu** :
- Chargement des datasets
- Définition du modèle U-Net 3D
- Configuration de l'entraînement (optimizer, loss, metrics)
- Entraînement avec validation
- Sauvegarde des checkpoints

**Sorties** : best_model.pth, courbes de loss/accuracy

---

### [04_inference.ipynb](04_inference.ipynb)
**Objectif** : Pipeline d'inférence complet

**Contenu** :
- Chargement du modèle entraîné
- Préprocessing d'une nouvelle série DICOM
- Extraction de cubes chevauchants
- Prédiction et agrégation
- Calcul des métriques (AUC, accuracy)

**Sorties** : Prédictions pour le test set

---

### [05_data_augmentation.ipynb](05_data_augmentation.ipynb)
**Objectif** : Augmentation de données par déformations élastiques

**Contenu** :
- Application de déformations aléatoires 3D
- Génération de multiples versions augmentées
- Visualisation des transformations
- Sauvegarde des datasets augmentés

**Sorties** : cubes_augmented.npy

---

### [06_gestion_erreurs.ipynb](06_gestion_erreurs.ipynb)
**Objectif** : Analyse des erreurs et extraction de hard negatives

**Contenu** :
- Détection des faux positifs/faux négatifs
- Analyse des cas difficiles
- Extraction de hard negatives pour réentraînement
- Visualisation des erreurs du modèle

**Sorties** : hard_negatives.npy

---

## Ordre d'exécution recommandé

```
01 → 02 → 05 → 03 → 06 → 04
 ↓     ↓    ↓    ↓    ↓    ↓
EDA  Data  Aug  Train Error Infer
```

1. **01_exploration_donnees** - Comprendre les données
2. **02_dataset_creation** - Créer les cubes
3. **05_data_augmentation** - Augmenter le dataset
4. **03_entrainement_modele** - Entraîner le modèle
5. **06_gestion_erreurs** - Analyser les erreurs
6. **04_inference** - Faire des prédictions

## Configuration requise

### Imports du package `src/`

Tous les notebooks utilisent le package `src/` du projet :

```python
import sys
sys.path.append("../")

from src.data import dicom_to_numpy, ajouter_Modality
from src.preprocessing import preprocessing_volume, resample, crop
from src.visualization import show_middle_slices
from src.augmentation import data_augmentation
```

### Installation des dépendances

```bash
pip install -r ../requirements.txt
```

### Structure attendue

```
ANEURYSM DETECTION/
├── data/
│   ├── train.csv
│   ├── train_localizers.csv
│   └── series/
├── notebooks/          # ← Vous êtes ici
├── src/                # Package Python
└── results/
```

## Notes importantes

### Nettoyage des outputs

Les notebooks peuvent contenir de gros outputs (images, arrays). Pour nettoyer :

```bash
# Installer nbstripout
pip install nbstripout

# Configurer pour nettoyer automatiquement
nbstripout --install

# Nettoyer manuellement
jupyter nbconvert --clear-output --inplace *.ipynb
```

### Chemins des données

Les chemins sont configurables en début de chaque notebook. Adaptez selon votre environnement :

```python
# Local
DATA_DIR = "../data"

# Kaggle
DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection"
```

## Troubleshooting

**Import Error du package src/** :
```python
# Vérifiez le chemin
import sys
print(sys.path)
sys.path.append("../")  # ou chemin absolu
```

**Erreur PixelSpacing** :
- Certains fichiers DICOM n'ont pas de PixelSpacing
- Utiliser un try/except ou filtrer au préalable

**Mémoire insuffisante** :
- Réduire le batch_size
- Traiter les données par lots
- Utiliser un subset pour les tests

## Contact

Pour toute question sur les notebooks, consultez la documentation du package `src/` ou ouvrez une issue.
