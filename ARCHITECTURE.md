# Architecture du Projet - Détection d'Anévrismes

## 📐 Vue d'Ensemble

Ce projet suit une architecture modulaire avec des composants de pipeline réutilisables (**bricks**), permettant une orchestration flexible et un déploiement facile en production.

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION                             │
│              (notebooks/00_orchestration.ipynb)              │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐       ┌──────────┐      ┌──────────┐
   │   EDA   │       │ Dataset  │      │ Training │
   │         │───────│ Builder  │──────│          │
   └─────────┘       └──────────┘      └──────────┘
                           │                  │
                           ▼                  ▼
                    ┌─────────────┐    ┌──────────┐
                    │ Augmentor   │    │ Predictor│
                    └─────────────┘    └──────────┘
                           │
                    ┌──────┴───────┐
                    ▼              ▼
              ┌──────────┐  ┌──────────┐
              │Preprocess│  │  Models  │
              └──────────┘  └──────────┘
```

## 📂 Structure du Projet

```
ANEURYSM DETECTION/
│
├── data/                           # Données (gitignored)
│   ├── train.csv
│   ├── train_localizers.csv
│   └── series/                     # Séries DICOM
│       └── <SeriesInstanceUID>/
│
├── src/                            # Package Python
│   ├── __init__.py                # Exports principaux
│   ├── config.py                  # Constantes globales
│   ├── paths.py                   # Gestion chemins Kaggle/local
│   │
│   ├── bricks/                    # ⭐ Composants de pipeline
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # Preprocessor
│   │   ├── dataset.py             # DatasetBuilder
│   │   ├── augmentation.py        # Augmentor
│   │   ├── eda.py                 # EDA
│   │   ├── training.py            # Trainer
│   │   ├── inference.py           # Predictor
│   │   └── README.md
│   │
│   ├── models/                    # Architectures de modèles
│   │   ├── __init__.py
│   │   └── unet3d.py             # UNet3DClassifier
│   │
│   ├── data/                      # Utilitaires DICOM
│   │   ├── dicom_loader.py
│   │   └── metadata.py
│   │
│   ├── preprocessing/             # Fonctions bas niveau
│   │   ├── transforms.py
│   │   ├── coordinates.py
│   │   └── pipeline.py
│   │
│   ├── augmentation/              # Déformations élastiques
│   │   └── elastic.py
│   │
│   ├── visualization/             # Visualisation
│   │   └── viewers.py
│   │
│   └── utils/                     # Utilitaires
│       └── __init__.py
│
├── notebooks/                     # Notebooks Jupyter
│   ├── 00_orchestration.ipynb    # ⭐ Pipeline principal
│   ├── 01_exploration_donnees.ipynb
│   ├── 02_dataset_creation.ipynb
│   ├── 03_entrainement_modele.ipynb
│   ├── 04_inference.ipynb
│   ├── 05_data_augmentation.ipynb
│   ├── 06_gestion_erreurs.ipynb
│   └── README.md
│
├── results/                       # Sorties (gitignored)
│   ├── processed/                # Datasets créés
│   ├── models/                   # Modèles entraînés
│   └── checkpoints/              # Checkpoints
│
├── tests/                         # Tests unitaires (à créer)
│
├── .gitignore                     # Protection données
├── requirements.txt               # Dépendances
├── README.md                      # Documentation projet
├── ARCHITECTURE.md               # Ce fichier
├── MIGRATION_GUIDE.md            # Guide migration
└── PATH_MIGRATION_COMPLETE.md    # Guide chemins

```

## 🧱 Composants Bricks

### 1. Preprocessor
**Responsabilité** : Preprocessing des volumes DICOM 3D
**Input** : Chemin vers série DICOM
**Output** : Volume normalisé (+ coordonnées transformées optionnel)
**Méthodes clés** :
- `process_volume(patient_path)` → volume
- `process_volume_with_coords(patient_path, coords)` → volume, coords

### 2. DatasetBuilder
**Responsabilité** : Construction de datasets d'entraînement
**Input** : DataFrames + Preprocessor
**Output** : Dataset au format dict/npz
**Méthodes clés** :
- `build_dataset(df_train, df_localizers, modality)` → dataset
- `save(dataset, path)` / `load(path)` → dataset

### 3. Augmentor
**Responsabilité** : Augmentation de données par déformations élastiques
**Input** : Dataset
**Output** : Dataset augmenté
**Méthodes clés** :
- `augment_dataset(dataset)` → augmented_dataset
- `save(dataset, path)` / `load(path)` → dataset

### 4. EDA
**Responsabilité** : Analyse exploratoire des données
**Input** : DataFrames + chemin séries
**Output** : Statistiques, visualisations, rapports
**Méthodes clés** :
- `analyze_modalities()` → stats
- `detect_defective_series()` → liste UIDs
- `generate_report()` → rapport complet

### 5. Trainer
**Responsabilité** : Entraînement de modèles PyTorch
**Input** : Modèle + DataLoaders
**Output** : Modèle entraîné + métriques
**Méthodes clés** :
- `fit(train_loader, val_loader, epochs)` → historique
- `save_checkpoint(path)` / `load_checkpoint(path)`
- `plot_history()` → visualisation

### 6. Predictor
**Responsabilité** : Inférence sur nouveaux volumes
**Input** : Modèle + volume
**Output** : Prédiction agrégée
**Méthodes clés** :
- `predict_volume(patient_path)` → prediction dict
- `predict_batch(patient_paths)` → liste predictions

## 🔄 Flux de Données

### Pipeline Complet

```
1. EDA
   │
   └──> Analyse des données
        - Distribution modalités
        - Séries défectueuses
        - Statistiques anévrismes
   │
   ▼
2. Dataset Creation
   │
   ├──> Preprocessor
   │    - Load DICOM
   │    - Resample
   │    - Crop
   │    - Normalize
   │
   └──> DatasetBuilder
        - Extract positives
        - Extract negatives
        - Create labels/positions
        - Save .npz
   │
   ▼
3. Augmentation
   │
   └──> Augmentor
        - Elastic deformations
        - N versions per cube
        - Save augmented .npz
   │
   ▼
4. Training
   │
   ├──> PyTorch Dataset
   │    - Load .npz
   │    - To tensors
   │
   ├──> DataLoader
   │    - Batching
   │    - Shuffling
   │
   └──> Trainer
        - Train epochs
        - Validate
        - Save checkpoints
        - Track metrics
   │
   ▼
5. Inference
   │
   └──> Predictor
        - Load model
        - Process volume
        - Aggregate predictions
        - Return results
```

## 🎯 Avantages de cette Architecture

### ✅ Modularité
- Chaque composant est indépendant
- Facile à tester unitairement
- Réutilisable dans d'autres projets

### ✅ Flexibilité
- On peut remplacer n'importe quel composant
- Facile d'ajouter de nouvelles features
- Support de multiples modalités

### ✅ Maintenabilité
- Code organisé et documenté
- Responsabilités claires
- Facile à débugger

### ✅ Production-Ready
- Pipeline complet dans un notebook
- Facile à déployer
- Versioning clair

### ✅ Collaboration
- Structure standard
- Documentation complète
- Facile pour nouveaux contributeurs

## 📊 Comparaison Avant/Après

### Avant (Notebooks monolithiques)
```python
# Notebook 1: 500 lignes
def dicom_to_numpy(...): ...
def resample(...): ...
def crop(...): ...
# ... 20 fonctions ...
# Code d'analyse
# Code de preprocessing
# Code d'entraînement
```

**Problèmes** :
- ❌ Duplication de code entre notebooks
- ❌ Difficile à maintenir
- ❌ Pas testable
- ❌ Difficile à déployer

### Après (Architecture modulaire)
```python
# Notebook d'orchestration: 50 lignes
from src.bricks import Preprocessor, DatasetBuilder, Trainer

preprocessor = Preprocessor()
builder = DatasetBuilder(preprocessor)
trainer = Trainer(model, criterion, optimizer)

dataset = builder.build_dataset(df_train, df_loc, modality='CTA')
trainer.fit(train_loader, val_loader, epochs=10)
```

**Avantages** :
- ✅ Code réutilisable
- ✅ Facile à maintenir
- ✅ Testable
- ✅ Production-ready

## 🚀 Utilisation

### Quick Start

```python
# Import tout depuis src
from src import *
from src.bricks import *
from src.models import *

# 1. EDA
eda = EDA(df_train, df_localizers, SERIES_DIR)
eda.generate_report()

# 2. Dataset
preprocessor = Preprocessor()
builder = DatasetBuilder(preprocessor, series_dir=SERIES_DIR)
dataset = builder.build_dataset(df_train, df_localizers, modality='CTA')

# 3. Augmentation
augmentor = Augmentor(n_augmentations=12)
dataset_aug = augmentor.augment_dataset(dataset)

# 4. Training
model = UNet3DClassifier()
trainer = Trainer(model, criterion, optimizer)
trainer.fit(train_loader, val_loader, epochs=10)

# 5. Inference
predictor = Predictor(model, preprocessor)
prediction = predictor.predict_volume(patient_path)
```

### Notebook d'Orchestration

Le notebook [`00_orchestration.ipynb`](notebooks/00_orchestration.ipynb) contient un pipeline complet prêt à l'emploi.

## 🧪 Tests (À Implémenter)

Structure recommandée pour les tests :

```
tests/
├── test_preprocessing.py
├── test_dataset.py
├── test_augmentation.py
├── test_eda.py
├── test_training.py
└── test_inference.py
```

## 📖 Documentation

- **README.md** : Vue d'ensemble du projet
- **ARCHITECTURE.md** : Ce fichier - architecture détaillée
- **MIGRATION_GUIDE.md** : Migration de l'ancien code
- **src/bricks/README.md** : Documentation des bricks
- **notebooks/README.md** : Guide des notebooks

## 🔮 Évolutions Futures

### Court Terme
- [ ] Tests unitaires pour chaque brick
- [ ] CI/CD avec GitHub Actions
- [ ] Notebook d'expérimentation

### Moyen Terme
- [ ] Support de nouvelles modalités
- [ ] Hyperparameter tuning
- [ ] MLflow pour tracking

### Long Terme
- [ ] API REST pour inférence
- [ ] Interface web
- [ ] Déploiement cloud

## 📚 Ressources

- [Documentation PyTorch](https://pytorch.org/docs/)
- [DICOM Standard](https://www.dicomstandard.org/)
- [UNet Paper](https://arxiv.org/abs/1505.04597)

---

**Version** : 0.2.0
**Dernière mise à jour** : 2025-12-26
