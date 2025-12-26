# Détection d'Anévrismes Cérébraux par Deep Learning 🧠

Projet de détection automatique d'anévrismes cérébraux à partir d'images médicales 3D (CTA, MRA, MRI) utilisant des réseaux de neurones convolutifs 3D.

---

## 📊 À propos de ce Repository

### Version de Démonstration

Ce repository contient une **version de démonstration locale** d'un projet plus large réalisé sur Kaggle.

| Aspect | Version Kaggle (Originale) | Version Démonstration (ce repo) |
|--------|---------------------------|----------------------------------|
| **Données** | 4000+ séries DICOM | 20 séries échantillons |
| **Environnement** | Kaggle GPU | Local (CPU/GPU) |
| **Objectif** | Entraînement complet | Démonstration & Portfolio |
| **Architecture** | Identique ✓ | Identique ✓ |
| **Pipeline** | Identique ✓ | Identique ✓ |

**But de cette version** : Permettre aux recruteurs et collaborateurs de :
- Visualiser la structure du projet et du code
- Comprendre le pipeline de preprocessing
- Voir des exemples concrets de datasets créés
- Tester le code localement sans télécharger 4000 séries

---

## 🏗️ Architecture du Projet

Le projet suit une **architecture modulaire brick-based** pour une réutilisabilité maximale :

```
src/
├── bricks/              # Composants réutilisables
│   ├── preprocessing.py # Pipeline de preprocessing 3D
│   └── dataset.py       # Construction de datasets
├── data/                # Utilitaires de chargement DICOM
├── preprocessing/       # Transformations (resample, crop, normalize)
├── visualization/       # Visualisation de volumes 3D
└── config.py            # Configuration centralisée
```

### Composants Principaux

**`Preprocessor`** : Pipeline de preprocessing pour volumes DICOM 3D
- Chargement DICOM
- Resampling à espacement cible
- Cropping automatique
- Normalisation

**`DatasetBuilder`** : Construction de datasets d'entraînement
- Extraction de cubes 3D (positifs/négatifs)
- One-hot encoding des positions anatomiques
- Sauvegarde au format `.npz`

---

## 📁 Structure des Données

### Modalités Supportées

Le projet gère 4 modalités d'imagerie médicale :

| Modalité | Description | Séries (démo) | Séries (Kaggle) |
|----------|-------------|---------------|-----------------|
| **CTA** | Angiographie par tomodensitométrie | 9 | ~1220 |
| **MRA** | Angiographie par résonance magnétique | 3 | ~661 |
| **MRI T2** | IRM pondérée T2 | 7 | ~280 |
| **MRI T1post** | IRM T1 avec contraste | 1 | ~93 |

### Datasets Créés

Chaque modalité génère un fichier `.npz` contenant :

```python
{
    'cubes': np.ndarray,      # (N, 48, 48, 48) - Cubes 3D normalisés
    'labels': np.ndarray,     # (N,) - 0=négatif, 1=positif
    'positions': np.ndarray,  # (N, 13) - One-hot encoding position anatomique
    'patient_ids': list       # (N,) - SeriesInstanceUID
}
```

**Fichiers générés** :
- `results/processed/cta_dataset.npz`
- `results/processed/mra_dataset.npz`
- `results/processed/mri_t2_dataset.npz`
- `results/processed/mri_t1post_dataset.npz`

---

## 📓 Notebooks

| Notebook | Description | Usage |
|----------|-------------|-------|
| `01_exploration_donnees.ipynb` | Analyse exploratoire des données DICOM | Comprendre les données |
| `02_dataset_creation.ipynb` | Création des 4 datasets par modalité | **Exécuter en premier** |
| `03_entrainement_modele.ipynb` | Entraînement du modèle 3D CNN | Après création datasets |
| `04_inference.ipynb` | Inférence sur nouveaux patients | Tests & démonstration |

> **Note**: Les notebooks utilisent un échantillon de 20 séries DICOM pour démonstration.
> Le projet original sur Kaggle a traité 4000+ séries avec la même architecture.

---

## 🚀 Utilisation

### Prérequis

```bash
pip install -r requirements.txt
```

### Configuration Automatique

Le projet détecte automatiquement l'environnement (Kaggle vs Local) :

```python
from src import SERIES_DIR, TRAIN_CSV, PROCESSED_DIR, print_config

print_config()  # Affiche les chemins configurés
```

### Création des Datasets

1. **Ouvrir** `notebooks/02_dataset_creation.ipynb`
2. **Exécuter** les cellules jusqu'à la section 5
3. **Décommenter** le code de création dans les sections 5, 6, 7
4. **Lancer** la création des 4 datasets

Les datasets seront sauvegardés dans `results/processed/`

### Structure des Résultats

```
results/
├── processed/           # Datasets créés (.npz)
├── models/             # Modèles entraînés (.pth)
└── checkpoints/        # Checkpoints d'entraînement
```

---

## 🎯 Résultats de la Démonstration

### Datasets de Démonstration Créés

| Modalité | Séries | Cubes positifs | Cubes négatifs | Balance |
|----------|--------|----------------|----------------|---------|
| CTA | 9 | ~6 | ~X | XX% |
| MRA | 3 | ~2 | ~X | XX% |
| MRI T2 | 7 | ~1 | ~X | XX% |
| MRI T1post | 1 | 0 | ~X | XX% |

*Note : Les valeurs exactes dépendent du nombre de cubes extraits par volume*

### Utilisation Recommandée

- **CTA** : Dataset principal pour démonstration (meilleure représentativité)
- **MRA** : Validation cross-modalité
- **MRI T2** : Tests de généralisation
- **MRI T1post** : Exemple de modalité rare

---

## 🔧 Technologies

- **Python 3.13+**
- **PyTorch** : Deep Learning
- **PyDICOM** : Lecture fichiers DICOM
- **NumPy & SciPy** : Traitement volumétrique
- **Pandas** : Manipulation de données
- **Matplotlib** : Visualisation

---

## 📝 Notes Importantes

### Limitations de la Version Démonstration

- ⚠️ **Données limitées** : 20 séries vs 4000 sur Kaggle
- ⚠️ **Performances** : Résultats non représentatifs (échantillon trop petit)
- ⚠️ **Usage** : Démonstration de l'architecture uniquement

### Points Forts du Projet

- ✅ **Architecture modulaire** : Code réutilisable et maintenable
- ✅ **Multi-modalités** : Support de 4 types d'imagerie médicale
- ✅ **Pipeline complet** : De DICOM brut à modèle entraîné
- ✅ **Gestion d'erreurs** : Robuste aux données manquantes
- ✅ **Documentation** : Code commenté et notebooks explicatifs

---

## 📚 Références

- **Dataset source** : [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)
- **Architecture** : Inspired by 3D ResNet for medical imaging

---

## 👤 Auteur

**Emma B.**

*Data Scientist spécialisée en Deep Learning pour l'imagerie médicale*

---

## 📄 License

Ce projet est à usage éducatif et de démonstration pour portfolio professionnel.

Les données DICOM ne sont pas incluses dans ce repository pour des raisons de confidentialité et de taille.
