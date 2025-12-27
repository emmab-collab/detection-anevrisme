# Bricks - Composants de Pipeline

Ce module contient les composants de pipeline réutilisables pour le projet de détection d'anévrismes.

## Composants Disponibles

### 1. `Preprocessor` - Preprocessing des volumes

Gère tout le preprocessing des volumes DICOM 3D :
- Chargement DICOM
- Resampling à un espacement cible
- Cropping pour retirer le fond noir
- Normalisation entre 0 et 1
- Transformation de coordonnées

**Exemple** :
```python
from src.bricks import Preprocessor

preprocessor = Preprocessor(target_spacing=(0.4, 0.4, 0.4))

# Process volume seul
volume = preprocessor.process_volume(patient_path)

# Process volume avec coordonnées
volume, coords = preprocessor.process_volume_with_coords(patient_path, aneurysm_coords)
```

### 2. `DatasetBuilder` - Construction de datasets

Construit des datasets d'entraînement à partir de volumes DICOM :
- Extraction de cubes positifs (avec anévrisme)
- Extraction de cubes négatifs (sans anévrisme)
- Création de vecteurs one-hot pour les positions
- Sauvegarde/chargement au format .npz

**Exemple** :
```python
from src.bricks import Preprocessor, DatasetBuilder

preprocessor = Preprocessor()
builder = DatasetBuilder(preprocessor, cube_size=48, series_dir=SERIES_DIR)

# Construire dataset pour CTA
dataset = builder.build_dataset(df_train, df_localizers, modality='CTA')

# Sauvegarder
builder.save(dataset, 'results/processed/cta_dataset.npz')

# Charger
dataset = DatasetBuilder.load('results/processed/cta_dataset.npz')
```

### 3. `Augmentor` - Augmentation de données

Applique des déformations élastiques pour augmenter le dataset :
- Déformations 3D aléatoires
- Augmentation configurable (nombre de versions)
- Support pour augmenter seulement les positifs

**Exemple** :
```python
from src.bricks import Augmentor

augmentor = Augmentor(n_augmentations=12)

# Augmenter un dataset complet
dataset_augmented = augmentor.augment_dataset(dataset, augment_negatives=False)

# Sauvegarder
augmentor.save(dataset_augmented, 'results/processed/dataset_augmented.npz')
```

### 4. `EDA` - Analyse Exploratoire

Analyse les données DICOM :
- Distribution des modalités
- Distribution des anévrismes
- Détection de séries défectueuses
- Statistiques sur les slices
- Génération de rapports

**Exemple** :
```python
from src.bricks import EDA

eda = EDA(df_train, df_localizers, SERIES_DIR)

# Analyses
eda.analyze_modalities()
eda.analyze_aneurysm_distribution()
defective = eda.detect_defective_series()

# Rapport complet
eda.generate_report()

# Visualisations
eda.plot_aneurysm_distribution()
```

### 5. `Trainer` - Entraînement de modèles

Gère l'entraînement de modèles PyTorch :
- Entraînement par epochs
- Validation
- Sauvegarde de checkpoints
- Suivi de métriques
- Visualisation de l'historique

**Exemple** :
```python
from src.bricks import Trainer
from src.models import UNet3DClassifier

model = UNet3DClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = Trainer(model, criterion, optimizer, checkpoint_dir='results/checkpoints')

# Entraîner
trainer.fit(train_loader, val_loader, epochs=10)

# Visualiser
trainer.plot_history()

# Sauvegarder
trainer.save_checkpoint('results/models/best_model.pth')
```

### 6. `Predictor` - Inférence

Effectue des prédictions sur de nouveaux volumes :
- Chargement de modèles entraînés
- Extraction de cubes chevauchants
- Prédiction par cube
- Agrégation de prédictions (mean, max, percentile)

**Exemple** :
```python
from src.bricks import Predictor, Preprocessor
from src.models import UNet3DClassifier

model = UNet3DClassifier()
preprocessor = Preprocessor()

predictor = Predictor(model, preprocessor)
predictor.load_model('results/models/best_model.pth')

# Prédire sur un volume
prediction = predictor.predict_volume(
    patient_path,
    threshold=0.5,
    top_k=5,
    aggregation='mean'
)

print(prediction)
# {'volume_prob': 0.87, 'volume_label': 1, 'n_cubes': 150, 'top_k_probs': [0.95, 0.92, ...]}
```

## Usage - Pipeline Complet

Voir le notebook [`00_orchestration.ipynb`](../../notebooks/00_orchestration.ipynb) pour un exemple complet d'utilisation de tous les composants ensemble.

## Structure des Fichiers

```
src/bricks/
├── __init__.py          # Exports des classes
├── preprocessing.py     # Classe Preprocessor
├── dataset.py           # Classe DatasetBuilder
├── augmentation.py      # Classe Augmentor
├── eda.py               # Classe EDA
├── training.py          # Classe Trainer
├── inference.py         # Classe Predictor
└── README.md           # Ce fichier
```

## Extension

Pour ajouter un nouveau composant :

1. Créer un nouveau fichier dans `src/bricks/`
2. Implémenter votre classe
3. L'ajouter à `src/bricks/__init__.py`
4. L'ajouter à `src/__init__.py`
5. Documenter dans ce README

## Principes de Design

Chaque classe suit ces principes :

- **Single Responsibility** : Une classe = une responsabilité claire
- **Configurabilité** : Paramètres dans `__init__`
- **Chaînabilité** : Les sorties peuvent être passées aux entrées d'autres classes
- **Testabilité** : Code isolé et facile à tester
- **Documentation** : Docstrings NumPy complètes

## Documentation Complète

Chaque classe contient des docstrings détaillées avec :
- Description
- Paramètres
- Returns
- Exemples d'utilisation

Utilisez `help(ClassName)` ou consultez les docstrings directement dans le code.
