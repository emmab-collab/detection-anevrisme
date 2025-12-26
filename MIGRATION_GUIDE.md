# Guide de Migration - Utilisation du Package `src/`

Ce guide vous explique comment utiliser le nouveau package `src/` dans vos notebooks.

## ✨ Ce qui a changé

### Avant
```python
# Copier-coller toutes les fonctions dans chaque notebook
def dicom_to_numpy(patient_path):
    dicom_files = sorted(glob.glob(patient_path+'/*.dcm'))
    # ... 20 lignes de code ...

def resample(volume, spacing):
    # ... code ...

def show_middle_slices(volume):
    # ... code ...
```

### Maintenant
```python
import sys
sys.path.append("../")

from src.data import dicom_to_numpy
from src.preprocessing import resample, crop, normalization
from src.visualization import show_middle_slices
```

## 📦 Structure du package `src/`

```
src/
├── __init__.py           # Exports principaux
├── config.py             # Constantes (TARGET_SPACING, etc.)
├── utils.py              # Utilitaires (get_pixelspacing)
│
├── data/                 # Chargement DICOM
│   ├── dicom_loader.py   # dicom_to_numpy, get_instance_number
│   └── metadata.py       # get_patient_ID, get_position
│
├── preprocessing/        # Transformations
│   ├── transforms.py     # resample, crop, normalization
│   ├── coordinates.py    # get_center, resample_coordonnees
│   └── pipeline.py       # preprocessing_volume
│
├── augmentation/         # Data augmentation
│   └── elastic.py        # random_deformation, data_augmentation
│
└── visualization/        # Affichage
    └── viewers.py        # show_middle_slices, show_slice_with_point
```

## 🚀 Guide de migration par fonction

### Chargement de données

```python
# Avant
def dicom_to_numpy(patient_path):
    # ... code copié ...

# Maintenant
from src.data import dicom_to_numpy
```

### Preprocessing

```python
# Avant
def resample(volume, spacing, target_spacing=(0.4, 0.4, 0.4)):
    # ... code copié ...

# Maintenant
from src.preprocessing import resample
from src.config import TARGET_SPACING

resample(volume, spacing, target_spacing=TARGET_SPACING)
```

### Visualisation

```python
# Avant
def show_middle_slices(volume):
    # ... code copié ...

# Maintenant
from src.visualization import show_middle_slices
```

### Augmentation

```python
# Avant
def random_deformation(volume, grid_size=3):
    # ... code copié ...

# Maintenant
from src.augmentation import random_deformation
```

## 📝 Template de notebook

Voici le template à utiliser en haut de chaque notebook :

```python
# ============================================
# IMPORTS STANDARDS
# ============================================
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# ============================================
# IMPORTS DU PACKAGE SRC
# ============================================
import sys
sys.path.append("../")  # Ou chemin absolu vers le projet

# Data loading
from src.data import (
    dicom_to_numpy,
    get_instance_number,
    get_patient_ID,
    ajouter_Modality
)

# Preprocessing
from src.preprocessing import (
    resample,
    crop,
    normalization,
    preprocessing_volume,
    preprocessing_volume_and_coords
)

# Visualization
from src.visualization import (
    show_middle_slices,
    show_slice_with_point
)

# Augmentation
from src.augmentation import (
    random_deformation,
    data_augmentation,
    dataset_augmented
)

# Configuration
from src.config import (
    TARGET_SPACING,
    ANEURYSM_POSITIONS
)

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = "../data"
series_path = os.path.join(DATA_DIR, "series")
```

## 🔧 Modifier les constantes

### Avant
```python
# Hardcodé dans le code
resample(volume, spacing, target_spacing=(0.4, 0.4, 0.4))
```

### Maintenant
```python
# Dans src/config.py
TARGET_SPACING = (0.4, 0.4, 0.4)

# Dans votre notebook
from src.config import TARGET_SPACING
resample(volume, spacing, target_spacing=TARGET_SPACING)

# Pour changer: éditez src/config.py
```

## 📚 Imports disponibles

### Depuis `src` directement
```python
from src import (
    # Tout est accessible depuis le __init__.py principal
    dicom_to_numpy,
    resample,
    crop,
    show_middle_slices,
    data_augmentation,
    TARGET_SPACING
)
```

### Imports spécifiques par module
```python
# Data
from src.data import dicom_to_numpy, get_patient_ID, ajouter_Modality

# Preprocessing
from src.preprocessing import resample, crop, normalization

# Visualization
from src.visualization import show_middle_slices, show_slice_with_point

# Augmentation
from src.augmentation import random_deformation, data_augmentation
```

## ⚙️ Configuration Kaggle vs Local

### Local
```python
import sys
sys.path.append("../")  # Depuis notebooks/

from src.preprocessing import preprocessing_volume
```

### Kaggle
Pour Kaggle, vous devrez :

1. **Option A** : Uploader `src/` comme dataset Kaggle
```python
import sys
sys.path.append("/kaggle/input/aneurysm-detection-src")

from src.preprocessing import preprocessing_volume
```

2. **Option B** : Copier les fichiers nécessaires
```python
# Copier uniquement ce dont vous avez besoin
!cp /kaggle/input/src-package/preprocessing.py .
from preprocessing import resample
```

## 🐛 Troubleshooting

### ImportError: No module named 'src'

**Solution** :
```python
import sys
import os

# Chemin absolu vers le projet
project_path = os.path.abspath("../")
sys.path.insert(0, project_path)

from src import dicom_to_numpy
```

### Les modifications du package ne sont pas prises en compte

**Solution** : Redémarrez le kernel Jupyter
```python
# Ou utilisez autoreload
%load_ext autoreload
%autoreload 2
```

### Erreur de chemin relatif

**Solution** : Utilisez des chemins absolus
```python
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath("../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
```

## 📖 Documentation

- Package `src/` : Voir docstrings dans chaque module
- Notebooks : Voir [notebooks/README.md](notebooks/README.md)
- Configuration : Voir [src/config.py](src/config.py)

## ✅ Checklist de migration

- [ ] `sys.path.append("../")` au début du notebook
- [ ] Remplacer les définitions de fonctions par des imports
- [ ] Utiliser `from src.config import TARGET_SPACING` au lieu de valeurs hardcodées
- [ ] Tester que le notebook fonctionne
- [ ] Nettoyer les cellules de code obsolètes

## 🎉 Avantages

✅ **Pas de duplication** : Code défini une seule fois
✅ **Maintenabilité** : Fix un bug → fixé partout
✅ **Documentation** : Docstrings complètes dans `src/`
✅ **Testabilité** : Code isolé et testable
✅ **Professionnalisme** : Structure standard Python

Bon coding ! 🚀
