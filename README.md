# 🚦 Analyse de Traffic 


---

## 📌 Description

Ce projet vise à **prédire le niveau de congestion du trafic 5 minutes dans le futur** à partir de 4 flux vidéo d'un rond-point à la Barbade.

**Targets :** `congestion_enter_rating` et `congestion_exit_rating` (classification multi-label ordinale)  
**Métrique :** Macro-F1 (70%) + Accuracy (30%)  
**Contrainte :** Prédiction en temps réel — il est interdit d'utiliser des données futures (look-ahead) lors de l'entraînement et de l'inférence

---

## 📁 Structure du projet

```
📦 barbados-traffic-challenge
 ┣ 📓 001eda_barbados.ipynb                  # Analyse exploratoire des données
 ┣ 📓 002-feature-extraction.ipynb  # Extraction de features vidéo (EfficientNet-B2)
 ┣ 📓 003-transfert-learning.ipynb        # Modèle Transfer Learning (LSTM + multi-label)
 ┗ 📄 requirements.txt                    # Bibliothèques nécessaires
```

---

## 🚀 Installation et utilisation

### Prérequis

- Python 3.8+
- GPU CUDA (recommandé pour l'extraction de features et l'entraînement)
- Accès au dataset Zindi : [reencoded-barbados-traffic](https://zindi.africa/competitions/barbados-traffic-analysis-challenge/data)

### 1. Cloner le repository

```bash
git clone https://github.com/ton-username/barbados-traffic-challenge.git
cd barbados-traffic-challenge
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Avec venv
python -m venv env
source env/bin/activate        # Linux / macOS
env\Scripts\activate           # Windows

# Ou avec conda
conda create -n barbados python=3.10
conda activate barbados
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> ⚠️ Pour PyTorch avec support CUDA, visiter [pytorch.org](https://pytorch.org/get-started/locally/) et adapter la commande à ta version de CUDA.


### 💡 Alternative — Kaggle (GPU gratuit)

1. Importer les notebooks dans [Kaggle](https://www.kaggle.com)
2. Ajouter le dataset `reencoded-barbados-traffic`
3. Activer le GPU dans les paramètres du notebook
4. Lancer dans l'ordre ci-dessus

---

## 🔍 Notebook 1 — EDA

Analyse exploratoire complète avant toute modélisation :

- Structure, types et valeurs manquantes
- Distribution des 4 classes de congestion (enter + exit)
- Détection du déséquilibre de classes
- Corrélation entre `congestion_enter_rating` et `congestion_exit_rating`
- Analyse des variables catégorielles (`signaling`, `cycle_phase`)
- Patterns temporels via `time_segment_id`
- Heatmap congestion × cycle_phase × signaling
- Série temporelle enter vs exit
- Matrice de corrélation globale

---

## ⚙️ Notebook 2 — Extraction de Features Vidéo

Pipeline d'extraction de représentations vidéo sans backpropagation :

| Composant | Détail |
|-----------|--------|
| Backbone | EfficientNet-B2 (pré-entraîné ImageNet, gelé) |
| Framework | PyTorch + timm + torchcodec |
| Nb frames | 128 frames / vidéo |
| Précision | Float16 (mixed precision) |
| Sortie | Fichiers Parquet (compression Snappy) |
| Pooling | Global Average Pooling |

---

## 🤖 Notebook 3 — Transfer Learning (modèle principal)

Architecture multi-label complète avec pipeline Keras-like :

| Composant | Détail |
|-----------|--------|
| Backbone | VisionBackbone (EfficientNet-B2 gelé + projection) |
| Encodeur temporel | LSTM bidirectionnel (2 couches, hidden=128) |
| Tête de classification | MultiLabelClassificationHead avec attention temporelle |
| Features auxiliaires | Embeddings `signaling` + `cycle_phase` normalisé |
| Loss | CrossEntropyLoss pondérée (class weights) |
| Optimiseur | AdamW (lr=1e-4, weight_decay=1e-4) |
| Dropout | 0.3 |

### Callbacks disponibles

| Callback | Rôle |
|----------|------|
| `ModelCheckpoint` | Sauvegarde du meilleur modèle |
| `EarlyStopping` | Arrêt si pas d'amélioration (patience=5) |
| `ReduceLROnPlateau` | Réduction du LR si stagnation |
| `GradientClipping` | Stabilité (max_norm=1.0) |
| `WarmupScheduler` | Warmup linéaire du LR (2 epochs) |
| `TrainingHistory` | Courbes d'entraînement |

### Métriques utilisées

| Métrique | Justification |
|----------|---------------|
| Accuracy | Métrique de base |
| Balanced Accuracy | Robuste au déséquilibre |
| F1-macro | Sensible aux classes minoritaires |
| F1-weighted | Pondéré par le support |
| QWK (Quadratic Weighted Kappa) | Idéal pour labels ordinaux |

---

## 🔄 Pipeline global

```
Vidéos brutes (4 caméras)
        ↓
  EDA — eda_barbados.ipynb
        ↓
  Extraction features — 005-feature-extraction-torch.ipynb
  (EfficientNet-B2 → embeddings parquet, backbone gelé)
        ↓
  Transfer Learning — 006-transfert-learning.ipynb
  (LSTM bidirectionnel + multi-label + class weights)
        ↓
  Prédiction congestion T+5min → soumission Zindi
```

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-orange)
![timm](https://img.shields.io/badge/timm-latest-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6+-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

