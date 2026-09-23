# Kit de démarrage — PSY2007D Automne 2026 — Cours M/EEG et oscillations 🚀

> **Projet 2026** : EEG mobile pendant la **marche** et une **tâche lexicale** (+ possiblement le **Five-Point Test**).

## Bienvenue !

Ce kit vous accompagne pour la partie programmation du cours consacré aux données M/EEG (MEG/EEG) et aux oscillations neuronales.

## Contenu du kit

### 🛠️ Module 0 : Outils essentiels
- Terminal et environnements Python
- VS Code et Jupyter
- Démarrage rapide sur Google Colab
- Paquets utiles pour M/EEG (NumPy, SciPy, MNE, Matplotlib, scikit-learn)

### 🔄 Module 1 : Contrôle de version et collaboration (Optionel)
- Git et GitHub
- Dépôts, branches et pull requests
- Bonnes pratiques pour projets d’analyse M/EEG

### 🐍 Module 2 : Python pour les neurosciences
- Bases Python (types, boucles, fonctions)
- Manipulation de données et notebooks
- Signaux 1D : génération, filtrage, DSP (PSD)

### 🧰 Module 3 : Boîte à outils M/EEG
- Bibliothèques clés : NumPy, SciPy, MNE, Matplotlib, scikit-learn
- Prétraitement et caractéristiques (bandes de fréquences)
- Démo ML simple sur des caractéristiques oscillatoires
- Exemples pratiques (scripts prêts à exécuter)

## Par où commencer
1. Commencez par le Module 0 pour préparer l’environnement.
2. Parcourez les modules dans l’ordre.
3. Testez les scripts d’exemples et modifiez-les pour expérimenter.

## Séance 4 (23 septembre 2026) — Outils et environnement de travail

1. **Installer l'environnement** (voir Module 0), puis vérifier :
   ```bash
   python verifier_installation.py
   ```
2. **Notebook de la séance** : [`seance4_outils_python.ipynb`](seance4_outils_python.ipynb)
   [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/PSY2007D-UdeM/blob/main/seance4_outils_python.ipynb)
   - chemins et arborescence BIDS, dictionnaires de conditions et de fenêtres ;
   - epochs EEG comme tableau NumPy `(essais, canaux, temps)`, amplitude moyenne dans une fenêtre ;
   - fonction + boucle + tableau pandas enregistré en CSV (données kiloword, tâche lexicale) ;
   - alignement EEG / plateforme de force et epochs autour des contacts du talon (données de marche simulées).

## Archives 2025

Les notebooks du projet 2025 (Go/NoGo, lecture de phrases, Five-Point : notebooks 01 à 08, notebook de la séance 2) et les diapositives 2025 sont dans la branche [`2025`](https://github.com/thecocolab/PSY2007D-UdeM/tree/2025).
Ils servent d'exemples ; de nouveaux notebooks seront écrits pour les données 2026 (marche + tâche lexicale).

Pour les récupérer en local :
```bash
git switch 2025      # revenir aux fichiers 2025
git switch main      # revenir aux fichiers 2026
```

## Installer localement (clone ou fork) et utiliser avec Colab

### Option A — Fork puis clone (si vous comptez contribuer)
1. Forkez le dépôt sur GitHub (depuis l’interface du dépôt d’origine).
2. Clonez votre fork en local:
   ```bash
   git clone https://github.com/<votre-compte>/<nom-du-repo>.git
   cd <nom-du-repo>
   ```
3. Créez un environnement (Python 3.13, voir Module 0) et installez les dépendances:
   ```bash
   python3.13 -m venv env_meeg      # Windows : py -3.13 -m venv env_meeg
   # macOS/Linux
   source env_meeg/bin/activate
   # Windows
   # env_meeg\Scripts\activate

   pip install -r requirements.txt
   ```
4. (Optionnel) Installez Jupyter et lancez-le:
   ```bash
   pip install jupyter
   jupyter notebook
   ```

### Option B — Clone direct (lecture seule)
```bash
git clone https://github.com/thecocolab/PSY2007D-UdeM.git
cd PSY2007D-UdeM
python3.13 -m venv env_meeg   # Windows : py -3.13 -m venv env_meeg
source env_meeg/bin/activate  # ou env_meeg\Scripts\activate (Windows)
pip install -r requirements.txt
```

### Utiliser ce dépôt dans Google Colab
- Ouvrez un nouveau notebook Colab, puis clonez le dépôt:
  ```python
  !git clone https://github.com/thecocolab/PSY2007D-UdeM.git
  %cd PSY2007D-UdeM
  !pip install -r requirements.txt
  ```
- Exécutez les scripts d’exemple directement dans Colab:
  ```python
  !python "module 3 - python for m_eeg/meeg_synthetic_psd.py"
  ```
- (Optionnel) Montez votre Google Drive pour sauvegarder/charger des fichiers:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

## Prérequis
- Un ordinateur avec accès à Internet
- Envie d’apprendre et d’explorer les données M/EEG ✨

## Besoin d’aide ?
- Chaque module contient des instructions et des exemples détaillés.
- Demandez conseil à l’équipe enseignante si besoin.

## Diapositives
- Séance 4 (2026) : « Outils et environnement de travail », partagées sur StudiUM.
- Diapositives 2025 (« LABO PSY2007D – cours 11/09/2025 ») : branche [`2025`](https://github.com/thecocolab/PSY2007D-UdeM/tree/2025).

## Bon apprentissage ! 🌟

Rappel : la pratique régulière est la clé. Concentrez-vous sur les concepts M/EEG (prétraitement, oscillations, analyse temps-fréquence) tout en consolidant vos bases Python.
