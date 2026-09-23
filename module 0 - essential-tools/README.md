# 🛠️ Module 0 : Outils essentiels

Bienvenue ! Ce module vous aide à préparer un environnement de travail adapté à l’analyse de données M/EEG (EEG/MEG) et aux exercices de programmation du cours.

## Deux façons de démarrer

### Option 1 : Google Colab (débutant recommandé)
Colab est un environnement Jupyter hébergé dans le cloud, sans installation locale.
- Zéro installation, fonctionne dans le navigateur
- Paquets IA/numériques faciles à ajouter
- Sauvegarde automatique

Pour installer des bibliothèques utiles au M/EEG dans un notebook Colab :
```python
!pip install mne mne-bids
```

### Option 2 : Installation locale (plus de contrôle)
Installez Python et VS Code pour travailler hors ligne et gérer finement votre environnement.

## Sommaire
1. [Préparer l’environnement Python](#préparer-lenvironnement-python)
2. [VS Code](#vs-code)
3. [Terminal: commandes de base](#terminal-commandes-de-base)

## Préparer l’environnement Python

### Python — de quoi s’agit-il ?
Un langage polyvalent, lisible et très utilisé en science des données et en neurosciences. L’écosystème open source (NumPy, SciPy, MNE, etc.) est un atout majeur.

### Installation de Python 3.13
Le cours utilise **Python 3.13**. Si une autre version est déjà installée, elle peut rester : on crée l'environnement du cours avec 3.13.

**Windows (10 ou 11)**
1. Aller sur https://www.python.org/downloads/windows/
2. Python 3.13.15 → « Windows installer (64-bit) » (PC ARM : « Windows installer (ARM64) »)
3. Cocher « Add python.exe to PATH » en bas de la première fenêtre, puis « Install Now »
4. Vérifier dans PowerShell : `py -3.13 --version`

**macOS (11 ou plus récent)**
1. Aller sur https://www.python.org/downloads/macos/
2. Python 3.13.15 → « macOS installer » (fichier .pkg)
3. Ouvrir le .pkg et suivre les étapes, puis lancer « Install Certificates.command » (Applications › Python 3.13)
4. Vérifier dans le Terminal : `python3.13 --version`

**Linux**
```bash
# Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.13 python3.13-venv
# Debian 13 (Python 3.13 par défaut)
sudo apt install python3 python3-venv
# Fedora
sudo dnf install python3.13
# Vérifier
python3.13 --version
```

**Autre appareil** (Chromebook, tablette, ordinateur sans droits d'administrateur) : utiliser Google Colab (bouton en haut du README principal).

### Environnements virtuels
Isolez les dépendances par projet (fortement recommandé pour l'analyse M/EEG). Depuis le dossier du dépôt :
```bash
# Créer l'environnement avec Python 3.13
# macOS / Linux
python3.13 -m venv env_meeg
# Windows
py -3.13 -m venv env_meeg

# Activer
# macOS / Linux
source env_meeg/bin/activate
# Windows
env_meeg\Scripts\activate

# Désactiver
deactivate
```
Une fois l'environnement activé, `python` désigne le Python 3.13 de `env_meeg`, quel que soit le système.

Conservez un fichier `requirements.txt` par projet :
```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

### Paquets du cours
Depuis le dossier du dépôt, environnement activé :
```bash
python -m pip install -r requirements.txt
```

### Vérifier l'installation
```bash
python verifier_installation.py
```
Le script doit se terminer par « Environnement prêt. ». Sinon, il indique les paquets manquants.

Dans VS Code, ouvrez un notebook puis choisissez l'environnement `env_meeg` (bouton « Select Kernel » en haut à droite).

## VS Code
Téléchargez VS Code : https://code.visualstudio.com/

Extensions utiles :
- Python (Microsoft)
- Jupyter
- Pylance
- GitLens

Fonctionnalités clés : terminal intégré, debug, Git, IntelliSense.

## Terminal: commandes de base
Sous Windows, utilisez WSL2 ou Git Bash pour des commandes type Unix.

Exemples :
```bash
pwd        # Répertoire courant
ls         # Lister fichiers
cd         # Changer de dossier
mkdir      # Créer un dossier
python --version
pip list
```

Dans un notebook Jupyter/Colab :
```python
!ls
!pip install mne
```

## Dépannage
- Documentation Python : https://docs.python.org/
- VS Code : https://code.visualstudio.com/docs
- MNE-Python : https://mne.tools/stable/index.html

