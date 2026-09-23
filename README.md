# PSY2007D — Laboratoire 1 : Connectivité et oscillations cérébrales (automne 2026)

Projet 2026 : EEG mobile pendant la marche et une tâche lexicale.

Les fichiers de 2025 (notebooks 01 à 08, séance 2, diapositives) sont dans la branche [`2025`](https://github.com/thecocolab/PSY2007D-UdeM/tree/2025).

## Séance 4 (23 septembre 2026) — Outils et environnement de travail

| Fichier | Rôle |
|---|---|
| `seance4_outils_python.ipynb` | Notebook de la séance |
| `verifier_installation.py` | Vérifie Python, les paquets du cours et MNE |
| `requirements.txt` | Paquets du cours |

Sans installation : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/PSY2007D-UdeM/blob/main/seance4_outils_python.ipynb)

Le notebook couvre :
- chemins et arborescence BIDS, dictionnaires de conditions et de fenêtres ;
- epochs EEG comme tableau NumPy `(essais, canaux, temps)`, amplitude moyenne dans une fenêtre ;
- fonction + boucle + tableau pandas enregistré en CSV (données kiloword, tâche lexicale) ;
- alignement EEG / plateforme de force et epochs autour des contacts du talon (données de marche simulées).

## 1. Installer Python 3.13

**Windows (10 ou 11)**
1. Aller sur https://www.python.org/downloads/windows/
2. Python 3.13.15 → « Windows installer (64-bit) » (PC ARM : « Windows installer (ARM64) »)
3. Cocher « Add python.exe to PATH » en bas de la première fenêtre, puis « Install Now »
4. Vérifier dans PowerShell : `py -3.13 --version`

**macOS (11 ou plus récent)**
1. Aller sur https://www.python.org/downloads/macos/
2. Python 3.13.15 → « macOS installer » (fichier .pkg)
3. Ouvrir le .pkg, puis lancer « Install Certificates.command » (Applications › Python 3.13)
4. Vérifier dans le Terminal : `python3.13 --version`

**Linux**
```bash
# Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.13 python3.13-venv
# Debian 13
sudo apt install python3 python3-venv
# Fedora
sudo dnf install python3.13
```

**Autre appareil** (Chromebook, tablette, ordinateur sans droits d'administrateur) : utiliser le bouton Colab ci-dessus.

## 2. VS Code et Git

- VS Code : https://code.visualstudio.com/, puis les extensions **Python** et **Jupyter** (icône Extensions, Ctrl/Cmd + Maj + X).
- Git : https://git-scm.com/ sous Windows ; sous macOS, taper `git` dans le Terminal et accepter l'installation proposée.

## 3. Dépôt, environnement et paquets

```bash
git clone https://github.com/thecocolab/PSY2007D-UdeM.git
cd PSY2007D-UdeM

# macOS / Linux
python3.13 -m venv env_meeg
source env_meeg/bin/activate
# Windows
py -3.13 -m venv env_meeg
env_meeg\Scripts\activate

python -m pip install -r requirements.txt
```

Une fois l'environnement activé, `python` désigne le Python 3.13 de `env_meeg`. À chaque nouvelle fenêtre de terminal, réactiver l'environnement.

## 4. Vérifier

```bash
python verifier_installation.py
```

La dernière ligne doit être « Environnement prêt. ». Sinon, le script affiche la commande `pip` à lancer.

Dans VS Code, ouvrir le notebook puis choisir le noyau `env_meeg` (bouton « Select Kernel » en haut à droite).

## Problèmes fréquents

| Message | Solution |
|---|---|
| Windows : `python` n'est pas reconnu | Utiliser `py -3.13`, ou réinstaller en cochant « Add python.exe to PATH » |
| PowerShell : `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| macOS : `command not found: python` | Taper `python3.13` tant que `env_meeg` n'est pas activé |
| macOS : `CERTIFICATE_VERIFY_FAILED` | Lancer « Install Certificates.command » |
| Ubuntu : `ensurepip is not available` | `sudo apt install python3.13-venv`, puis recréer `env_meeg` |
| VS Code : `ModuleNotFoundError: No module named 'mne'` | Mauvais noyau : Select Kernel → `env_meeg` |

## Mises à jour

Chaque semaine, avant d'ouvrir les notebooks :
```bash
git pull
```
