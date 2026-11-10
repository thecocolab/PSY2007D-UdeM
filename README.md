# PSY2007D — Laboratoire 1 : Connectivité et oscillations cérébrales (automne 2026)

Projet 2026 : EEG mobile pendant la marche et une tâche lexicale.

Les fichiers de 2025 (notebooks 01 à 08, séance 2, diapositives) sont dans la branche [`2025`](https://github.com/thecocolab/PSY2007D-UdeM/tree/2025).

## Projet Brain Walk — décision lexicale, assis et en marche

Les notebooks du projet travaillent sur les données du projet Brain Walk : une décision lexicale (mots fréquents, mots rares,
pseudo-mots) faite en position assise et en marchant sur un tapis roulant, enregistrée avec un casque DSI-24 à électrodes sèches
(19 électrodes et un accéléromètre), chez 17 participants.

| Fichier | Rôle |
|---|---|
| `brainwalk_1_pretraitement.ipynb` | Le prétraitement, pas à pas, sur un participant (assis, puis ce qui change en marche) ; la même chaîne pour tous (`outils_brainwalk.py`) |
| `09_analysis1_lexicaldecision.ipynb` | L'analyse : comportement, ERP, composantes (P2, N400, LPC) et tests, sur tous les participants |
| `10_analysis2_lexicaldecision.ipynb` | Les rythmes : spectres, temps-fréquence |
| `11_apprentissage_machine_lexicaldecision.ipynb` | L'apprentissage automatique : reconnaître la condition d'un essai |

Sans installation :
- prétraitement : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/PSY2007D-UdeM/blob/main/brainwalk_1_pretraitement.ipynb)

**Les données.** La première cellule de chaque notebook télécharge les données du projet (2,2 Go, depuis Google Drive) et les
décompresse dans `tasks/brainwalk/`, à côté des notebooks. Elle ne le fait qu'une fois : si les données sont déjà là, elle passe. Le
dossier `tasks/` est ignoré par Git. Il n'y a rien à télécharger à la main. Dans Colab, les fichiers sont effacés à la fin de chaque
session : la cellule retélécharge alors les données. Le travail sur votre ordinateur est donc recommandé.

Les fichiers prétraités fournis (`tasks/brainwalk/pretraite_00/`) ont été faits avec la chaîne du notebook `brainwalk_1` : les notebooks
d'analyse partent de ces fichiers, et ne dépendent pas du notebook de prétraitement.

Le notebook `brainwalk_1` couvre :
- les données (BIDS), les mauvaises électrodes et les périodes suspectes, les filtres, l'Auto-CCA (muscles), la référence moyenne ;
- l'ICA (Picard assis, AMICA en marche), ICLabel et une règle de décision par preuves ;
- les époques, un contrôle local des électrodes, et les ERP (P2, N400, LPC) avant et après le nettoyage ;
- en marche : la régression sur l'accéléromètre du casque et le gabarit de foulée ; puis l'export des fichiers pour tous les participants.

## Séance 4 (23 septembre 2026) — Outils et environnement de travail

| Fichier | Rôle |
|---|---|
| `seance4_outils_python.ipynb` | Notebook de la séance : outils Python |
| `seance4_intro_mne.ipynb` | Notebook de la séance : introduction à MNE |
| `verifier_installation.py` | Vérifie Python, les paquets du cours et MNE |
| `requirements.txt` | Paquets du cours |

Sans installation :
- outils Python : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/PSY2007D-UdeM/blob/main/seance4_outils_python.ipynb)
- introduction à MNE : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/PSY2007D-UdeM/blob/main/seance4_intro_mne.ipynb)

Le notebook `seance4_outils_python.ipynb` couvre :
- chemins et arborescence BIDS, dictionnaires de conditions et de fenêtres ;
- epochs EEG comme tableau NumPy `(essais, canaux, temps)`, amplitude moyenne dans une fenêtre ;
- fonction + boucle + tableau pandas enregistré en CSV (données kiloword, tâche lexicale) ;
- alignement EEG / plateforme de force et epochs autour des contacts du talon (données de marche simulées).

Le notebook `seance4_intro_mne.ipynb` couvre (données kiloword, signal continu reconstruit avec bruit ajouté) :
- `Raw` et `info`, positions des électrodes, visualisation du signal ;
- spectre de puissance (PSD), filtrage passe-bande et coupe-bande ;
- événements, `Epochs` avec métadonnées ;
- `Evoked`, comparaison de conditions, cartes topographiques.

## Séance 5 — Du signal brut au décodage

| Fichier | Rôle |
|---|---|
| `seance5_1_pretraitement_bids.ipynb` | Notebook 1 : télécharger, organiser en BIDS, nettoyer avec mne-denoise |
| `seance5_2_caracteristiques.ipynb` | Notebook 2 : ERP, spectres, temps-fréquence, 1/f, complexité, connectivité |
| `seance5_3_apprentissage.ipynb` | Notebook 3 : apprentissage automatique (repos/mouvement, gauche/droite) |
| `outils_seance5.py` | Fonctions communes : téléchargement, et rattrapage automatique si un notebook précédent n'a pas été exécuté |

Sans installation :
- notebook 1 : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/PSY2007D-UdeM/blob/main/seance5_1_pretraitement_bids.ipynb)
- notebook 2 : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/PSY2007D-UdeM/blob/main/seance5_2_caracteristiques.ipynb)
- notebook 3 : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecocolab/PSY2007D-UdeM/blob/main/seance5_3_apprentissage.ipynb)

Les trois notebooks s'enchaînent : chacun lit les fichiers écrits par le précédent (dossier `donnees_seance5/`, ou `MyDrive/PSY2007D/donnees_seance5` dans Colab). S'ils manquent, le notebook les refait avec `outils_seance5.py`.

Données : EEG Motor Movement/Imagery Dataset (PhysioNet ; Schalk et al., 2004), 10 participants, runs 3, 7 et 11 (ouvrir/fermer le poing gauche ou droit), 64 électrodes. Environ 75 Mo, téléchargés depuis la copie Amazon S3 de PhysioNet.

Le notebook 1 couvre :
- téléchargement et vérification des fichiers EDF ; noms des canaux, positions, événements ;
- écriture et lecture BIDS avec `mne-bids` (`BIDSPath`, `write_raw_bids`, `read_raw_bids`, fichiers d'accompagnement) ;
- nettoyage avec `mne-denoise` : ZapLine (secteur 60 Hz), DSS (clignements), ASR (bouffées d'artefacts) ; filtres et référence moyenne ;
- enregistrement dans `derivatives/pretraitement/`.

Le notebook 2 couvre :
- epochs avec métadonnées ; ERP, grand average, potentiel latéralisé et contrôle des mouvements des yeux par régression ;
- spectres de puissance, bandes de fréquence, désynchronisation μ/β (ERD) par canal et par participant ;
- temps-fréquence (ondelettes de Morlet) : ERD et rebond β ;
- composante apériodique avec `specparam` ; entropie de permutation et Lempel-Ziv avec `antropy` ; cohérence et wPLI avec `mne-connectivity` ;
- tableau des caractéristiques (une ligne par essai) dans `derivatives/caracteristiques/`.

Le notebook 3 couvre :
- régression logistique, validation croisée, niveau du hasard et test de permutation ;
- une ou plusieurs caractéristiques, un capteur ou tous, un modèle par capteur (carte des scores), familles de caractéristiques ;
- intra-sujet vs nouvelle personne (leave-one-subject-out) ;
- gauche/droite : spectre, ERP, piège des mouvements des yeux, décodage dans le temps, CSP.

Durée d'exécution sur un portable récent : environ 3 min (notebook 1, téléchargement compris), 3 min (notebook 2), 1 min (notebook 3).

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
