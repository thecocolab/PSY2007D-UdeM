"""
Fonctions de la séance 5 — PSY2007D, automne 2026.

Les notebooks de la séance montrent chaque étape en détail. Ce fichier regroupe les
mêmes étapes en fonctions, pour deux usages :
  - le téléchargement des données (notebook 1) ;
  - le rattrapage : si le notebook 2 (ou 3) ne trouve pas les fichiers produits par le
    notebook précédent, il appelle `preparer_donnees` (ou `preparer_caracteristiques`).

Données : EEG Motor Movement/Imagery Dataset (PhysioNet, Schalk et al., 2004),
runs 3, 7 et 11 : ouvrir/fermer le poing gauche ou droit.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Paramètres communs aux trois notebooks
SUJETS = list(range(1, 11))              # S001 à S010
RUNS = [3, 7, 11]                        # mouvement réel, main gauche ou droite
TACHE = "mouvement"
EVENT_ID = {"repos": 1, "mouvement/gauche": 2, "mouvement/droite": 3}
RENOMMER = {"T0": "repos", "T1": "mouvement/gauche", "T2": "mouvement/droite"}
BANDES = {"theta": (4, 8), "mu": (8, 13), "beta": (13, 30)}
FENETRES_ERP = {"erp100": (0.10, 0.20), "erp300": (0.30, 0.50)}

MIROIR = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/"   # copie de PhysioNet (rapide)
PHYSIONET = "https://physionet.org/files/eegmmidb/1.0.0/"           # site d'origine (lent)


# =====================================================================================
# Notebook 1 : téléchargement, BIDS, prétraitement
# =====================================================================================
def telecharger_eegbci(sujets, runs, dossier, n_parallele=10):
    """Télécharge les fichiers EDF et renvoie leurs chemins (S001/S001R03.edf, ...).

    Essaie d'abord le miroir S3 de PhysioNet, puis PhysioNet. Les fichiers déjà présents
    ne sont pas retéléchargés. Chaque fichier est vérifié par sa somme de contrôle.
    """
    import pooch
    from importlib.resources import files

    lignes = files("mne").joinpath("data", "eegbci_checksums.txt").read_text().splitlines()
    registre = dict(ligne.split()[:2] for ligne in lignes if ligne.strip())
    dossier = Path(dossier)

    def un_fichier(nom):
        cible = dossier / nom
        if cible.exists():
            return cible
        erreur = None
        for base in (MIROIR, PHYSIONET):
            try:
                pooch.retrieve(base + nom, known_hash=registre.get(nom), fname=cible.name,
                               path=cible.parent, progressbar=False)
                return cible
            except Exception as err:          # serveur indisponible : essayer le suivant
                erreur = err
        raise erreur

    noms = [f"S{s:03d}/S{s:03d}R{r:02d}.edf" for s in sujets for r in runs]
    with ThreadPoolExecutor(n_parallele) as pool:
        return list(pool.map(un_fichier, noms))


def edf_vers_bids(chemin_edf, sujet, run, racine):
    """Écrit un fichier EDF dans l'arborescence BIDS (données brutes, non modifiées)."""
    import mne
    from mne.datasets import eegbci
    from mne_bids import BIDSPath, write_raw_bids

    raw = mne.io.read_raw_edf(chemin_edf, preload=False, verbose=False)
    eegbci.standardize(raw)                        # noms standard : "Fc5." -> "FC5"
    raw.set_montage("colin27_1005")                # positions standard des électrodes (10-05)
    raw.annotations.rename(RENOMMER)               # T0/T1/T2 -> noms lisibles
    raw.info["line_freq"] = 60                     # secteur à 60 Hz (États-Unis)
    chemin_bids = BIDSPath(subject=f"{sujet:03d}", task=TACHE, run=f"{run:02d}",
                           datatype="eeg", root=racine)
    write_raw_bids(raw, chemin_bids, event_id=EVENT_ID, overwrite=True, verbose=False)
    return chemin_bids


def charger_bids(racine, sujet):
    """Lit les trois runs d'un participant dans BIDS et les met bout à bout."""
    import mne
    from mne_bids import BIDSPath, read_raw_bids

    raws = []
    for run in RUNS:
        chemin = BIDSPath(subject=f"{sujet:03d}", task=TACHE, run=f"{run:02d}",
                          datatype="eeg", root=racine)
        raws.append(read_raw_bids(chemin, on_ch_mismatch="rename", verbose=False).load_data())
    return mne.concatenate_raws(raws)


def pretraiter(raw, rapport=None):
    """Chaîne de prétraitement de la séance (mne + mne-denoise). Renvoie une copie nettoyée.

    Si `rapport` est un dictionnaire, il reçoit quelques mesures de contrôle.
    """
    import mne
    from mne_denoise.asr import ASR
    from mne_denoise.dss import DSS, CycleAverageBias
    from mne_denoise.zapline import ZapLine

    raw = raw.copy().load_data()
    raw.resample(180.0)                                    # 180 / 60 = 3 échantillons par cycle
    raw.filter(l_freq=1.0, h_freq=None)                    # retire la dérive lente
    zapline = ZapLine(sfreq=180.0, line_freq=60.0)
    raw = zapline.fit_transform(raw)                       # bruit du secteur

    clignements = mne.preprocessing.find_eog_events(raw, ch_name=["Fp1", "Fp2"], verbose=False)
    echantillons = clignements[:, 0] - raw.first_samp
    marge = int(0.5 * raw.info["sfreq"])
    echantillons = echantillons[(echantillons > marge) & (echantillons < raw.n_times - marge)]
    if len(echantillons) >= 5:                             # assez de clignements pour la DSS
        biais = CycleAverageBias(echantillons, window=(-0.4, 0.4), window_unit="seconds",
                                 sfreq=raw.info["sfreq"])
        raw = DSS(bias=biais, n_select=1, component_action="subtract").fit_transform(raw)

    asr = ASR(cutoff=20)
    raw = asr.fit_transform(raw)                           # bouffées de grande amplitude
    raw.filter(l_freq=None, h_freq=40.0)                   # on analyse sous 40 Hz
    raw.resample(100.0)                                    # 100 Hz suffit sous 40 Hz
    raw.set_eeg_reference("average", verbose=False)

    if rapport is not None:
        rapport["zapline_composantes"] = int(zapline.n_removed_)
        rapport["clignements"] = len(echantillons)
        rapport["asr_fenetres_%"] = round(100 * asr.fraction_reconstructed_windows_, 1)
    return raw


def chemin_propre(racine, sujet):
    """BIDSPath du fichier prétraité d'un participant (dans derivatives/pretraitement)."""
    from mne_bids import BIDSPath

    return BIDSPath(root=Path(racine) / "derivatives" / "pretraitement", subject=f"{sujet:03d}",
                    task=TACHE, processing="clean", datatype="eeg", suffix="eeg",
                    extension=".fif", check=False)


def preparer_donnees(racine, sujets=SUJETS):
    """Refait tout le notebook 1 (téléchargement, BIDS, prétraitement) si nécessaire."""
    import mne
    from mne_bids import make_dataset_description

    racine = Path(racine)
    mne.set_log_level("ERROR")
    for sujet in sujets:
        sortie = chemin_propre(racine, sujet)
        if sortie.fpath.exists():
            continue
        print(f"Préparation de S{sujet:03d} ...")
        chemins = telecharger_eegbci([sujet], RUNS, racine / "sourcedata" / "eegbci")
        for chemin, run in zip(chemins, RUNS):
            edf_vers_bids(chemin, sujet, run, racine)
        raw_propre = pretraiter(charger_bids(racine, sujet))
        sortie.mkdir()
        raw_propre.save(sortie.fpath, overwrite=True)
    make_dataset_description(path=racine / "derivatives" / "pretraitement",
                             name="PSY2007D séance 5 : prétraitement", dataset_type="derivative",
                             generated_by=[{"Name": "mne-denoise"}, {"Name": "MNE-Python"}],
                             overwrite=True, verbose=False)


# =====================================================================================
# Notebook 2 : caractéristiques par essai
# =====================================================================================
def decouper(raw, sujet, tmin=-1.0, tmax=4.0):
    """Epochs autour de chaque indice (repos, mouvement/gauche, mouvement/droite)."""
    import mne

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)
    noms = {v: k for k, v in EVENT_ID.items()}
    etiquettes = [noms[code] for code in events[:, 2]]
    metadata = pd.DataFrame({
        "sujet": sujet,
        "condition": [e.split("/")[-1] for e in etiquettes],      # repos, gauche, droite
        "mouvement": [int(e.startswith("mouvement")) for e in etiquettes],
    })
    return mne.Epochs(raw, events, EVENT_ID, tmin=tmin, tmax=tmax, baseline=None,
                      metadata=metadata, preload=True, verbose=False)


def caracteristiques_essais(epochs):
    """Tableau pandas : une ligne par essai, une colonne par (mesure, canal)."""
    import antropy as ant
    from specparam import SpectralGroupModel

    canaux = epochs.ch_names
    colonnes = {}

    # 1. Puissance par bande (log10 µV²/Hz), fenêtre 0,5 à 4 s
    spectre = epochs.compute_psd(method="welch", tmin=0.5, tmax=4.0, fmin=2, fmax=40,
                                 n_fft=100, verbose=False)
    psd, freqs = spectre.get_data() * 1e12, spectre.freqs
    for bande, (fmin, fmax) in BANDES.items():
        puissance = np.log10(psd[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1))
        colonnes.update({f"{bande}_{c}": puissance[:, i] for i, c in enumerate(canaux)})

    # 2. Exposant apériodique (specparam), un ajustement par essai et par canal
    modele = SpectralGroupModel(peak_width_limits=(1, 8), max_n_peaks=3, aperiodic_mode="fixed",
                                verbose=False)
    modele.fit(freqs, psd.reshape(-1, len(freqs)), freq_range=(3, 40))
    exposant = modele.get_params("aperiodic", "exponent").reshape(len(epochs), len(canaux))
    colonnes.update({f"expo_{c}": exposant[:, i] for i, c in enumerate(canaux)})

    # 3. Complexité : entropie de permutation et Lempel-Ziv, fenêtre 0,5 à 4 s
    x = epochs.copy().crop(0.5, 4.0).get_data()
    pe = np.apply_along_axis(lambda s: ant.perm_entropy(s, order=3, delay=1, normalize=True), -1, x)
    lz = np.apply_along_axis(lambda s: ant.lziv_complexity(s > np.median(s), normalize=True), -1, x)
    colonnes.update({f"pe_{c}": pe[:, i] for i, c in enumerate(canaux)})
    colonnes.update({f"lz_{c}": lz[:, i] for i, c in enumerate(canaux)})

    # 4. ERP : amplitude moyenne (µV) dans deux fenêtres, ligne de base -0,2 à 0 s
    erp = epochs.copy().crop(-0.2, 0.8).apply_baseline((-0.2, 0), verbose=False)
    for nom, (debut, fin) in FENETRES_ERP.items():
        amplitude = erp.copy().crop(debut, fin).get_data().mean(axis=-1) * 1e6
        colonnes.update({f"{nom}_{c}": amplitude[:, i] for i, c in enumerate(canaux)})

    tableau = pd.concat([epochs.metadata.reset_index(drop=True), pd.DataFrame(colonnes)], axis=1)
    tableau.insert(1, "essai", np.arange(len(tableau)))
    return tableau


def chemin_caracteristiques(racine):
    return Path(racine) / "derivatives" / "caracteristiques" / "caracteristiques_essais.csv"


def chemin_epochs_erp(racine, sujet):
    """BIDSPath des epochs courtes (-0,2 à 1 s) utilisées pour le décodage dans le temps."""
    from mne_bids import BIDSPath

    return BIDSPath(root=Path(racine) / "derivatives" / "caracteristiques", subject=f"{sujet:03d}",
                    task=TACHE, processing="clean", description="erp", datatype="eeg",
                    suffix="epo", extension=".fif", check=False)


def preparer_caracteristiques(racine, sujets=SUJETS):
    """Refait la dernière partie du notebook 2 (tableau + epochs ERP) si nécessaire."""
    import mne

    mne.set_log_level("ERROR")
    preparer_donnees(racine, sujets)
    tableaux = []
    for sujet in sujets:
        print(f"Caractéristiques de S{sujet:03d} ...")
        raw = mne.io.read_raw_fif(chemin_propre(racine, sujet).fpath, preload=True)
        epochs = decouper(raw, sujet)
        tableaux.append(caracteristiques_essais(epochs))
        erp = epochs.copy().crop(-0.2, 1.0).apply_baseline((-0.2, 0))
        sortie = chemin_epochs_erp(racine, sujet)
        sortie.mkdir()
        erp.save(sortie.fpath, overwrite=True)
    fichier = chemin_caracteristiques(racine)
    fichier.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tableaux, ignore_index=True).to_csv(fichier, index=False)
    return fichier
