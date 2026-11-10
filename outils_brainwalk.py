"""
Fonctions du projet Brain Walk — PSY2007D, automne 2026.

Le notebook `brainwalk_1_pretraitement.ipynb` montre chaque étape du prétraitement en détail, sur un
participant. Ce fichier regroupe les mêmes étapes en fonctions, pour trois usages :
  - le téléchargement des données (tous les notebooks Brain Walk) ;
  - `pretraiter(participant, posture)` : toute la chaîne d'un coup, pour la boucle sur les 17 participants
    et pour vérifier que le notebook et ce fichier donnent le même résultat ;
  - `exporter(...)` : les fichiers de `pretraite_00/` que lisent les notebooks 2, 3 et 4.

Les étapes qui modifient les données sont écrites et expliquées dans le notebook ; elles sont recopiées
ici sans changement. Les figures du notebook sont aussi ici, pour garder le notebook court.

Données : étude Brain Walk, décision lexicale assis et en marchant sur un tapis, casque sec DSI-24
(19 électrodes, 300 Hz). Description complète : tasks/brainwalk/LISEZMOI.md.
"""

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================================================
# Données
# =====================================================================================================
DOSSIER = Path("tasks/brainwalk")              # les données, dans tasks/ à côté des notebooks
ID_DRIVE = "19nnPiRuypz5fCPISybipdjxeop2EA_P8"   # le fichier labo1_donnees_2026.zip sur Google Drive
EMPREINTE = "caa7140ef7a4ddab"                  # l'empreinte de manifest.json dans cette version des données

PARTICIPANTS = ["P002", "P003", "P004", "P006", "P007", "P008", "P009", "P010", "P012",
                "P014", "P015", "P016", "P017", "P018", "P019", "P020", "P021"]
POSTURES = ["assis", "marche"]

# Les enregistrements de chaque posture : notre nom court -> les étiquettes tâche et acquisition du nom BIDS
ETIQUETTES = {
    "assis": {"repos_debut": "task-RestingState_acq-seatedPre",
              "bloc1": "task-LexicalDecision_acq-seatedBlock01",
              "bloc2": "task-LexicalDecision_acq-seatedBlock02",
              "bloc3": "task-LexicalDecision_acq-seatedBlock03",
              "bloc4": "task-LexicalDecision_acq-seatedBlock04",
              "repos_fin": "task-RestingState_acq-seatedPost"},
    "marche": {"marche_seule": "task-WalkingOnly_acq-walking",
               "bloc1": "task-LexicalDecision_acq-walkingBlock01",
               "bloc2": "task-LexicalDecision_acq-walkingBlock02",
               "bloc3": "task-LexicalDecision_acq-walkingBlock03",
               "bloc4": "task-LexicalDecision_acq-walkingBlock04"},
}
# Les déclencheurs entre lesquels on garde un enregistrement sans tâche (début, fin)
DECLENCHEURS = {"repos_debut": ("rest_start", "rest_end"), "repos_fin": ("rest_start", "rest_end"),
                "marche_seule": ("walk_start", "walk_end")}

# =====================================================================================================
# Réglages de la chaîne (les mêmes que dans le notebook, partie 0)
# =====================================================================================================
# Étape 3 : mauvaises électrodes et périodes suspectes
RAPPORT_PLAT = 0.1          # plate : sous ce rapport à l'amplitude typique (sur le signal enregistré)
ET_PLAT_UV = 1.0            # plate aussi si son écart-type est sous cette valeur (µV)
RAPPORT_BRUIT = 3.0         # bruitée : au-dessus de ce rapport à l'amplitude typique
PART_BRUIT = 0.5            # en marche : bruitée dans plus de cette part de ses fenêtres de 5 s
MAX_MAUVAISES = 3           # plus de mauvaises électrodes que cela : regarder l'enregistrement
FENETRE_QC_S = 5.0          # les fenêtres du contrôle de qualité (s)
RAPPORT_QC = 3.0            # période suspecte : une électrode, ou tout le scalp, au-dessus de ce rapport

# Étape 4 : le filtre des données analysées
BANDE_ANALYSE = (0.1, 100.0)

# Étape 5 (assis) : Auto-CCA
CCA_FENETRE_S = 10.0        # fenêtres de 10 s qui se chevauchent de moitié
CCA_BANDE_PENTE = (7.0, 75.0)
CCA_PENTE = 0.0             # composante retirée si la pente de son spectre est au-dessus de 0

# Étapes 7 et 8 : ICA
BANDE_ICA = {"assis": (1.0, 100.0), "marche": (2.5, 100.0)}
TRANSITION_ICA_MARCHE = 1.0  # en marche : demi-amplitude à 2 Hz
BANDE_ICLABEL = (1.0, 100.0)
RAPPORT_HORS_ICA = 5.0      # assis : une période QC au-dessus de ce rapport est retirée de l'apprentissage de l'ICA
GRAINE_ICA = 97
# Décision assis : deux preuves du même genre doivent s'accorder
P_ICLABEL = 0.5
CORR_OEIL_ASSIS = 0.5
SCORE_MUSCLE = 0.5
PRUDENCE_CERVEAU = 0.3
# Décision en marche : la règle des analyses Brain Walk
P_OEIL = 0.6
CORR_OEIL_MARCHE = 0.4
P_MUSCLE = 0.5
FOCAL = 0.6
VARIANCE_ELECTRODE = 10.0
# AMICA (en marche)
AMICA_ITERATIONS = 500
AMICA_DECIM = 2
AMICA_REJET = dict(do_reject=True, rejsig=3.0, numrej=10, rejstart=2, rejint=3)

# En marche : clignements, accéléromètre, gabarit de foulée
Z_CLIGNEMENT = 5.0
BANDE_ACCEL = (0.5, 12.0)
DELAIS_ACCEL_S = np.arange(0, 0.21, 0.02)
FENETRE_ACCEL_S = 60.0
PENALITES_ACCEL = (0.01, 0.1, 1.0, 10.0)
GAIN_MIN_ACCEL = 0.02
FOULEES_AUTOUR = 10
PASSE_HAUT_FOULEE = 1.0

# Étape 10 : époques, contrôle local, ERP
TMIN, TMAX = -0.2, 0.8
LIGNE_DE_BASE = (-0.2, 0.0)
BANDE_ERP = (1.0, 40.0)
BIP_MAX_MS = 5.0
ECART_QC = 3.0
DERIVE_UV = 46.2
MAX_LAISSEES = 0.5
FACTEUR_SME = 2.0
PARTENAIRES = {"C3": "C4", "C4": "C3", "P3": "P4", "P4": "P3"}
COMPOSANTES = {
    "P2": dict(fenetre=(0.14, 0.22), region=["F3", "Fz", "F4", "C3", "Cz", "C4"]),
    "N400": dict(fenetre=(0.30, 0.50), region=["C3", "Cz", "C4", "P3", "Pz", "P4"]),
    "LPC": dict(fenetre=(0.45, 0.80), region=["C3", "Cz", "C4", "P3", "Pz", "P4"]),
}
CONDITIONS = {"mot/HF": "tab:blue", "mot/LF": "tab:green", "pseudo": "tab:red"}
EVENT_ID = {"mot/HF": 1, "mot/LF": 2, "pseudo": 3}

# Export (pretraite_00/)
TYPES = {"mot/HF": "HF", "mot/LF": "LF", "pseudo": "pseudo"}
CODES_EXPORT = {"assis/HF": 1, "assis/LF": 2, "assis/pseudo": 3, "marche/HF": 4, "marche/LF": 5, "marche/pseudo": 6}
REGION_TAPE = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]
MAX_TAPE_UV = 300.0


# =====================================================================================================
# 0. Télécharger les données
# =====================================================================================================
def donnees_a_jour(dossier=DOSSIER):
    """True si `dossier` contient cette version des données (l'empreinte de son manifest.json)."""
    manifeste = Path(dossier) / "manifest.json"
    return manifeste.exists() and hashlib.sha256(manifeste.read_bytes()).hexdigest()[:16] == EMPREINTE


def telecharger_donnees(dossier=DOSSIER):
    """Télécharge et décompresse les données (2,2 Go) si elles manquent ou sont d'une ancienne version."""
    dossier = Path(dossier)
    if donnees_a_jour(dossier):
        print("Données :", dossier, "(prêtes)")
        return dossier
    archive = dossier.parent / "labo1_donnees_2026.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)

    def progression(blocs, taille, total):
        if total > 0 and blocs * taille * 10 // total > (blocs - 1) * taille * 10 // total:
            print(f"  {min(100, 100 * blocs * taille // total)} %")

    print("Téléchargement des données depuis Google Drive (2,2 Go, quelques minutes) :")
    urllib.request.urlretrieve(f"https://drive.usercontent.google.com/download?id={ID_DRIVE}&export=download&confirm=t",
                               archive, progression)
    if not zipfile.is_zipfile(archive):            # Google Drive a envoyé une page web au lieu du fichier
        archive.unlink()
        raise RuntimeError("Google Drive n'a pas envoyé les données (trop de téléchargements aujourd'hui ?). Réessayez plus tard.")
    print("Décompression ...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dossier.parent)               # crée tasks/brainwalk/
    archive.unlink()                               # l'archive ne sert plus
    print("Données :", dossier, "(prêtes)" if donnees_a_jour(dossier) else "(INTROUVABLES : relancez la cellule)")
    return dossier


# =====================================================================================================
# 1-2. Lire, couper aux déclencheurs, étiqueter les mots, joindre les blocs
# =====================================================================================================
def dossier_eeg(participant, dossier=DOSSIER):
    return Path(dossier) / "bids" / f"sub-{participant}" / "ses-001" / "eeg"


def fichier_bids(participant, posture, nom, suffixe, dossier=DOSSIER):
    """Le chemin d'un fichier BIDS d'un enregistrement (ex. 'bloc1', 'events.tsv'), ou None. Un enregistrement
    recommencé après un problème a le numéro de run suivant : on prend le dernier."""
    motif = f"sub-{participant}_ses-001_{ETIQUETTES[posture][nom]}_run-*_{suffixe}"
    return max(dossier_eeg(participant, dossier).glob(motif), default=None)


def lire_enregistrements(participant, posture, dossier=DOSSIER):
    """Les enregistrements bruts d'une posture (dict nom -> Raw) et leurs événements (dict nom -> DataFrame)."""
    import mne

    raws, evenements = {}, {}
    for nom in ETIQUETTES[posture]:
        vhdr = fichier_bids(participant, posture, nom, "eeg.vhdr", dossier)
        if vhdr is None:
            continue
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
        raw.set_channel_types({"CM": "misc", "A1": "misc", "A2": "misc", "Trigger": "stim"})
        raw.set_montage("standard_1020")
        raws[nom] = raw
        evenements[nom] = pd.read_csv(fichier_bids(participant, posture, nom, "events.tsv", dossier), sep="\t",
                                      na_values="n/a")
    return raws, evenements


def condition_de(ligne):
    """mot/HF, mot/LF ou pseudo, à partir d'une ligne d'un fichier events.tsv."""
    if ligne["lexicality"] == "pseudoword":
        return "pseudo"
    return "mot/HF" if ligne["frequency_class"] == "high" else "mot/LF"


def etiqueter_mots(raw, ev):
    """Une copie de raw dont les annotations sont les mots (nommés par condition) et les pertes de données (BAD)."""
    import mne

    raw = raw.copy()
    sf = raw.info["sfreq"]
    mots = ev[ev["trial_type"] == "stimulus"]
    annotations = mne.Annotations(onset=mots["sample"] / sf, duration=0.0,
                                  description=list(mots.apply(condition_de, axis=1)))
    pertes = ev.loc[ev["trial_type"] == "data_gap", "sample"] / sf
    if len(pertes):
        annotations += mne.Annotations(onset=np.maximum(pertes - 1, 0), duration=2.0, description="BAD_perte")
    raw.set_annotations(annotations)
    return raw


def couper_aux_declencheurs(raw, ev, premier, dernier):
    """Une copie de raw coupée entre deux déclencheurs (ou au début / à la fin du fichier s'il en manque un)."""
    debut = ev.loc[ev["trial_type"] == premier, "onset"]
    fin = ev.loc[ev["trial_type"] == dernier, "onset"]
    t0 = debut.iloc[0] if len(debut) else 0.0
    t1 = min(fin.iloc[0], raw.times[-1]) if len(fin) else raw.times[-1]
    return raw.copy().crop(t0, t1)


def assembler(raws, evenements):
    """Coupe chaque enregistrement à ses déclencheurs, étiquette les mots et joint les blocs.

    Rend : coupes (nom -> Raw coupé), enregistrements (sans tâche + 'tache'), essais (une ligne par mot gardé),
    parties (nom de la coupe -> (enregistrement, premier échantillon dans cet enregistrement))."""
    import mne

    blocs = [n for n in raws if n.startswith("bloc")]
    coupes = {}
    for nom in raws:
        if nom in blocs:
            coupes[nom] = couper_aux_declencheurs(etiqueter_mots(raws[nom], evenements[nom]), evenements[nom],
                                                  "recording_start", "recording_stop")
        else:
            coupes[nom] = couper_aux_declencheurs(raws[nom].copy().set_annotations(None), evenements[nom],
                                                  *DECLENCHEURS[nom])
    enregistrements = {}
    for nom in coupes:
        if nom not in blocs and nom != "repos_fin":
            enregistrements[nom] = coupes[nom].copy()
    enregistrements["tache"] = mne.concatenate_raws([coupes[b].copy() for b in blocs])
    if "repos_fin" in coupes:
        enregistrements["repos_fin"] = coupes["repos_fin"].copy()
    essais, parties = essais_et_parties(raws, evenements, coupes)
    return coupes, enregistrements, essais, parties


def essais_et_parties(raws, evenements, coupes):
    """Les essais gardés par les coupes (une ligne par mot) et la place de chaque coupe dans les enregistrements
    (nom de la coupe -> (enregistrement, premier échantillon dans cet enregistrement))."""
    blocs = [n for n in raws if n.startswith("bloc")]
    premier = {b: coupes[b].first_samp - raws[b].first_samp for b in blocs}
    essais = pd.concat([evenements[b][(evenements[b]["trial_type"] == "stimulus")
                                      & evenements[b]["sample"].between(premier[b], premier[b] + coupes[b].n_times - 1)]
                        for b in blocs], ignore_index=True)
    essais["condition"] = essais.apply(condition_de, axis=1)
    parties, debut = {}, 0
    for nom, raw in coupes.items():
        parties[nom] = ("tache", debut) if nom in blocs else (nom, 0)
        debut += raw.n_times if nom in blocs else 0
    return essais, parties


# =====================================================================================================
# 3. Mauvaises électrodes et périodes suspectes
# =====================================================================================================
def mad(x, axis=-1):
    """Amplitude robuste : 1,4826 × l'écart absolu médian (égale l'écart-type pour un bruit gaussien)."""
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)


def amplitudes(raw, electrodes_reference=None):
    """Écart-type et amplitude robuste (µV) de chaque électrode EEG sauf Pz, sur une copie filtrée de 1 à 40 Hz.
    electrodes_reference=None : le signal tel qu'enregistré (chaque électrode moins Pz). Sinon : moins la
    référence médiane temporaire de ces électrodes ; les autres électrodes (les plates) sont laissées de côté."""
    eeg = raw.copy().pick("eeg").filter(1, 40, verbose=False)
    x = eeg.get_data() * 1e6
    if electrodes_reference is not None:
        lignes = [eeg.ch_names.index(c) for c in electrodes_reference]
        y = np.full_like(x, np.nan)
        y[lignes] = x[lignes] - np.median(x[lignes], axis=0)
        x = y
    return pd.DataFrame({"et": x.std(axis=1), "robuste": mad(x)}, index=eeg.ch_names).drop(index="Pz")


def electrodes_plates(coupes):
    """Les électrodes plates sur le signal enregistré, dans au moins une coupe. Rend (plates, rapport, et)."""
    amp = {nom: amplitudes(raw) for nom, raw in coupes.items()}
    rapport = pd.DataFrame({nom: a["robuste"] for nom, a in amp.items()})
    rapport = rapport / rapport.median()
    et = pd.DataFrame({nom: a["et"] for nom, a in amp.items()})
    plate = ((rapport < RAPPORT_PLAT) | (et < ET_PLAT_UV)).any(axis=1)
    return sorted(plate.index[plate]), rapport, et


def qualite_fenetres(donnees, coupes, parties, montrees, reference):
    """Le contrôle de 5 s. Chaque partie (repos, bloc, marche) est coupée en fenêtres de FENETRE_QC_S. Dans chaque
    fenêtre : l'amplitude robuste de chaque électrode de `montrees` après la référence médiane temporaire des
    électrodes `reference` (1-40 Hz), divisée par l'amplitude typique de la fenêtre (la médiane sur `reference`) ;
    et dans la ligne 'tout le scalp', l'amplitude typique de la fenêtre divisée par la fenêtre typique de sa partie.
    Rend ce tableau (une colonne par fenêtre) et les fenêtres (partie, enregistrement, debut en s)."""
    import mne

    lignes = [montrees.index(c) for c in reference]
    colonnes, fenetres = [], []
    for partie, (enregistrement, premier) in parties.items():
        raw = donnees[enregistrement]
        x = raw.get_data(picks=montrees, start=premier, stop=premier + coupes[partie].n_times) * 1e6
        x = mne.filter.filter_data(x, raw.info["sfreq"], 1, 40, verbose=False)
        x = x - np.median(x[lignes], axis=0)
        n = int(FENETRE_QC_S * raw.info["sfreq"])
        amp = np.array([mad(p) for p in (x[:, k:k + n] for k in range(0, x.shape[1] - n + 1, n))]).T
        typique = np.median(amp[lignes], axis=0)
        colonnes.append(np.vstack([typique / np.median(typique), amp / typique]))
        fenetres += [(partie, enregistrement, (premier + k * n) / raw.info["sfreq"]) for k in range(amp.shape[1])]
    return (pd.DataFrame(np.hstack(colonnes), index=["tout le scalp"] + list(montrees)),
            pd.DataFrame(fenetres, columns=["partie", "enregistrement", "debut"]))


def mauvaises_electrodes(posture, coupes, enregistrements, parties):
    """Les mauvaises électrodes d'une posture. Rend un dictionnaire : mauvaises, plates, bruitees, utilisables (la
    référence médiane), qualite et fenetres (le contrôle de 5 s), et les mesures de chaque règle."""
    plates, rapport_pz, et = electrodes_plates(coupes)
    eeg = coupes[next(iter(coupes))].copy().pick("eeg").ch_names
    if posture == "assis":
        # Bruitée : au-dessus de RAPPORT_BRUIT × l'amplitude typique, dans au moins un enregistrement
        utilisables = [c for c in eeg if c != "Pz" and c not in plates]
        amp = {nom: amplitudes(raw, utilisables) for nom, raw in coupes.items()}
        rapport = pd.DataFrame({nom: a["robuste"] for nom, a in amp.items()})
        typique = rapport.median()
        rapport = rapport / typique
        bruitee = (rapport > RAPPORT_BRUIT).any(axis=1) & ~rapport.index.isin(plates)
        bruitees = list(rapport.index[bruitee])
        # Pz est jugée aussi : par rapport à la référence médiane, son signal est celui du capteur de référence
        ref = {nom: np.median(raw.copy().pick(utilisables).filter(1, 40, verbose=False).get_data() * 1e6, axis=0)
               for nom, raw in coupes.items()}
        rapport_pz_ref = pd.Series({nom: mad(m) for nom, m in ref.items()}) / typique
        if (rapport_pz_ref > RAPPORT_BRUIT).any():
            bruitees.append("Pz")
        mauvaises = sorted(plates + bruitees)
        utilisables = [c for c in utilisables if c not in mauvaises]
        qualite, fenetres = qualite_fenetres(enregistrements, coupes, parties, utilisables, utilisables)
        mesures = dict(rapport=rapport, rapport_pz_reference=rapport_pz_ref)
    else:
        # Bruitée : au-dessus de RAPPORT_BRUIT × l'amplitude typique de sa fenêtre de 5 s, dans plus de PART_BRUIT de
        # ses fenêtres ; deux fois, la deuxième sans les bruitées de la première dans la référence médiane
        montrees = [c for c in eeg if c not in plates]
        bruitees = []
        for _ in range(2):
            utilisables = [c for c in eeg if c != "Pz" and c not in plates + bruitees]
            qualite, fenetres = qualite_fenetres(enregistrements, coupes, parties, montrees, utilisables)
            part = (qualite.drop(index="tout le scalp") > RAPPORT_BRUIT).mean(axis=1)
            bruitees = sorted(set(bruitees) | set(part.index[part > PART_BRUIT]))
        mauvaises = sorted(plates + bruitees)
        utilisables = [c for c in utilisables if c not in mauvaises]
        mesures = dict(part=part)
    return dict(mauvaises=mauvaises, plates=plates, bruitees=sorted(bruitees), utilisables=utilisables,
                qualite=qualite, fenetres=fenetres, rapport_pz=rapport_pz, et=et, **mesures)


def periodes_qc(qualite, fenetres, mauvaises):
    """Les périodes suspectes : une électrode au-dessus de RAPPORT_QC × l'amplitude typique de sa fenêtre
    (QC_grande_amplitude) ou sous RAPPORT_PLAT (QC_contact), tout le scalp au-dessus de RAPPORT_QC
    (QC_grande_amplitude sans électrode). Les fenêtres consécutives d'un même problème forment une période."""
    trouvees = []
    for k, f in fenetres.iterrows():
        colonne = qualite.iloc[:, k]
        if colonne["tout le scalp"] > RAPPORT_QC:
            trouvees.append((f.enregistrement, f.partie, f.debut, "QC_grande_amplitude", "toutes", colonne["tout le scalp"]))
        for c in [c for c in qualite.index[1:] if c not in mauvaises]:
            if colonne[c] > RAPPORT_QC:
                trouvees.append((f.enregistrement, f.partie, f.debut, "QC_grande_amplitude", c, colonne[c]))
            elif colonne[c] < RAPPORT_PLAT:
                trouvees.append((f.enregistrement, f.partie, f.debut, "QC_contact", c, colonne[c]))
    periodes = []
    for enregistrement, partie, debut, description, electrodes, valeur in sorted(trouvees, key=lambda t: (t[0], t[4], t[3], t[2])):
        der = periodes[-1] if periodes else None
        if (der and (der["enregistrement"], der["electrodes"], der["description"]) == (enregistrement, electrodes, description)
                and abs(der["debut"] + der["duree"] - debut) < 1e-6):
            der["duree"] += FENETRE_QC_S
            der["rapport"] = min(der["rapport"], valeur) if description == "QC_contact" else max(der["rapport"], valeur)
        else:
            periodes.append(dict(enregistrement=enregistrement, partie=partie, debut=debut, duree=FENETRE_QC_S,
                                 description=description, electrodes=electrodes, rapport=valeur))
    return pd.DataFrame(periodes, columns=["enregistrement", "partie", "debut", "duree", "description", "electrodes", "rapport"])


def ajouter_annotations(donnees, tableau, prefixe=None, par_electrode=True):
    """Ajoute une annotation par ligne de `tableau` (debut en s depuis le début de l'enregistrement). Avec `prefixe`,
    la description devient prefixe + description sans 'QC_' ; par_electrode=False : sans ch_names."""
    for p in tableau.itertuples():
        raw = donnees[p.enregistrement]
        description = p.description if prefixe is None else prefixe + p.description.removeprefix("QC_")
        noms = [[p.electrodes]] if par_electrode and p.electrodes != "toutes" else None
        raw.annotations.append(onset=raw.first_time + p.debut, duration=p.duree, description=description, ch_names=noms)


# =====================================================================================================
# 5. Auto-CCA (assis)
# =====================================================================================================
def auto_cca(raw):
    """BSS-CCA (De Clercq et al., 2006) avec la règle de la pente (Janani et al., 2018), par fenêtres de 10 s qui se
    chevauchent de moitié (mne-denoise). Pz (encore toute à zéro) et les mauvaises électrodes restent hors de
    l'ajustement. Rend une copie nettoyée et l'estimateur (pentes, composantes gardées)."""
    from mne_denoise.bss_cca import BSSCCA

    copie = raw.copy()
    mauvaises = list(copie.info["bads"])
    copie.info["bads"] = sorted(set(mauvaises) | {"Pz"})       # hors de l'ajustement, inchangées
    cca = BSSCCA(lag_samples=1, slope_threshold=CCA_PENTE, slope_band=CCA_BANDE_PENTE,
                 segment_len=CCA_FENETRE_S, overlap=0.5)
    propre = cca.fit_transform(copie)
    propre.info["bads"] = mauvaises
    return propre, cca


# =====================================================================================================
# 6. Clignements (les deux postures) ; accéléromètre (en marche)
# =====================================================================================================
def trouver_clignements(raw, bonnes):
    """Un événement par clignement, à son pic (échantillon, 0, 998). Fp1 et Fp2 moins la médiane des autres bonnes
    électrodes (1-10 Hz) ; à chaque échantillon, la plus petite des deux ; un clignement est un pic au-dessus de la
    médiane de plus de Z_CLIGNEMENT écarts-types robustes, à au moins 0,5 s du précédent, hors des annotations BAD."""
    from scipy.signal import find_peaks

    eeg = raw.copy().pick(bonnes).filter(1, 10, verbose=False)
    x = eeg.get_data() * 1e6
    yeux = [eeg.ch_names.index(c) for c in ["Fp1", "Fp2"] if c in eeg.ch_names]
    autres = [k for k, c in enumerate(eeg.ch_names) if c != "Pz"]
    deux = (x[yeux] - np.median(x[autres], axis=0)).min(axis=0)
    mauvais = np.isnan(eeg.get_data(picks=[0], reject_by_annotation="NaN")[0])
    niveau = np.median(deux[~mauvais])
    pics, _ = find_peaks(np.where(mauvais, niveau, deux), height=niveau + Z_CLIGNEMENT * mad(deux[~mauvais]),
                         distance=int(0.5 * raw.info["sfreq"]))
    return np.column_stack([pics + raw.first_samp, np.zeros(len(pics), int), np.full(len(pics), 998)])


def acceleration_tete(participant, posture, nom, raws, coupes, dossier=DOSSIER):
    """L'accélération de la tête pendant l'enregistrement `nom`, sur les échantillons de coupes[nom] : les trois axes
    et la norme (m/s²), filtrés BANDE_ACCEL ; NaN où l'accéléromètre n'a pas d'échantillon."""
    import mne

    eeg = dossier_eeg(participant, dossier)
    debuts = pd.read_csv(eeg.parent / f"sub-{participant}_ses-001_scans.tsv", sep="\t")
    debuts = dict(zip(debuts["filename"], pd.to_datetime(debuts["acq_time"])))
    vhdr = fichier_bids(participant, posture, nom, "eeg.vhdr", dossier)
    fichier = eeg.parent / "motion" / vhdr.name.replace("_acq-", "_tracksys-DSI24_acq-").replace("_eeg.vhdr", "_motion.tsv")
    mouvement = pd.read_csv(fichier, sep="\t", header=None, names=["x", "y", "z", "temps"])
    delai = (debuts[f"motion/{fichier.name}"] - debuts[f"eeg/{vhdr.name}"]).total_seconds()
    temps = delai + mouvement["temps"].to_numpy() - (coupes[nom].first_time - raws[nom].first_time)
    acc = mouvement[["x", "y", "z"]].to_numpy().T
    acc = np.vstack([acc, np.linalg.norm(acc, axis=0)])
    acc = mne.filter.filter_data(acc - acc.mean(axis=1, keepdims=True), 30.0, *BANDE_ACCEL, verbose=False)
    return np.array([np.interp(coupes[nom].times, temps, a, left=np.nan, right=np.nan) for a in acc])


def trouver_pas(acc, sfreq):
    """Un pic de l'accélération verticale (z) par pas."""
    from scipy.signal import find_peaks

    return find_peaks(np.nan_to_num(acc[2]), distance=int(0.35 * sfreq), prominence=0.5 * np.nanstd(acc[2]))[0]


def avec_delais(acc, delais):
    """L'accélération et ses copies retardées : une ligne par signal et par délai, NaN là où la copie n'existe pas."""
    lignes = []
    for d in delais:
        copie = np.full_like(acc, np.nan)
        copie[:, d:] = acc[:, :acc.shape[1] - d]
        lignes.append(copie)
    return np.vstack(lignes)


def prediction(X, Y, ajuste, penalite, sfreq):
    """Ce que X prédit de chaque ligne de Y : une régression ridge ajustée sur les échantillons `ajuste` de chaque
    fenêtre de FENETRE_ACCEL_S (chevauchement de moitié), les fenêtres fondues avec une fenêtre de Hann."""
    n, fen = Y.shape[1], int(FENETRE_ACCEL_S * sfreq)
    total, poids = np.zeros_like(Y), np.zeros(n)
    for debut in sorted(set(range(0, max(n - fen, 0) + 1, fen // 2)) | {max(n - fen, 0)}):
        x, y, u = X[:, debut:debut + fen], Y[:, debut:debut + fen], ajuste[debut:debut + fen]
        if u.sum() < 10 * len(x):
            continue
        mx, my = x[:, u].mean(axis=1, keepdims=True), y[:, u].mean(axis=1, keepdims=True)
        gram = (x[:, u] - mx) @ (x[:, u] - mx).T
        w = np.linalg.solve(gram + penalite * np.trace(gram) / len(gram) * np.eye(len(gram)), (x[:, u] - mx) @ (y[:, u] - my).T)
        fenetre_hann = np.hanning(y.shape[1] + 2)[1:-1]
        total[:, debut:debut + fen] += fenetre_hann * (w.T @ np.nan_to_num(x - mx))
        poids[debut:debut + fen] += fenetre_hann
    return total / np.maximum(poids, 1e-12)


def regression_mouvement(analyse, acceleration, coupes, parties, bonnes_toutes):
    """Retire de chaque bonne électrode ce que l'accélération de la tête en prédit, partie par partie, si la
    régression explique au moins GAIN_MIN_ACCEL des données mises de côté. Modifie `analyse` ; rend le tableau."""
    sfreq = analyse["tache"].info["sfreq"]
    delais = np.round(DELAIS_ACCEL_S * sfreq).astype(int)
    lignes = []
    for partie, (enregistrement, premier) in parties.items():
        raw, n = analyse[enregistrement], coupes[partie].n_times
        X = avec_delais(acceleration[partie], delais)
        Y = raw.get_data(picks=bonnes_toutes, start=premier, stop=premier + n) * 1e6
        ok = (np.isfinite(X).all(axis=0)
              & ~np.isnan(raw.get_data(picks=bonnes_toutes[0], start=premier, stop=premier + n, reject_by_annotation="NaN")[0]))
        de_cote = (np.arange(n) // int(5 * sfreq)) % 4 == 0
        explique = {}
        for a in PENALITES_ACCEL:
            erreur = Y - prediction(X, Y, ok & ~de_cote, a, sfreq)
            explique[a] = 1 - erreur[:, ok & de_cote].var(axis=1).sum() / Y[:, ok & de_cote].var(axis=1).sum()
        a = max(explique, key=explique.get)
        applique = explique[a] >= GAIN_MIN_ACCEL
        if applique:
            correction = np.zeros((len(bonnes_toutes), raw.n_times))
            correction[:, premier:premier + n] = prediction(X, Y, ok, a, sfreq) * 1e-6
            raw.apply_function(lambda x: x - correction, picks=bonnes_toutes, channel_wise=False)
        lignes.append(dict(enregistrement=partie, penalite=a, variance_expliquee_pct=round(100 * explique[a], 1),
                           appliquee=applique))
    return pd.DataFrame(lignes).set_index("enregistrement")


# =====================================================================================================
# 7-8. ICA : copies, apprentissage, preuves, décision
# =====================================================================================================
def copies_ica(posture, analyse, tableau_qc):
    """La copie qui entraîne l'ICA (avec ses annotations BAD_ICA_) et la copie que lit ICLabel. Rend aussi la
    colonne « retiree de l'ICA » du tableau QC."""
    import mne

    if posture == "assis":
        copie = {n: r.copy().filter(*BANDE_ICA["assis"], picks="eeg", verbose=False) for n, r in analyse.items()}
        retiree = ((tableau_qc["description"] == "QC_contact") | (tableau_qc["electrodes"] == "toutes")
                   | (tableau_qc["rapport"] >= RAPPORT_HORS_ICA))
        iclabel = copie
    else:
        copie = {n: r.copy().filter(*BANDE_ICA["marche"], l_trans_bandwidth=TRANSITION_ICA_MARCHE, picks="eeg", verbose=False)
                 for n, r in analyse.items()}
        retiree = (tableau_qc["description"] == "QC_contact") | (tableau_qc["electrodes"] == "toutes")
        iclabel = {n: r.copy().filter(*BANDE_ICLABEL, picks="eeg", verbose=False) for n, r in analyse.items()}
    ajouter_annotations(copie, tableau_qc[retiree], prefixe="BAD_ICA_", par_electrode=False)
    entrainement = mne.concatenate_raws([r.copy() for r in copie.values()])
    entrainement_iclabel = (entrainement if posture == "assis"
                            else mne.concatenate_raws([r.copy() for r in iclabel.values()]))
    return entrainement, entrainement_iclabel, retiree


def ajuster_ica(posture, entrainement):
    """Picard (assis) ou AMICA (en marche), avec autant de composantes que le rang des bonnes électrodes."""
    import mne

    n = mne.compute_rank(entrainement, tol=1e-6, tol_kind="relative", verbose=False)["eeg"]
    if posture == "assis":
        ica = mne.preprocessing.ICA(n_components=n, method="picard", fit_params=dict(ortho=False, extended=True),
                                    random_state=GRAINE_ICA, max_iter=2000)
        ica.fit(entrainement, reject_by_annotation=True, verbose=False)
    else:
        from jamica import fit_ica

        morceaux = mne.make_fixed_length_epochs(entrainement, duration=2.0, preload=True, verbose=False)
        ica = fit_ica(morceaux, n_components=n, max_iter=AMICA_ITERATIONS, num_mix=3, decim=AMICA_DECIM,
                      random_state=GRAINE_ICA, fit_params=AMICA_REJET)
    return ica


CLASSES = ["brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other"]


def etiquettes_iclabel(entrainement_iclabel, ica):
    from mne_icalabel.iclabel import iclabel_label_components

    p = np.array(iclabel_label_components(entrainement_iclabel, ica, inplace=False))
    return pd.DataFrame(p, columns=CLASSES, index=[f"ICA{i:03d}" for i in range(ica.n_components_)])


def correlations_yeux(ica, entrainement_iclabel, mauvaises, index):
    """La corrélation (1-10 Hz) de chaque composante avec Fp1, Fp2 et HEOG = F7 − F8 (des substituts de l'EOG)."""
    import mne

    heog = not {"F7", "F8"} & set(mauvaises)
    oculaire = entrainement_iclabel.copy()
    if heog:
        oculaire = mne.set_bipolar_reference(oculaire, anode="F7", cathode="F8", ch_name="HEOG", drop_refs=False,
                                             verbose=False)
        oculaire.set_channel_types({"HEOG": "eog"})
    substituts = [c for c in ["Fp1", "Fp2"] if c not in mauvaises] + ["HEOG"] * heog
    return pd.DataFrame({s: ica.find_bads_eog(oculaire, ch_name=s, verbose=False)[1] for s in substituts}, index=index)


def forme_oeil(ica, k):
    """'clignement' si Fp1 et Fp2 portent toutes deux la composante k avec le même signe ; 'horizontal' si la gauche
    et la droite du front la portent avec des signes opposés. Seulement si son plus grand poids est à Fp1, Fp2, F7, F8."""
    poids = ica.get_components() / np.abs(ica.get_components()).max(axis=0)
    noms = ica.ch_names
    w = {c: poids[noms.index(c), k] for c in ["Fp1", "Fp2", "F7", "F8"] if c in noms}
    grands = {c for c, v in w.items() if abs(v) >= 0.5}
    if noms[np.abs(poids[:, k]).argmax()] not in w:
        return ""
    if {"Fp1", "Fp2"} <= grands and w["Fp1"] * w["Fp2"] > 0:
        return "clignement"
    if any({a, b} <= grands and w[a] * w[b] < 0 for a, b in [("Fp1", "Fp2"), ("F7", "F8")]):
        return "horizontal"
    return ""


def decision_assis(ica, etiquettes, corr, entrainement):
    """Assis : une composante est retirée quand deux preuves du même genre s'accordent ; revue (gardée) si une seule
    preuve, ou si ICLabel lui donne au moins PRUDENCE_CERVEAU de « cerveau »."""
    score = pd.Series(ica.find_bads_muscle(entrainement, verbose=False)[1], index=etiquettes.index)
    preuves = pd.DataFrame({"ICLabel_cerveau": etiquettes["brain"], "ICLabel_muscle": etiquettes["muscle artifact"],
                            "ICLabel_oeil": etiquettes["eye blink"]}).join(corr.add_prefix("corr_"))
    preuves["score_muscle_mne"] = score
    preuves["forme_oeil"] = [forme_oeil(ica, k) for k in range(ica.n_components_)]
    votes = {"oeil": {"ICLabel oeil": preuves["ICLabel_oeil"] >= P_ICLABEL,
                      "suit un substitut de l'œil": corr.abs().max(axis=1) >= CORR_OEIL_ASSIS,
                      "topographie de l'œil": preuves["forme_oeil"] != ""},
             "muscle": {"ICLabel muscle": preuves["ICLabel_muscle"] >= P_ICLABEL,
                        "score muscle MNE": preuves["score_muscle_mne"] >= SCORE_MUSCLE}}

    def decider(k):
        trouves = {g: [nom for nom, v in vs.items() if v.iloc[k]] for g, vs in votes.items()}
        genre = "oeil" if len(trouves["oeil"]) >= len(trouves["muscle"]) else "muscle"
        if not trouves[genre]:
            return etiquettes.columns[etiquettes.iloc[k].argmax()], "garder", "aucune preuve d'artefact"
        if genre == "oeil":
            r = corr.iloc[k].abs()
            genre += " (" + (preuves["forme_oeil"].iloc[k]
                             or ("horizontal" if r.get("HEOG", 0) > r.drop("HEOG", errors="ignore").max() else "clignement")) + ")"
        raison = ", ".join(trouves["oeil"] + trouves["muscle"])
        cerveau = preuves["ICLabel_cerveau"].iloc[k]
        if cerveau >= PRUDENCE_CERVEAU:
            return genre, "revoir", raison + f" (mais ICLabel cerveau {cerveau:.2f})"
        if len(trouves[genre.split()[0]]) < 2:
            return genre, "revoir", raison + " (une seule preuve)"
        return genre, "retirer", raison

    return preuves.join(pd.DataFrame([decider(k) for k in range(ica.n_components_)], index=preuves.index,
                                     columns=["classe", "decision", "raison"]))


def part_de_variance(ica, tache_30, bonnes_toutes, composantes):
    """La part (%) de la variance de la tâche (0,1-30 Hz, hors BAD) que portent ces composantes."""
    total = tache_30.get_data(picks=bonnes_toutes, reject_by_annotation="omit").var(axis=1).sum()
    refait = ica.apply(tache_30.copy(), include=list(composantes), exclude=[], verbose=False)
    return 100 * refait.get_data(picks=bonnes_toutes, reject_by_annotation="omit").var(axis=1).sum() / total


def decision_marche(ica, etiquettes, corr, tache, bonnes_toutes):
    """En marche : la règle des analyses Brain Walk (œil, muscle, bruit d'une électrode)."""
    topo = ica.get_components()
    focal = pd.Series((topo ** 2).max(axis=0) / (topo ** 2).sum(axis=0), index=etiquettes.index)
    sommet = pd.Series(np.array(ica.ch_names)[np.abs(topo).argmax(axis=0)], index=etiquettes.index)
    tache_30 = tache.copy().filter(None, 30, verbose=False)
    variance = pd.Series([part_de_variance(ica, tache_30, bonnes_toutes, [k]) for k in range(ica.n_components_)],
                         index=etiquettes.index)
    preuves = pd.DataFrame({"ICLabel_cerveau": etiquettes["brain"], "ICLabel_oeil": etiquettes["eye blink"],
                            "ICLabel_muscle": etiquettes["muscle artifact"],
                            "ICLabel_electrode": etiquettes["channel noise"]}).join(corr.add_prefix("corr_"))
    preuves["focalite"] = focal
    preuves["electrode_max"] = sommet
    preuves["variance_pct"] = variance
    oeil_premier = etiquettes.idxmax(axis=1) == "eye blink"
    une_electrode = (focal >= FOCAL) & (variance >= VARIANCE_ELECTRODE)
    regles = {"oeil": (preuves["ICLabel_oeil"] >= P_OEIL) | (oeil_premier & (corr.abs().max(axis=1) >= CORR_OEIL_MARCHE)),
              "muscle": preuves["ICLabel_muscle"] >= P_MUSCLE,
              "electrode": une_electrode | (preuves["ICLabel_electrode"] >= 0.5)}

    def decider(k):
        e = preuves.iloc[k]
        if regles["oeil"].iloc[k] and not (une_electrode.iloc[k] and e.ICLabel_oeil < P_OEIL):
            r = corr.iloc[k].abs()
            genre = "oeil (horizontal)" if r.get("HEOG", 0) > r.drop("HEOG", errors="ignore").max() else "oeil (clignement)"
            if e.ICLabel_oeil >= P_OEIL:
                return genre, "retirer", f"ICLabel oeil {e.ICLabel_oeil:.2f}"
            return genre, "retirer", f"oeil premier choix d'ICLabel ({e.ICLabel_oeil:.2f}) et |r| = {r.max():.2f} avec {r.idxmax()}"
        if regles["muscle"].iloc[k]:
            return "muscle", "retirer", f"ICLabel muscle {e.ICLabel_muscle:.2f}"
        if regles["electrode"].iloc[k]:
            if une_electrode.iloc[k]:
                return "bruit d'électrode", "retirer", f"{100 * e.focalite:.0f} % de sa carte sur {e.electrode_max}, {e.variance_pct:.0f} % de la variance"
            return "bruit d'électrode", "retirer", f"ICLabel bruit d'électrode {e.ICLabel_electrode:.2f}"
        une = f", une électrode ({e.electrode_max})" if e.focalite >= FOCAL else ""
        return etiquettes.columns[etiquettes.iloc[k].argmax()] + une, "garder", ""

    return preuves.join(pd.DataFrame([decider(k) for k in range(ica.n_components_)], index=preuves.index,
                                     columns=["classe", "decision", "raison"]))


# =====================================================================================================
# 9. Gabarit de foulée (en marche) ; contrôle après l'ICA
# =====================================================================================================
def artefact_foulee(x, foulees, demi):
    """Ce que chaque foulée de x (électrodes × échantillons) partage avec les FOULEES_AUTOUR foulées de chaque côté :
    leur moyenne, chaque foulée étirée en deux moitiés de `demi` échantillons, ajustée à la foulée par moindres carrés
    dans chaque électrode, et ramenée à sa durée. Zéro hors des foulées."""
    grille = np.linspace(0, 1, demi, endpoint=False)

    def etirer(a, b, c):
        return np.hstack([[np.interp(grille * (f - d), np.arange(f - d), e) for e in x[:, d:f]] for d, f in ((a, b), (b, c))])

    etirees = np.stack([etirer(*f) for f in foulees])
    artefact = np.zeros_like(x)
    for i, (a, b, c) in enumerate(foulees):
        autour = np.r_[max(0, i - FOULEES_AUTOUR):i, i + 1:min(len(foulees), i + 1 + FOULEES_AUTOUR)]
        gabarit = etirees[autour].mean(axis=0)
        gabarit -= gabarit.mean(axis=1, keepdims=True)
        celle_ci = etirees[i] - etirees[i].mean(axis=1, keepdims=True)
        ajuste = gabarit * ((celle_ci * gabarit).sum(axis=1) / (gabarit ** 2).sum(axis=1))[:, None]
        artefact[:, a:c] = np.hstack([[np.interp(np.arange(f - d) / (f - d), grille, e) for e in ajuste[:, k * demi:(k + 1) * demi]]
                                      for k, (d, f) in enumerate(((a, b), (b, c)))])
    return artefact


def retirer_foulees(nettoye, pas, parties, bonnes_toutes):
    """Soustrait de chaque bonne électrode ce qui se répète à chaque foulée (Gwin et al., 2010), appris au-dessus de
    PASSE_HAUT_FOULEE. Modifie `nettoye` ; rend le nombre de foulées."""
    appris = {n: r.copy().filter(PASSE_HAUT_FOULEE, None, picks=bonnes_toutes, verbose=False) for n, r in nettoye.items()}
    total = 0
    for partie, (enregistrement, premier) in parties.items():
        raw = nettoye[enregistrement]
        dans_bad = np.isnan(raw.get_data(picks=bonnes_toutes[0], reject_by_annotation="NaN")[0])
        s = np.sort(pas[partie]) + premier
        foulees = [(s[i], s[i + 1], s[i + 2]) for i in range(0, len(s) - 2, 2) if not dans_bad[s[i]:s[i + 2]].any()]
        if not foulees:
            continue
        artefact = artefact_foulee(appris[enregistrement].get_data(picks=bonnes_toutes), foulees, int(np.median(np.diff(s))))
        raw.apply_function(lambda x: x - artefact, picks=bonnes_toutes, channel_wise=False)
        total += len(foulees)
    return total


def bad_final(tableau_qc, qualite_apres, fenetres, final):
    """Compare chaque période QC avant et après l'ICA ; une période de tout le scalp encore au-delà du seuil devient
    BAD_final_. Modifie `final` et le tableau QC."""
    def apres(p):
        dedans = ((fenetres["enregistrement"] == p.enregistrement) & (fenetres["debut"] >= p.debut - 1e-6)
                  & (fenetres["debut"] < p.debut + p.duree - 1e-6)).to_numpy()
        v = qualite_apres.loc["tout le scalp" if p.electrodes == "toutes" else p.electrodes, dedans]
        return v.min() if p.description == "QC_contact" else v.max()

    tableau_qc["rapport_apres_ica"] = [apres(p) for p in tableau_qc.itertuples()]
    encore = np.where(tableau_qc["description"] == "QC_contact", tableau_qc["rapport_apres_ica"] < RAPPORT_PLAT,
                      tableau_qc["rapport_apres_ica"] > RAPPORT_QC)
    tableau_qc["apres_ica"] = np.where(encore, "encore au-delà du seuil", "sous le seuil, toujours suspecte")
    final_bad = encore & (tableau_qc["electrodes"] == "toutes").to_numpy()
    tableau_qc["final"] = np.where(final_bad, "BAD_final", "QC seulement")
    ajouter_annotations(final, tableau_qc[final_bad], prefixe="BAD_final_", par_electrode=False)
    return tableau_qc


# =====================================================================================================
# 10. Époques, contrôle local, ERP
# =====================================================================================================
def reference_par_epoque(donnees, garde):
    """Chaque époque moins la moyenne des électrodes qu'elle garde (garde : époques × électrodes)."""
    return donnees - (donnees * garde[:, :, None]).sum(axis=1, keepdims=True) / np.maximum(garde.sum(axis=1), 1)[:, None, None]


def epoques_mots(final_tache, enregistrement_tache, essais, mauvaises):
    """Une époque par mot, avant et après le nettoyage (1-40 Hz, ligne de base), les mêmes époques dans les deux :
    hors des BAD, réponse correcte, mot confirmé par son bip."""
    import mne

    avant = enregistrement_tache.copy().filter(*BANDE_ERP, picks="eeg", verbose=False).set_eeg_reference("average", verbose=False)
    apres = final_tache.copy().filter(*BANDE_ERP, picks="eeg", verbose=False)
    evenements, _ = mne.events_from_annotations(apres, event_id=EVENT_ID, verbose=False)
    assert (evenements[:, 2] == essais["condition"].map(EVENT_ID).to_numpy()).all()
    meta = essais[["trial", "block", "trial_in_block", "stimulus", "condition", "accuracy", "response_time",
                   "marker_residual_ms"]].reset_index(drop=True)
    epochs = {e: mne.Epochs(d, evenements, EVENT_ID, tmin=TMIN, tmax=TMAX, baseline=LIGNE_DE_BASE, metadata=meta,
                            preload=True, verbose=False)
              for e, d in [("avant nettoyage", avant), ("après nettoyage", apres)]}
    communes = np.intersect1d(epochs["avant nettoyage"].selection, epochs["après nettoyage"].selection)
    epochs = {e: ep[np.isin(ep.selection, communes)] for e, ep in epochs.items()}
    md = epochs["après nettoyage"].metadata
    garder = ((md["marker_residual_ms"].abs() <= BIP_MAX_MS) & (md["accuracy"] == 1)).to_numpy()
    return {e: ep[garder] for e, ep in epochs.items()}


def controle_local(propres, mauvaises, tableau_qc, debut_tache_s):
    """Le contrôle local : une électrode est laissée de côté dans une époque si elle est trop grande, dans une période
    QC encore au-delà du seuil, ou si elle dérive ; chaque époque est re-référencée à ce qu'elle garde ; une époque qui
    perd plus de la moitié de ses bonnes électrodes est abandonnée ; une électrode dont la SME de la N400 est trop
    grande est laissée de côté pour la posture, avec sa partenaire. Rend un dictionnaire de résultats ; ne modifie
    pas `propres`."""
    import mne

    scalp = mne.pick_types(propres.info, eeg=True, exclude=[])
    noms = np.array(propres.ch_names)[scalp]
    jugees = ~np.isin(noms, mauvaises)
    x = propres.get_data(picks=scalp, units="uV")
    # règle 1 : trop grande
    log_cc = np.log(np.ptp(x, axis=-1))
    milieu = np.median(log_cc, axis=0)
    limite = np.exp(milieu + ECART_QC * 1.4826 * np.median(np.abs(log_cc - milieu), axis=0))
    trop_grande = (np.ptp(x, axis=-1) > limite) & jugees
    # règle 2 : une période QC de cette électrode encore au-delà du seuil après l'ICA
    debuts = (propres.events[:, 0] / propres.info["sfreq"]) - debut_tache_s
    dans_qc = np.zeros_like(trop_grande)
    for p in tableau_qc[(tableau_qc["enregistrement"] == "tache") & (tableau_qc["apres_ica"] == "encore au-delà du seuil")].itertuples():
        chevauche = (debuts + TMIN < p.debut + p.duree) & (debuts + TMAX > p.debut)
        dans_qc[chevauche] |= jugees & ((noms == p.electrodes) | (p.electrodes == "toutes"))
    # règle 3 : une dérive
    pente = np.polyfit(propres.times, x.reshape(-1, len(propres.times)).T, 1)[0].reshape(x.shape[:2])
    derive = (np.abs(pente) * (propres.times[-1] - propres.times[0]) > DERIVE_UV) & jugees
    laissee = trop_grande | dans_qc | derive
    encore = (np.ptp(reference_par_epoque(x, jugees & ~laissee), axis=-1) > limite) & jugees & ~laissee
    laissee |= encore
    perdue = laissee[:, jugees].mean(axis=1) > MAX_LAISSEES
    # la SME de la N400 de chaque électrode
    y = reference_par_epoque(x, jugees & ~laissee)
    n400 = (propres.times >= COMPOSANTES["N400"]["fenetre"][0] - 1e-9) & (propres.times <= COMPOSANTES["N400"]["fenetre"][1] + 1e-9)
    sme = np.full(len(noms), np.nan)
    for k in np.flatnonzero(jugees):
        gardees = ~perdue & ~laissee[:, k]
        if gardees.sum() >= 10:
            v = pd.Series(y[gardees, k][:, n400].mean(axis=1))
            v -= v.groupby(propres.events[gardees, 2]).transform("mean")
            sme[k] = v.std() / np.sqrt(len(v))
    limite_sme = FACTEUR_SME * np.nanmedian(sme[jugees])
    trop_bruitees = [str(c) for c in noms[sme > limite_sme]]
    trop_bruitees += [PARTENAIRES[c] for c in trop_bruitees if c in PARTENAIRES and PARTENAIRES[c] not in trop_bruitees + mauvaises]
    regle = np.select([dans_qc, trop_grande, derive, encore], [1, 2, 3, 4], 0)
    return dict(noms=noms, jugees=jugees, laissee=laissee, perdue=perdue, limite=limite, sme=sme, limite_sme=limite_sme,
                trop_bruitees=trop_bruitees, regle=regle)


def appliquer_controle_local(epochs, cl, mauvaises):
    """Applique le contrôle local aux époques après nettoyage (référence par époque, colonne `laissees`, info['bads'])
    et abandonne les mêmes époques dans les deux versions. Modifie `epochs`."""
    import mne

    propres = epochs["après nettoyage"]
    scalp = mne.pick_types(propres.info, eeg=True, exclude=[])
    garde = cl["jugees"] & ~cl["laissee"]
    propres.apply_function(lambda d: reference_par_epoque(d, garde), picks=scalp, channel_wise=False)
    propres.metadata = propres.metadata.assign(laissees=[" ".join(cl["noms"][r]) for r in cl["laissee"]])
    for ep in epochs.values():
        ep.drop(cl["perdue"], reason="contrôle local", verbose=False)
    propres.info["bads"] = list(mauvaises) + cl["trop_bruitees"]


def electrodes_gardees(ep, electrodes):
    """Époques × électrodes : True là où l'époque garde l'électrode (pas dans sa colonne `laissees`)."""
    laissees = ep.metadata["laissees"] if "laissees" in ep.metadata else pd.Series("", index=ep.metadata.index)
    return np.array([[c not in str(m).split() for c in electrodes] for m in laissees])


def moyenne_gardee(ep):
    """L'ERP des époques ep, chaque électrode moyennée sur les époques qui la gardent."""
    import mne

    garde = electrodes_gardees(ep, ep.ch_names)
    return mne.EvokedArray((ep.get_data() * garde[:, :, None]).sum(axis=0) / np.maximum(garde.sum(axis=0), 1)[:, None],
                           ep.info, tmin=ep.tmin, nave=len(ep))


def mesures_erp(epochs, mauvaises_posture):
    """Amplitude moyenne et SME de chaque composante, condition et version (avant / après nettoyage)."""
    amplitude, sme = {}, {}
    for comp, c in COMPOSANTES.items():
        region = [e for e in c["region"] if e not in mauvaises_posture]
        for version, ep in epochs.items():
            garde = electrodes_gardees(ep, region)
            dans = ep.copy().pick(region).crop(*c["fenetre"]).get_data(units="uV").mean(axis=-1)
            v = np.where(garde.any(axis=1), (dans * garde).sum(axis=1) / np.maximum(garde.sum(axis=1), 1), np.nan)
            par = {n: v[(ep.metadata["condition"] == n).to_numpy() & ~np.isnan(v)] for n in CONDITIONS}
            amplitude[(comp, version)] = {n: x.mean() for n, x in par.items()}
            sme[(comp, version)] = {n: x.std(ddof=1) / np.sqrt(len(x)) for n, x in par.items()}
    return pd.DataFrame(amplitude), pd.DataFrame(sme)


# =====================================================================================================
# La chaîne complète
# =====================================================================================================
def pretraiter(participant, posture, dossier=DOSSIER, verbeux=True):
    """Toute la chaîne de prétraitement d'un participant dans une posture ('assis' ou 'marche').

    Rend un dictionnaire : les enregistrements nettoyés (final), les époques avant / après nettoyage, le contrôle
    local, les mauvaises électrodes, le tableau QC, les preuves de l'ICA, les clignements et un résumé."""
    import mne

    mne.set_log_level("ERROR")

    def dire(*args):
        if verbeux:
            print(f"[{participant} {posture}]", *args)

    raws, evenements = lire_enregistrements(participant, posture, dossier)
    coupes, enregistrements, essais, parties = assembler(raws, evenements)
    dire("enregistrements :", ", ".join(coupes))

    # 3. Mauvaises électrodes et périodes suspectes
    me = mauvaises_electrodes(posture, coupes, enregistrements, parties)
    mauvaises = me["mauvaises"]
    for raw in enregistrements.values():
        raw.info["bads"] = list(mauvaises)
    tableau_qc = periodes_qc(me["qualite"], me["fenetres"], mauvaises)
    ajouter_annotations(enregistrements, tableau_qc)
    dire("mauvaises électrodes :", mauvaises, "| périodes QC :", len(tableau_qc))

    # 4. Filtre ; 5. Auto-CCA (assis)
    filtre_pz = {n: r.copy().filter(*BANDE_ANALYSE, picks="eeg") for n, r in enregistrements.items()}
    tableaux_cca = {}
    if posture == "assis":
        analyse_pz = {}
        for n, r in filtre_pz.items():
            analyse_pz[n], cca = auto_cca(r)
            tableaux_cca[n] = cca
    else:
        analyse_pz = filtre_pz

    # 6. Référence moyenne ; clignements ; accéléromètre (en marche)
    analyse = {n: r.copy().set_eeg_reference("average") for n, r in analyse_pz.items()}
    bonnes_toutes = [c for c in analyse["tache"].copy().pick("eeg").ch_names if c not in mauvaises]
    clignements = trouver_clignements(analyse["tache"], bonnes_toutes)
    tableau_mouvement, pas = None, None
    if posture == "marche":
        acceleration = {n: acceleration_tete(participant, posture, n, raws, coupes, dossier) for n in coupes}
        pas = {n: trouver_pas(a, analyse["tache"].info["sfreq"]) for n, a in acceleration.items()}
        tableau_mouvement = regression_mouvement(analyse, acceleration, coupes, parties, bonnes_toutes)

    # 7. Copies pour l'ICA ; 8. ICA et décision
    entrainement, entrainement_iclabel, retiree = copies_ica(posture, analyse, tableau_qc)
    tableau_qc["retiree_de_l_ica"] = retiree
    ica = ajuster_ica(posture, entrainement)
    etiquettes = etiquettes_iclabel(entrainement_iclabel, ica)
    corr = correlations_yeux(ica, entrainement_iclabel, mauvaises, etiquettes.index)
    if posture == "assis":
        preuves = decision_assis(ica, etiquettes, corr, entrainement)
    else:
        preuves = decision_marche(ica, etiquettes, corr, analyse["tache"], bonnes_toutes)
    ica.exclude = [k for k in range(ica.n_components_) if preuves["decision"].iloc[k] == "retirer"]
    dire(f"ICA : {ica.n_components_} composantes, retirées :", [preuves.index[k] for k in ica.exclude])

    # 9. Appliquer l'ICA (puis le gabarit de foulée en marche) ; contrôle après l'ICA
    nettoye = {n: ica.apply(r.copy()) for n, r in analyse.items()}
    foulees = retirer_foulees(nettoye, pas, parties, bonnes_toutes) if posture == "marche" else 0
    montrees = list(me["qualite"].index[1:])
    qualite_apres, _ = qualite_fenetres(nettoye, coupes, parties, montrees, me["utilisables"])
    qualite_apres.loc[[c for c in mauvaises if c in qualite_apres.index]] = np.nan
    final = {n: r.copy() for n, r in nettoye.items()}
    tableau_qc = bad_final(tableau_qc, qualite_apres, me["fenetres"], final)

    # 10. Époques et contrôle local
    epochs = epoques_mots(final["tache"], enregistrements["tache"], essais, mauvaises)
    cl = controle_local(epochs["après nettoyage"], mauvaises, tableau_qc, final["tache"].first_time)
    appliquer_controle_local(epochs, cl, mauvaises)
    amplitude, sme = mesures_erp(epochs, mauvaises + cl["trop_bruitees"])
    dire("époques gardées :", {n: len(epochs["après nettoyage"][n]) for n in CONDITIONS},
         "| laissées de côté pour la posture :", cl["trop_bruitees"])

    resume = {"participant": participant, "posture": posture, "mauvaises électrodes": mauvaises,
              "périodes QC / hors ICA / BAD_final": f"{len(tableau_qc)} / {int(tableau_qc['retiree_de_l_ica'].sum())} / "
                                                     f"{int((tableau_qc['final'] == 'BAD_final').sum())}",
              "composantes ICA": int(ica.n_components_),
              "composantes retirées": [preuves.index[k] for k in ica.exclude],
              "électrodes laissées de côté pour la posture": cl["trop_bruitees"],
              "époques gardées (HF, LF, pseudo)": [len(epochs["après nettoyage"][n]) for n in CONDITIONS]}
    if posture == "marche":
        resume["régression du mouvement appliquée"] = tableau_mouvement["appliquee"].to_dict()
        resume["foulées"] = foulees
    return dict(participant=participant, posture=posture, raws=raws, coupes=coupes, enregistrements=enregistrements,
                essais=essais, parties=parties, mauvaises=mauvaises, tableau_qc=tableau_qc, final=final, ica=ica,
                preuves=preuves, clignements=clignements, epochs=epochs, controle_local=cl, amplitude=amplitude,
                sme=sme, tableau_mouvement=tableau_mouvement, resume=resume)


# =====================================================================================================
# Export : les fichiers de pretraite_00/ (voir tasks/brainwalk/LISEZMOI.md)
# =====================================================================================================
def _sans_clignement(r, debuts_mots_s, taps_s):
    """True si aucun pic de clignement entre 0,25 s avant le mot et 0,25 s après la tape."""
    pics = (r["clignements"][:, 0] - r["final"]["tache"].first_samp) / r["final"]["tache"].info["sfreq"]
    return np.array([not np.any((pics >= m - 0.25) & (pics <= t + 0.25)) for m, t in zip(debuts_mots_s, taps_s)])


def _metadonnees(r, essais_gardes, comportement):
    """Les métadonnées d'export d'une posture (une ligne par époque gardée)."""
    c = comportement[(comportement["participant"] == r["participant"]) & (comportement["posture"] == r["posture"])]
    c = c.set_index("essai")
    md = pd.DataFrame({
        "participant": r["participant"],
        "essai": essais_gardes["trial"].astype(int).to_numpy(),
        "bloc": essais_gardes["block"].astype(int).to_numpy(),
        "posture": r["posture"],
        "type": essais_gardes["condition"].map(TYPES).to_numpy(),
        "lexicalite": np.where(essais_gardes["condition"] == "pseudo", "pseudo-mot", "mot"),
        "stimulus": essais_gardes["stimulus"].to_numpy(),
    })
    md["tr_ms"] = c.loc[md["essai"], "tr_ms"].to_numpy()
    md["tr_valide"] = c.loc[md["essai"], "tr_valide"].to_numpy().astype(bool)
    return md


def _epoques_export(raw, debuts_echantillons, codes, tmin, tmax, md, garde_noms, decim=1):
    """Des époques sans ligne de base, chacune référencée à la moyenne des électrodes qu'elle garde. Rend les époques
    (celles qui touchent un BAD sont abandonnées) et le masque des lignes gardées."""
    import mne

    ev = np.column_stack([debuts_echantillons, np.zeros(len(codes), int), codes])
    ep = mne.Epochs(raw, ev, CODES_EXPORT, tmin=tmin, tmax=tmax, baseline=None, metadata=md.reset_index(drop=True),
                    preload=True, decim=decim, on_missing="ignore", event_repeated="drop", verbose=False)
    gardees = np.array([len(log) == 0 for log in ep.drop_log])
    return ep, gardees


def exporter(resultats, sortie, comportement=None, dossier=DOSSIER):
    """Écrit les fichiers de pretraite_00/ d'un participant à partir des résultats de pretraiter() pour ses postures :
    époques du mot, de la tape, longues (mot et tape) et enregistrements sans tâche. Rend le résumé de l'export."""
    import mne

    sortie = Path(sortie)
    if comportement is None:
        comportement = pd.read_csv(Path(dossier) / "comportement" / "essais.csv")
    participant = resultats[0]["participant"]
    bads_fichier = sorted({c for r in resultats for c in r["mauvaises"] + r["controle_local"]["trop_bruitees"]},
                          key=lambda c: resultats[0]["final"]["tache"].ch_names.index(c))
    sorties = {k: [] for k in ["mots", "tape", "longues", "longues_tape"]}
    sans_tache = []
    for r in resultats:
        posture, cl, tache = r["posture"], r["controle_local"], r["final"]["tache"]
        sf = tache.info["sfreq"]
        propres = r["epochs"]["après nettoyage"]
        noms = list(cl["noms"])
        scalp = np.arange(len(cl["noms"]))                          # après pick("eeg") : les 19 électrodes
        mauvaises_posture = r["mauvaises"] + cl["trop_bruitees"]
        # les essais gardés à l'étape 10, et leurs électrodes laissées de côté (dans l'ordre de cl["laissee"])
        sel = propres.selection                                     # indices des essais gardés
        laissee = cl["laissee"][~cl["perdue"]]                      # une ligne par époque gardée, dans le même ordre
        essais_gardes = r["essais"].iloc[sel].reset_index(drop=True)
        md = _metadonnees(r, essais_gardes, comportement)
        valides = md["tr_valide"].to_numpy()
        md, essais_gardes = md[valides].reset_index(drop=True), essais_gardes[valides].reset_index(drop=True)
        masques = laissee[valides]
        debuts = propres.events[valides, 0] if len(sel) else np.zeros(0, int)
        taps = debuts + np.round(md["tr_ms"].to_numpy() / 1000 * sf).astype(int)
        md["sans_clignement"] = _sans_clignement(r, (debuts - tache.first_samp) / sf, (taps - tache.first_samp) / sf)
        garde = r["controle_local"]["jugees"][None, :] & ~masques
        md["mauvaises"] = [" ".join(c for c, g in zip(noms, ligne) if (not g) or c in cl["trop_bruitees"]) for ligne in garde]
        md = md[["participant", "essai", "bloc", "posture", "type", "lexicalite", "tr_ms", "stimulus", "sans_clignement",
                 "mauvaises", "tr_valide"]]
        codes = md["type"].map(lambda t: CODES_EXPORT[f"{posture}/{t}"]).to_numpy()
        continu = tache.copy().pick("eeg").filter(*BANDE_ERP, verbose=False)   # les 19 électrodes du scalp

        def referencer(ep, g):
            ep.apply_function(lambda d: reference_par_epoque(d, g), picks=scalp, channel_wise=False)
            return ep

        # époques du mot : -0,2 à 1,0 s
        ep_mot, ok_mot = _epoques_export(continu, debuts, codes, -0.2, 1.0, md.drop(columns="tr_valide"), noms)
        g_mot = garde[ok_mot]
        referencer(ep_mot, g_mot)
        sorties["mots"].append(ep_mot)
        # époques de la tape : -0,8 à 0,2 s, seulement si le mot en a une et si ≤ 300 µV sur la région gardée
        md_mot = md[ok_mot].reset_index(drop=True)
        ep_tape, ok_tape = _epoques_export(continu, taps[ok_mot], codes[ok_mot], -0.8, 0.2, md_mot.drop(columns="tr_valide"), noms)
        g_tape = g_mot[ok_tape]
        referencer(ep_tape, g_tape)
        region = np.isin(noms, REGION_TAPE)
        cc = np.ptp(ep_tape.get_data(picks=scalp, units="uV"), axis=-1)
        trop = ((cc > MAX_TAPE_UV) & g_tape & region & ~np.isin(noms, cl["trop_bruitees"])).any(axis=1)   # électrodes gardées par l'essai
        ep_tape.drop(trop, reason="> 300 µV", verbose=False)
        sorties["tape"].append(ep_tape)
        # époques longues, 100 Hz : mot -1,0 à 1,5 s, tape -1,5 à 1,0 s ; règle 1 du contrôle local sur toute la fenêtre
        for cle, centres, (t0, t1) in (("longues", debuts[ok_mot], (-1.0, 1.5)), ("longues_tape", taps[ok_mot], (-1.5, 1.0))):
            ep_l, ok_l = _epoques_export(continu, centres, codes[ok_mot], t0, t1, md_mot.drop(columns="tr_valide"), noms, decim=3)
            g_l = g_mot[ok_l].copy()
            cc = np.ptp(ep_l.get_data(picks=scalp, units="uV"), axis=-1)
            log_cc = np.log(cc)                                     # la limite de chaque électrode, sur ces époques longues
            milieu = np.median(log_cc, axis=0)
            limite_l = np.exp(milieu + ECART_QC * 1.4826 * np.median(np.abs(log_cc - milieu), axis=0))
            g_l &= ~((cc > limite_l) & cl["jugees"])
            perdue = (~g_l[:, cl["jugees"]]).mean(axis=1) > MAX_LAISSEES
            referencer(ep_l, g_l)
            ep_l.metadata = ep_l.metadata.assign(mauvaises=[" ".join(c for c, g in zip(noms, ligne) if not g or c in cl["trop_bruitees"])
                                                            for ligne in g_l])
            ep_l.drop(perdue, reason="contrôle local", verbose=False)
            sorties[cle].append(ep_l)
        # enregistrements sans tâche : fenêtres de 2 s qui se chevauchent de 1 s, 100 Hz
        for nom, raw in r["final"].items():
            if nom == "tache":
                continue
            continu_st = raw.copy().pick("eeg").filter(*BANDE_ERP, verbose=False)
            fen = mne.make_fixed_length_epochs(continu_st, duration=2.0, overlap=1.0, preload=True, verbose=False)
            n_total = len(fen.drop_log)
            x = fen.get_data(picks=scalp, units="uV")
            jugees = cl["jugees"]
            log_cc = np.log(np.ptp(x, axis=-1))
            milieu = np.median(log_cc, axis=0)
            limite = np.exp(milieu + ECART_QC * 1.4826 * np.median(np.abs(log_cc - milieu), axis=0))
            debut_s = (fen.events[:, 0] - raw.first_samp) / sf
            dans_qc = np.zeros((len(fen), len(noms)), bool)
            tq = r["tableau_qc"]
            for p in tq[(tq["enregistrement"] == nom) & (tq["apres_ica"] == "encore au-delà du seuil")].itertuples():
                chev = (debut_s < p.debut + p.duree) & (debut_s + 2.0 > p.debut)
                dans_qc[chev] |= jugees & ((np.array(noms) == p.electrodes) | (p.electrodes == "toutes"))
            t = fen.times
            pente = np.polyfit(t, x.reshape(-1, len(t)).T, 1)[0].reshape(x.shape[:2])
            laissee_st = ((np.ptp(x, axis=-1) > limite) | dans_qc | (np.abs(pente) * (t[-1] - t[0]) > DERIVE_UV)) & jugees
            g = jugees & ~laissee_st & ~np.isin(noms, cl["trop_bruitees"])
            perdue = (~g[:, jugees]).mean(axis=1) > MAX_LAISSEES
            fen.apply_function(lambda d: reference_par_epoque(d, jugees & ~laissee_st), picks=scalp, channel_wise=False)
            fen.metadata = pd.DataFrame({"participant": participant, "posture": posture, "enregistrement": nom,
                                         "debut_s": np.round(debut_s, 3),
                                         "mauvaises": [" ".join(c for c, k in zip(noms, ligne) if not k) for ligne in g]})
            fen.drop(perdue, reason="contrôle local", verbose=False)
            fen.event_id = {nom: 1}
            fen.events[:, 2] = 1
            fen.info["bads"] = list(mauvaises_posture)
            fen = fen.decimate(3, verbose=False)
            (sortie / "sans_tache").mkdir(parents=True, exist_ok=True)
            fen.save(sortie / "sans_tache" / f"sub-{participant}_{nom}_epo.fif", overwrite=True, verbose=False)
            sans_tache.append(dict(participant=participant, enregistrement=nom, posture=posture,
                                   duree_s=round(raw.times[-1], 1), fenetres=n_total, fenetres_gardees=len(fen),
                                   mauvaises=" ".join(c for c in noms if c in mauvaises_posture),
                                   part_laissee=round(float(laissee_st[~perdue][:, jugees].mean()) if len(fen) else np.nan, 3)))
    ecrits = {}
    for cle, chemin in (("mots", f"sub-{participant}_epo.fif"), ("tape", f"tape/sub-{participant}_tape_epo.fif"),
                        ("longues", f"longues/sub-{participant}_longues_epo.fif"),
                        ("longues_tape", f"longues/tape/sub-{participant}_tape_longues_epo.fif")):
        for ep in sorties[cle]:
            ep.info["bads"] = list(bads_fichier)
        ep = mne.concatenate_epochs(sorties[cle], add_offset=True, verbose=False) if len(sorties[cle]) > 1 else sorties[cle][0]   # deux postures : numéros d'échantillon décalés
        ep.event_id = {k: v for k, v in CODES_EXPORT.items() if v in set(ep.events[:, 2])}
        (sortie / chemin).parent.mkdir(parents=True, exist_ok=True)
        ep.save(sortie / chemin, overwrite=True, verbose=False)
        ecrits[cle] = len(ep)
    return dict(epoques=ecrits, sans_tache=pd.DataFrame(sans_tache))


# =====================================================================================================
# Figures (les figures du notebook ; elles ne modifient jamais les données)
# =====================================================================================================
ETAPES = ["Lire,\ncouper", "Mauvaises\nélectrodes", "Filtrer", "Auto-CCA\n(assis)", "Référence", "Préparer\nl'ICA",
          "ICA", "Appliquer\nl'ICA", "Époques,\nERP"]


def carte_chaine(etape):
    """Les étapes de la chaîne : l'étape en cours en orange, celles déjà faites en vert."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 1.2))
    for i, nom in enumerate(ETAPES, start=1):
        couleur = "#f4a261" if i == etape else "#b7e4c7" if i < etape else "#e9ecef"
        ax.add_patch(FancyBboxPatch((i - 0.43, 0.3), 0.86, 0.6, boxstyle="round,pad=0.02", fc=couleur, ec="0.3"))
        ax.text(i, 0.6, nom, ha="center", va="center", fontsize=7.5, weight="bold" if i == etape else "normal")
        if i < len(ETAPES):
            ax.annotate("", xy=(i + 0.58, 0.6), xytext=(i + 0.44, 0.6), arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set(xlim=(0.5, len(ETAPES) + 0.5), ylim=(0.2, 1))
    ax.axis("off")
    plt.show()


def montrer_animation(fig, anim, dpi=72):
    """Montre une animation matplotlib dans le notebook, avec des boutons lecture et pause."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    plt.close(fig)
    fig.set_dpi(dpi)
    with plt.rc_context({"animation.frame_format": "jpeg"}):
        display(HTML(anim.to_jshtml(default_mode="once")))


def figure_essais(raw, ev, debut=3, fin=18):
    """15 s du premier bloc : le canal des déclencheurs, les mots (couleur = condition), les tapes, et Cz."""
    import matplotlib.pyplot as plt

    morceau = raw.copy().crop(debut, fin)
    eeg = morceau.copy().pick("eeg").filter(1, 30, verbose=False)
    cz = (eeg.get_data(picks="Cz")[0] - np.median(eeg.get_data(), axis=0)) * 1e6
    declencheur = morceau.get_data(picks="Trigger")[0]
    e = ev[(ev["onset"] >= debut) & (ev["onset"] <= fin)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True, height_ratios=[1, 2.5])
    ax1.plot(morceau.times + debut, declencheur / max(declencheur.max(), 1e-12), color="0.3", lw=0.8)
    ax2.plot(morceau.times + debut, cz, color="k", lw=0.8)
    for _, ligne in e.iterrows():
        if ligne["trial_type"] == "stimulus":
            c = CONDITIONS[condition_de(ligne)]
            for ax in (ax1, ax2):
                ax.axvline(ligne["onset"], color=c, lw=1.5)
            ax2.text(ligne["onset"] + 0.05, 24, ligne["stimulus"], color=c, fontsize=9)
        elif ligne["trial_type"] == "response":
            ax2.plot(ligne["onset"], -25, "v", color="0.2", ms=7)
    ax1.set(ylabel="déclencheur")
    ax2.set(xlabel="temps dans le bloc (s)", ylabel="Cz (µV)", ylim=(-30, 30))
    ax2.set_title("Cz nettoyée pour cette figure seulement : 1-30 Hz, moins la médiane des 19 électrodes", fontsize=9, loc="left")
    for n, c in CONDITIONS.items():
        ax2.plot([], [], color=c, label=n)
    ax2.plot([], [], "v", color="0.2", label="réponse (tape)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.show()


def figure_signal_brut(raw, debut, fin, marques, titre, zoom=None):
    """Le signal brut (référence Pz) de 19 électrodes, de l'avant à l'arrière, les électrodes des yeux en rouge.
    marques : liste de (début, fin, nom, couleur). zoom : (début, fin, [électrodes]) pour un agrandissement."""
    import matplotlib.pyplot as plt

    yeux = ["Fp1", "Fp2", "F7", "F8"]
    ordre = yeux + ["F3", "Fz", "F4", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
    morceau = raw.copy().pick("eeg").crop(debut, fin)
    t = morceau.times + debut
    d = morceau.get_data(picks=ordre) * 1e6
    d -= d.mean(axis=1, keepdims=True)
    lignes = 2 if zoom else 1
    fig, axes = plt.subplots(lignes, 1, figsize=(12, 8.6 if zoom else 7), height_ratios=[4.3, 1.2] if zoom else None,
                             squeeze=False)
    ax1 = axes[0, 0]
    ecart = 100
    for k, (nom, y) in enumerate(zip(ordre, d)):
        ax1.plot(t, y - k * ecart, lw=0.6, color="tab:red" if nom in yeux else "0.2")
    ax1.set(yticks=[-k * ecart for k in range(len(ordre))], yticklabels=ordre, xlim=(debut, fin),
            ylim=(-len(ordre) * ecart + 40, 330), xlabel="temps dans l'enregistrement (s)", title=titre)
    ax1.plot([fin + 0.1] * 2, [-17.5 * ecart, -18.5 * ecart], color="k", lw=3, clip_on=False)
    ax1.text(fin + 0.25, -18 * ecart, "100 µV", va="center", fontsize=8)
    for t0, t1, nom, couleur in marques:
        ax1.axvspan(t0, t1, color=couleur, alpha=0.2, lw=0)
        ax1.text((t0 + t1) / 2, 0.975, nom, color=couleur, ha="center", va="top", weight="bold",
                 transform=ax1.get_xaxis_transform())
    if zoom:
        ax2 = axes[1, 0]
        z0, z1, electrodes = zoom
        dedans = (t >= z0) & (t <= z1)
        for nom, couleur in zip(electrodes, ["tab:red", "tab:blue"]):
            y = d[ordre.index(nom)]
            ax2.plot(t[dedans], (y - y[dedans][: max(1, dedans.sum() // 3)].mean())[dedans], color=couleur, lw=1, label=nom)
        ax2.set(xlim=(z0, z1), xlabel="temps dans l'enregistrement (s)", ylabel="µV", title="Agrandissement")
        ax2.axhline(0, color="0.7", lw=0.6)
        ax2.legend(loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def carte_qualite(tableau, fenetres, parties, mauvaises, ax, titre, lignes=None, periodes=None):
    """Un tableau de qualite_fenetres en carte de chaleur : une colonne par fenêtre de 5 s ; la ligne 'tout le scalp'
    en haut, puis les électrodes. Couleur : log2 du rapport, de 1/8 à 8. periodes : un cadre par période QC."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lignes = lignes or list(tableau.index)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    image = ax.imshow(np.log2(tableau.reindex(lignes).to_numpy(dtype=float)), cmap=cmap, vmin=-3, vmax=3, aspect="auto",
                      interpolation="nearest")
    p = fenetres["partie"].to_numpy()
    bords = np.flatnonzero(p[1:] != p[:-1]) + 1
    for b in bords:
        ax.axvline(b - 0.5, color="k", lw=1)
    ax.axhline(0.5, color="k", lw=1)
    ax.set(yticks=range(len(lignes)), yticklabels=lignes, xticks=(np.r_[0, bords] + np.r_[bords, len(p)]) / 2 - 0.5,
           xticklabels=list(parties), title=titre)
    for e in ax.get_yticklabels():
        if e.get_text() in mauvaises:
            e.set_color("tab:red")
    if periodes is not None:
        for q in periodes.itertuples():
            dedans = np.flatnonzero((fenetres["enregistrement"] == q.enregistrement) & (fenetres["debut"] >= q.debut - 1e-6)
                                    & (fenetres["debut"] < q.debut + q.duree - 1e-6))
            ligne = 0 if q.electrodes == "toutes" else lignes.index(q.electrodes)
            ax.add_patch(Rectangle((dedans[0] - 0.5, ligne - 0.5), len(dedans), 1, fill=False, ec="k", lw=1.5))
    return image


def barre_couleurs(fig, image, axes):
    barre = fig.colorbar(image, ax=axes, ticks=[-3, -2, -1, 0, 1, 2, 3], shrink=0.7)
    barre.ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])


def figure_composantes(ica, preuves, entrainement_iclabel, choisies, titre):
    """Une ligne par composante : sa carte et ICLabel, 4 s de son décours (gris : tout ; noir : 1-10 Hz), son spectre."""
    import matplotlib.pyplot as plt
    import mne

    morceaux = mne.make_fixed_length_epochs(entrainement_iclabel, duration=4, preload=True, verbose=False)
    sources = ica.get_sources(morceaux).get_data()
    sf = entrainement_iclabel.info["sfreq"]
    lentes = mne.filter.filter_data(sources, sf, 1, 10, verbose=False)
    spectres = ica.get_sources(entrainement_iclabel).compute_psd(fmin=2, fmax=100, picks="misc", n_fft=int(2 * sf),
                                                                 verbose=False)
    fig, axes = plt.subplots(len(choisies), 3, figsize=(13, 2.3 * len(choisies)), width_ratios=[1, 2.4, 1.1], squeeze=False)
    for (ax_c, ax_t, ax_s), k in zip(axes, choisies):
        e = preuves.iloc[k]
        ica.plot_components(picks=k, axes=ax_c, colorbar=False, show=False)
        ax_c.set_title(f"{preuves.index[k]} : {e.decision}, {e.classe}\nICLabel : cerveau {e.ICLabel_cerveau:.2f}, "
                       f"œil {e.ICLabel_oeil:.2f}, muscle {e.ICLabel_muscle:.2f}", fontsize=7.5)
        foyer = lentes if str(e.classe).startswith("oeil") else sources
        m = np.ptp(foyer[:, k], axis=-1).argmax()
        ax_t.plot(morceaux.times, sources[m, k], color="0.6", lw=0.6)
        ax_t.plot(morceaux.times, lentes[m, k], color="k", lw=1.2)
        ax_t.set(yticks=[], xlim=(0, morceaux.times[-1]), xlabel="temps (s)")
        ax_t.set_title(str(e.raison), fontsize=8, loc="left")
        ax_s.loglog(spectres.freqs, spectres.get_data(picks=[k])[0], color="k", lw=0.8)
        ax_s.yaxis.set_major_locator(plt.NullLocator())
        ax_s.yaxis.set_minor_locator(plt.NullLocator())
        ax_s.set_xlabel("fréquence (Hz)")
    fig.suptitle(titre, fontsize=10)
    plt.tight_layout()
    plt.show()


def figure_erp(epochs, mauvaises_posture, titre=""):
    """La P2, la N400 et la LPC, avant (en haut) et après (en bas) le nettoyage, sur la même échelle."""
    import matplotlib.pyplot as plt
    import mne

    erps = {v: {n: moyenne_gardee(ep[n]) for n in CONDITIONS} for v, ep in epochs.items()}
    fig, axes = plt.subplots(2, len(COMPOSANTES), figsize=(15, 7.5))
    for col, (comp, c) in enumerate(COMPOSANTES.items()):
        region = [e for e in c["region"] if e not in mauvaises_posture]
        ondes = [ev.copy().pick(region).data.mean(axis=0) * 1e6 for v in erps.values() for ev in v.values()]
        ylim = (1.15 * min(o.min() for o in ondes), 1.15 * max(o.max() for o in ondes))
        for ligne, (version, evokeds) in enumerate(erps.items()):
            mne.viz.plot_compare_evokeds(evokeds, picks=region, combine="mean", colors=CONDITIONS, axes=axes[ligne, col],
                                         ylim=dict(eeg=ylim), title=f"{comp}, {version}",
                                         legend="upper right" if (ligne, col) == (1, 0) else False,
                                         show_sensors="upper left", show=False)
            axes[ligne, col].axvspan(*c["fenetre"], color="0.85", alpha=0.5, zorder=0)
    fig.suptitle(titre)
    fig.subplots_adjust(hspace=0.35, wspace=0.22)
    plt.show()


# =====================================================================================================
# Figures et démonstrations supplémentaires du notebook 1
# =====================================================================================================
def part_retiree(avant, apres, electrodes, fmin, fmax):
    """La part de la puissance entre fmin et fmax (Hz) qu'une étape a retirée (médiane sur les électrodes)."""
    from scipy.signal import welch

    sf = avant.info["sfreq"]
    a = avant.get_data(picks=electrodes)
    f, p_avant = welch(a, fs=sf, nperseg=int(2 * sf))
    _, p_retire = welch(a - apres.get_data(picks=electrodes), fs=sf, nperseg=int(2 * sf))
    b = (f >= fmin) & (f < fmax)
    return float(np.median(p_retire[:, b].sum(axis=1) / p_avant[:, b].sum(axis=1)))


def tableau_cca(cca, sfreq):
    """Une ligne par composante de chaque fenêtre de l'Auto-CCA : début de la fenêtre, pente, retirée ou non."""
    lignes = []
    for (debut, _), pentes, gardees in zip(cca.spans_, cca.slopes_, cca.kept_mask_):
        lignes += [(debut / sfreq, j, p, not g) for j, (p, g) in enumerate(zip(pentes, gardees))]
    return pd.DataFrame(lignes, columns=["debut_s", "composante", "pente", "retiree"])


def resume_cca(filtre, propre, cca, mauvaises):
    """Notebook 1, 5.1 : pour chaque enregistrement, combien de composantes l'Auto-CCA a retirées par fenêtre, et quelle part de la
    puissance elle a retirée aux tempes (30-100 Hz, les muscles) et à O1/O2 (8-13 Hz, l'alpha)."""
    tempes = [c for c in ["F7", "F8", "T7", "T8"] if c not in mauvaises]
    arriere = [c for c in ["O1", "O2"] if c not in mauvaises]
    for nom in propre:
        t = tableau_cca(cca[nom], filtre[nom].info["sfreq"])
        par_fenetre = t.groupby("debut_s")["retiree"].sum()
        print(f"{nom:10s} {len(par_fenetre):3d} fenêtres ; composantes retirées par fenêtre : médiane {par_fenetre.median():.0f}, au plus "
              f"{par_fenetre.max()} sur {t['composante'].nunique()} | retiré : 30-100 Hz aux tempes "
              f"{100 * part_retiree(filtre[nom], propre[nom], tempes, 30, 100):.0f} %, "
              f"alpha (8-13 Hz) à {'/'.join(arriere)} {100 * part_retiree(filtre[nom], propre[nom], arriere, 8, 13):.0f} %")


def figure_cca(cca, filtre, seuil=CCA_PENTE):
    """Notebook 1, 5.2 : la pente de chaque composante de chaque fenêtre (rouge : retirée), et combien chaque fenêtre en perd."""
    import matplotlib.pyplot as plt

    sfreq = filtre[next(iter(filtre))].info["sfreq"]
    t = pd.concat({nom: tableau_cca(c, sfreq) for nom, c in cca.items()}, names=["enregistrement"]).reset_index(level=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.6), width_ratios=[1, 1.7])
    ax1.scatter(t["composante"] + np.random.default_rng(0).uniform(-0.3, 0.3, len(t)), t["pente"], s=3,
                c=np.where(t["retiree"], "tab:red", "0.6"))
    ax1.axhline(seuil, color="k", ls="--", lw=0.8)
    ax1.set(xlabel="composante (0 : la plus autocorrélée)", ylabel="pente 7-75 Hz (log-log)", title="Chaque composante de chaque fenêtre")
    decalage = 0
    for nom, c in cca.items():
        par_fenetre = tableau_cca(c, sfreq).groupby("debut_s")["retiree"].sum()
        ax2.step((decalage + par_fenetre.index) / 60, par_fenetre.to_numpy(), where="post", color="k", lw=0.8)
        ax2.text((decalage + filtre[nom].times[-1] / 2) / 60, par_fenetre.max() + 0.3, nom, ha="center", fontsize=8)
        decalage += filtre[nom].times[-1]
        ax2.axvline(decalage / 60, color="0.6", lw=0.8)
    ax2.set(xlabel="temps (min ; les enregistrements l'un après l'autre)", ylabel="composantes retirées", title="Combien chaque fenêtre en perd")
    plt.tight_layout()
    plt.show()


def demo_filtres():
    """Passe-haut, passe-bas et passe-bande sur un signal fabriqué : une dérive (0,2 Hz), alpha (10 Hz), 45 Hz."""
    import matplotlib.pyplot as plt
    import mne

    sf = 300
    t = np.arange(0, 20, 1 / sf)
    signal = 20 * np.sin(2 * np.pi * 0.2 * t) + 8 * np.sin(2 * np.pi * 10 * t) + 4 * np.sin(2 * np.pi * 45 * t)
    filtres = {"passe-haut à 1 Hz": (1, None), "passe-bas à 30 Hz": (None, 30), "passe-bande de 1 à 30 Hz": (1, 30)}
    vu = (t >= 8) & (t < 12)
    f = np.fft.rfftfreq(len(t), 1 / sf)[1:]
    fig, axes = plt.subplots(3, 2, figsize=(12, 6.4), width_ratios=[2.5, 1], sharex="col", sharey="col")
    for (ax_t, ax_f), (nom, (lf, hf)) in zip(axes, filtres.items()):
        apres = mne.filter.filter_data(signal, sf, lf, hf, verbose=False)
        ax_t.plot(t[vu] - 8, signal[vu], color="k", lw=0.8)
        ax_t.plot(t[vu] - 8, apres[vu], color="tab:red", lw=1.2)
        ax_t.set(ylabel="µV", title=nom)
        ax_f.axvspan(lf or 0.05, hf or 150, color="0.9", lw=0)
        for x, c in ((signal, "k"), (apres, "tab:red")):
            ax_f.plot(f, np.abs(np.fft.rfft(x))[1:] / len(t) * 2, color=c, lw=1.2)
        ax_f.set(xscale="log", xlim=(0.05, 150), ylabel="µV")
        ax_f.set_xticks([0.1, 1, 10, 100], ["0,1", "1", "10", "100"])
    axes[0, 1].set_title("spectre (gris : gardé)", fontsize=9)
    axes[-1, 0].set_xlabel("temps (s)")
    axes[-1, 1].set_xlabel("fréquence (Hz)")
    fig.suptitle("noir : avant le filtre, rouge : après", fontsize=10)
    plt.tight_layout()
    plt.show()


def animation_passe_haut():
    """Animation : un ERP fabriqué (P2, N400, LPC) après des passe-hauts de plus en plus hauts."""
    import matplotlib.pyplot as plt
    import mne
    from matplotlib import animation

    sf = 100
    t = np.arange(-1, 2, 1 / sf)

    def bosse(centre, largeur, taille):
        return taille * np.exp(-((t - centre) / largeur) ** 2)

    erp = bosse(0.20, 0.035, 2.0) + bosse(0.45, 0.11, -3.0) + bosse(0.72, 0.18, 4.0)
    fn, fl = COMPOSANTES["N400"]["fenetre"], COMPOSANTES["LPC"]["fenetre"]
    dn, dl = (t >= fn[0]) & (t <= fn[1]), (t >= fl[0]) & (t <= fl[1])
    coupures = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    marge = 60 * sf
    filtres = [mne.filter.filter_data(np.r_[np.zeros(marge), erp, np.zeros(marge)], sf, c, None, verbose=False)[marge:-marge]
               for c in coupures]
    fig, ax = plt.subplots(figsize=(9, 3.8))

    def dessiner(k):
        ax.clear()
        ax.plot(t, erp, color="0.6", lw=3, label="ERP avant le filtre")
        ax.plot(t, filtres[k], color="tab:red" if coupures[k] > 0.3 else "tab:green", lw=2, label="après le passe-haut")
        for nom, fen, gris in (("N400", fn, "0.93"), ("LPC", fl, "0.86")):
            ax.axvspan(*fen, color=gris)
            ax.text(np.mean(fen), 4.6, nom, ha="center", fontsize=9)
        ax.set(xlim=(-0.5, 1.5), ylim=(-4.5, 5.2), xlabel="temps depuis le mot (s)", ylabel="µV")
        ax.set_title(f"Passe-haut à {coupures[k]:g} Hz : fenêtre N400 {filtres[k][dn].mean():+.1f} µV (vrai {erp[dn].mean():+.1f}), "
                     f"fenêtre LPC {filtres[k][dl].mean():+.1f} µV (vrai {erp[dl].mean():+.1f})", fontsize=9)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.legend(loc="lower right", fontsize=8)

    montrer_animation(fig, animation.FuncAnimation(fig, dessiner, frames=len(coupures), interval=1200))


def animation_ica():
    """Deux sources (clignements, alpha), deux électrodes : l'ICA tourne les données blanchies jusqu'à ce que les sorties
    soient le moins gaussiennes possible."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from scipy.stats import kurtosis

    rng = np.random.default_rng(1)
    sf = 100
    t = np.arange(0, 20, 1 / sf)
    clignements = sum(np.exp(-((t - b) / 0.08) ** 2) for b in rng.uniform(0.5, 19.5, 9))
    alpha = np.sin(2 * np.pi * 10 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.15 * t))
    sources = np.vstack([clignements - clignements.mean(), alpha])
    melange = np.array([[1.0, 0.4], [0.5, 1.0]])
    enregistre = melange @ sources
    centre = enregistre - enregistre.mean(axis=1, keepdims=True)
    valeurs, vecteurs = np.linalg.eigh(np.cov(centre))
    blanc = vecteurs @ np.diag(valeurs ** -0.5) @ vecteurs.T @ centre

    def tourner(angle):
        a = np.deg2rad(angle)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ blanc

    angles = np.arange(0, 91, 1)
    non_gauss = np.array([np.abs(kurtosis(tourner(a), axis=1)).sum() for a in angles])
    fig, axes = plt.subplots(2, 2, figsize=(12, 3.4), sharex=True)
    for k, (s, e, nom) in enumerate(zip(sources, enregistre, ["Fp1", "Cz"])):
        axes[k, 0].plot(t[:1000], s[:1000], color="tab:purple")
        axes[k, 0].set_title(f"source {k + 1} : {'clignements' if k == 0 else 'rythme alpha'}", fontsize=9, loc="left")
        axes[k, 1].plot(t[:1000], e[:1000], color="k")
        axes[k, 1].set_title(f"ce qu'enregistre « {nom} » : un mélange", fontsize=9, loc="left")
    plt.tight_layout()
    plt.show()
    images = list(range(0, 91, 6))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.6), width_ratios=[1, 1.8, 1.2])

    def dessiner(k):
        angle = images[k]
        s = tourner(angle)
        for ax in (a1, a2, a3):
            ax.clear()
        a1.scatter(s[0, ::3], s[1, ::3], s=2, color="0.4")
        a1.set(xlim=(-4, 7), ylim=(-4, 7), xlabel="sortie 1", ylabel="sortie 2", title="les données, tournées", aspect="equal")
        a2.plot(t[:1000], s[0, :1000] + 4, color="tab:blue")
        a2.plot(t[:1000], s[1, :1000] - 2, color="tab:orange")
        a2.set(yticks=[], xlabel="temps (s)", title="les deux sorties")
        a3.plot(angles, non_gauss, color="k")
        a3.axvline(angle, color="tab:red")
        a3.set(xlabel="rotation (degrés)", ylabel="non-gaussianité", title=f"rotation {angle}°")

    montrer_animation(fig, animation.FuncAnimation(fig, dessiner, frames=len(images), interval=400))


def figure_iclabel(etiquettes):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.3))
    bas = np.zeros(len(etiquettes))
    noms = {"brain": "cerveau", "muscle artifact": "muscle", "eye blink": "œil", "heart beat": "cœur",
            "line noise": "secteur", "channel noise": "bruit d'électrode", "other": "autre"}
    for cl, c in zip(CLASSES, ["tab:green", "tab:red", "tab:blue", "tab:pink", "tab:olive", "tab:brown", "0.7"]):
        ax.bar(etiquettes.index, etiquettes[cl], bottom=bas, color=c, label=noms[cl])
        bas += etiquettes[cl].to_numpy()
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set(ylabel="probabilité ICLabel", title="ICLabel : probabilité de chaque genre de source")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    plt.show()


def figure_clignement_references(avant_ref, apres_ref, clignements, bonnes_toutes):
    """Le même clignement avec la référence Pz et avec la référence moyenne, et la carte au pic."""
    import matplotlib.pyplot as plt
    import mne

    sf = apres_ref.info["sfreq"]
    yeux = mne.filter.filter_data(apres_ref.get_data(picks=[c for c in ["Fp1", "Fp2"] if c in bonnes_toutes]), sf, 1, 10,
                                  verbose=False).min(axis=0)
    net = clignements[np.argmax(yeux[clignements[:, 0] - apres_ref.first_samp]), 0]
    t0 = (net - apres_ref.first_samp) / sf
    montrees = ["Fp1", "Fz", "Cz", "Pz", "O1"]
    fig = plt.figure(figsize=(13, 4))
    for col, (nom, d) in enumerate([("référence Pz", avant_ref), ("référence moyenne", apres_ref)]):
        m = d.copy().crop(t0 - 1.5, t0 + 1.5)
        ax = fig.add_subplot(1, 4, 2 * col + 1)
        for k, c in enumerate(montrees):
            ax.plot(m.times - 1.5, m.get_data(picks=c)[0] * 1e6 - 120 * k, color="k", lw=0.8)
            ax.text(-1.6, -120 * k, c, ha="right", va="center")
        ax.set(title=nom, xlabel="temps depuis le pic (s)", yticks=[])
        pic = d.copy().pick(bonnes_toutes).crop(t0, t0)
        ax = fig.add_subplot(1, 4, 2 * col + 2)
        im, _ = mne.viz.plot_topomap(pic.get_data()[:, 0] * 1e6, pic.info, axes=ax, show=False, cmap="RdBu_r")
        plt.colorbar(im, ax=ax, shrink=0.6, label="µV")
        ax.set_title(f"carte au pic\n({nom})", fontsize=9)
    plt.tight_layout()
    plt.show()
    return t0


def figure_clignement_ica(avant, apres, clignements, bonnes_toutes):
    """Le clignement moyen avant et après l'ICA, à Fp1 et Cz, et sa carte au pic."""
    import matplotlib.pyplot as plt
    import mne

    moy = {nom: mne.Epochs(d, clignements, tmin=-0.5, tmax=0.5, baseline=(-0.5, -0.3), preload=True, verbose=False).average()
           for nom, d in (("avant l'ICA", avant), ("après l'ICA", apres))}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3), width_ratios=[1.6, 1.6, 1, 1])
    avant_yeux = [c for c in ["Fp1", "Fp2"] if c in bonnes_toutes][0]
    for ax, c in zip(axes[:2], [avant_yeux, "Cz"]):
        for (nom, ev), coul in zip(moy.items(), ["tab:red", "k"]):
            ax.plot(ev.times, ev.get_data(picks=c)[0] * 1e6, color=coul, lw=2, label=nom)
        ax.set(title=f"clignement moyen à {c}", xlabel="temps depuis le pic (s)", ylabel="µV")
        ax.legend(fontsize=8)
    pic = moy["avant l'ICA"].get_data(picks=bonnes_toutes).max() * 1e6
    for ax, (nom, ev) in zip(axes[2:], moy.items()):
        d = ev.copy().pick(bonnes_toutes).crop(0, 0)
        mne.viz.plot_topomap(d.get_data()[:, 0] * 1e6, d.info, axes=ax, show=False, vlim=(-pic, pic), cmap="RdBu_r")
        ax.set_title(f"{nom}\n(même échelle)", fontsize=9)
    plt.tight_layout()
    plt.show()


def figure_traces(versions, t0, electrodes=("Fp1", "F7", "Fz", "Cz", "O1"), titre=""):
    """Les mêmes 8 s (autour de t0) dans plusieurs versions des données (dict nom -> (Raw, couleur))."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.8))
    for k, c in enumerate(electrodes):
        for nom, (d, coul) in versions.items():
            m = d.copy().crop(t0 - 4, t0 + 4)
            ax.plot(m.times - 4, m.get_data(picks=c)[0] * 1e6 - 100 * k, color=coul, lw=0.9)
        ax.text(-4.2, -100 * k, c, ha="right", va="center")
    for nom, (_, coul) in versions.items():
        ax.plot([], [], color=coul, label=nom)
    ax.set(yticks=[], xlabel="temps (s)", title=titre)
    ax.legend(loc="upper right", fontsize=8)
    plt.show()


def figure_controle_local(cl):
    import matplotlib.pyplot as plt

    vues = np.flatnonzero(cl["jugees"])
    noms = cl["noms"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 3.8), width_ratios=[3, 2])
    bas = np.zeros(len(vues))
    for k, (regle, c) in enumerate([("période QC encore au-delà du seuil", "tab:purple"), ("trop grande", "tab:orange"),
                                    ("dérive", "tab:brown"), ("trop grande après la nouvelle référence", "tab:pink")], start=1):
        part = 100 * (cl["regle"][:, vues] == k).mean(axis=0)
        axes[0].bar(noms[vues], part, bottom=bas, color=c, label=regle)
        bas += part
    axes[0].set(ylabel="% des époques", title="Combien de fois chaque électrode est laissée de côté, par règle")
    axes[0].legend(fontsize=8)
    axes[1].bar(noms[vues], cl["sme"][vues], color=["tab:red" if c in cl["trop_bruitees"] else "0.6" for c in noms[vues]])
    axes[1].axhline(cl["limite_sme"], color="k", ls="--", lw=1)
    axes[1].set(ylabel="SME (µV)", title=f"SME de la N400 (tirets : {FACTEUR_SME:g} × la médiane ; rouge : laissée de côté)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=90)
    plt.show()


def pas_moyen(donnees, pas, parties, bonnes_toutes):
    """La moyenne des données autour des pas (-0,3 à 0,3 s du pic de l'accélération verticale)."""
    import mne

    evokeds = []
    for enregistrement, raw in donnees.items():
        e = np.concatenate([pas[p] + premier for p, (rec, premier) in parties.items() if rec == enregistrement])
        ev = np.column_stack([e + raw.first_samp, np.zeros(len(e), int), np.ones(len(e), int)])
        evokeds.append(mne.Epochs(raw, ev, tmin=-0.3, tmax=0.3, baseline=(None, None), picks=bonnes_toutes, preload=True,
                                  verbose=False).average())
    return mne.combine_evoked(evokeds, weights="nave")


def figure_pas(versions, titre):
    """La taille du pas moyen à chaque électrode (RMS, µV) dans plusieurs versions (dict nom -> Evoked)."""
    import matplotlib.pyplot as plt

    taille = pd.DataFrame({nom: np.sqrt((ev.data ** 2).mean(axis=1)) for nom, ev in versions.items()},
                          index=next(iter(versions.values())).ch_names) * 1e6
    pire = taille.iloc[:, 0].idxmax()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 3.6), width_ratios=[1, 2])
    for (nom, ev), c in zip(versions.items(), ["tab:red", "k", "tab:blue"]):
        a1.plot(ev.times, ev.get_data(picks=pire)[0] * 1e6, color=c, label=nom)
    a1.set(xlabel="temps depuis le pas (s)", ylabel="µV", title=f"Le pas moyen à {pire}")
    a1.legend(fontsize=8)
    x = np.arange(len(taille))
    w = 0.8 / taille.shape[1]
    for j, ((nom, col), c) in enumerate(zip(taille.items(), ["tab:red", "k", "tab:blue"])):
        a2.bar(x + (j - (taille.shape[1] - 1) / 2) * w, col, w, color=c, label=nom)
    a2.set(xticks=x, ylabel="RMS (µV)", title=titre)
    a2.set_xticklabels(taille.index, fontsize=8)
    a2.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return taille


# =====================================================================================================
# Figures du notebook 2 (ERP)
# =====================================================================================================
def figure_conditions(tableaux, couleurs, prefixe="", largeur=4, recherche_latence=None):
    """Pour chaque mesure, une barre par condition (la moyenne du groupe) et un trait par participant (F1 et F6).

    tableaux : {mesure: tableau participants x [A, B]} ; couleurs : {condition: couleur} ; prefixe : le début du titre de
    chaque panneau ; recherche_latence : (début, fin) en s de la fenêtre de recherche du pic, ombrée dans le dernier panneau
    (la latence, en ms)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tableaux), figsize=(largeur * len(tableaux), 4), squeeze=False)
    for ax, (mesure, large) in zip(axes[0], tableaux.items()):
        a, b = large.columns
        ax.bar([0, 1], large.mean(), color=[couleurs[a], couleurs[b]], alpha=0.3)       # la moyenne du groupe
        ax.plot([0, 1], large.T.to_numpy(), color="0.2", marker="o")                    # un trait par participant
        ax.set(xticks=[0, 1], xticklabels=[a, b], title=prefixe + mesure)
    if recherche_latence is not None:
        debut, fin = recherche_latence[0] * 1000, recherche_latence[1] * 1000
        axes[0, -1].axhspan(debut, fin, color="0.95", zorder=0)
        axes[0, -1].set_ylim(debut - 20, fin + 20)
    plt.tight_layout()
    plt.show()


def pentes_region(pentes, region):
    """La pente de la ligne de base de chaque participant dans la région (2.0) : un Evoked à une électrode virtuelle, 'région',
    la moyenne des électrodes de la région où la pente est estimée. pentes : {participant: Evoked de la pente} ; un participant
    sans aucune électrode estimée dans la région est laissé de côté."""
    import mne

    reference = next(iter(pentes.values()))
    info_region = mne.create_info(["région"], reference.info["sfreq"], "eeg")
    return [mne.EvokedArray(p.get_data(picks=[c for c in region if c not in p.info["bads"]]).mean(axis=0, keepdims=True),
                            info_region, tmin=p.tmin) for p in pentes.values() if any(c not in p.info["bads"] for c in region)]


def figure_pente(pentes, pente_region, region, fenetre, zero):
    """La pente de la ligne de base (2.0) : dans la région au cours du temps (la moyenne des participants et son IC à 95 %), et
    sur la tête dans la fenêtre (à chaque électrode, la moyenne des participants chez qui elle est estimée, s'ils sont au moins
    3 et la moitié d'entre eux)."""
    import matplotlib.pyplot as plt
    import mne

    reference = next(iter(pentes.values()))
    valeurs = np.array([np.where(np.isin(p.ch_names, p.info["bads"])[:, None], np.nan, p.data) for p in pentes.values()])
    vue = np.isfinite(valeurs[:, :, 0]).sum(axis=0) >= max(3, len(pentes) / 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[2, 1])
    mne.viz.plot_compare_evokeds({"pente": pente_region}, picks="région", ci=0.95, colors=["k"], axes=axes[0], legend=False,
                                 show_sensors=False, title=f'Pente de la ligne de base, région {", ".join(region)}', show=False)
    bas, haut = axes[0].get_ylim()
    axes[0].set_ylim(min(bas, -0.1), max(haut, 1.15))           # 0 et 1 toujours visibles
    for valeur, texte in ((1, "soustraction"), (0, "aucune correction")):
        axes[0].axhline(valeur, color="0.6", ls="--", lw=1)
        axes[0].text(axes[0].get_xlim()[0], valeur, f" {texte}", va="bottom", fontsize=8, color="0.4")
    axes[0].axvspan(*fenetre, color="0.9", zorder=0)
    axes[0].set(xlabel=f"Temps depuis {zero} (s)", ylabel="pente (sans unité)")
    if vue.any():
        carte = mne.EvokedArray(np.nanmean(valeurs[:, vue], axis=0), mne.pick_info(reference.info, np.flatnonzero(vue)),
                                tmin=reference.tmin)
        carte.info["bads"] = []
        carte.plot_topomap(times=np.mean(fenetre), average=fenetre[1] - fenetre[0], axes=axes[1], vlim=(0, 1.5), cmap="viridis",
                           contours=0, colorbar=False, show=False, time_format="")
        fig.colorbar(axes[1].images[0], ax=axes[1], shrink=0.7, label="pente")
    axes[1].set_title(f"Pente moyenne, {fenetre[0] * 1000:.0f} à {fenetre[1] * 1000:.0f} ms")
    plt.show()


def figure_ondes(ondes, couleurs, titre, zero, fenetre=None, amas=(), p_amas=(), passe_bas=20.0):
    """Les ERP du groupe de l'électrode virtuelle 'région' (F2, F3) : pour chaque onde, la moyenne des participants et son IC à
    95 %. ondes : {nom: liste d'Evoked, un par participant} ; fenetre : ombrée en gris ; amas et p_amas (2.2) : chaque amas
    en vert s'il est significatif (p < .05), en gris sinon (p < .05 instant par instant seulement)."""
    import matplotlib.pyplot as plt
    import mne

    if passe_bas:                                                # pour l'affichage seulement
        ondes = {nom: [e.copy().filter(None, passe_bas, verbose=False) for e in liste] for nom, liste in ondes.items()}
    figure = mne.viz.plot_compare_evokeds(ondes, picks="région", ci=0.95, colors=couleurs, show_sensors=False, title=titre,
                                          truncate_xaxis=False, truncate_yaxis=False, show=False)[0]
    ax = figure.axes[0]
    if fenetre is not None:
        ax.axvspan(*fenetre, color="0.9", zorder=0)              # la fenêtre de la composante
    temps = next(iter(ondes.values()))[0].times
    for masque, p in zip(amas, p_amas):
        ax.axvspan(temps[masque][0], temps[masque][-1], color="C2" if p < 0.05 else "0.5", alpha=0.3, lw=0, zorder=0)
    ax.set_xlabel(f"Temps depuis {zero} (s)")
    if passe_bas:
        ax.text(1.0, -0.16, f"affichage : passe-bas à {passe_bas:.0f} Hz (les mesures utilisent les données de 1 à 40 Hz)",
                transform=ax.transAxes, ha="right", fontsize=8, color="0.45")
    plt.show()


def figure_image_essais(essais, centre, zero, titre):
    """L'image des essais (2.3) : une ligne par essai de l'électrode virtuelle 'région', du TR le plus court (en bas) au plus
    long, avec l'autre événement (la tape après le mot, ou le mot avant la tape) en trait noir, et leur moyenne en dessous."""
    import matplotlib.pyplot as plt
    import mne

    tr = essais.metadata["tr_ms"].to_numpy() / 1000             # le temps de réaction de chaque essai (s)
    autre = tr if centre == "mot" else -tr
    autre = np.where((autre >= essais.tmin) & (autre <= essais.tmax), autre, np.nan)   # (hors de l'époque : pas tracé)
    figure = mne.viz.plot_epochs_image(essais, picks="région", order=np.argsort(tr), overlay_times=autre, sigma=2, title=titre,
                                       show=False)[0]
    figure.axes[0].set_ylabel("Essais (temps de réaction croissant)")
    figure.axes[1].set_xlabel(f"Temps depuis {zero} (s)")
    plt.show()


def erp_groupe_visibles(erp_A, erp_B, deux, vue, cond_a, cond_b):
    """L'ERP du groupe à chaque électrode visible (F4) : la moyenne des participants qui l'ont mesurée dans les deux conditions.
    deux : participants x électrodes, mesurée dans les deux conditions ; vue : les électrodes visibles. Rend {A, B, B - A} (des
    Evoked aux seules électrodes visibles) et l'info de ces électrodes."""
    import mne

    visibles = np.flatnonzero(vue)
    info_vue = mne.pick_info(erp_A[0].info, visibles)
    info_vue["bads"] = []
    groupe = {}
    for nom, erps in ((cond_a, erp_A), (cond_b, erp_B)):
        donnees = np.array([e.data for e in erps])               # participants x électrodes x instants (V)
        groupe[nom] = mne.EvokedArray(np.array([donnees[deux[:, j], j].mean(axis=0) for j in visibles]), info_vue,
                                      tmin=erps[0].tmin)
    groupe[f"{cond_b} - {cond_a}"] = mne.combine_evoked([groupe[cond_b], groupe[cond_a]], weights=[1, -1])
    return groupe, info_vue


def figure_topographies(groupe, info, vue, region, p_electrodes, fenetre, titre):
    """Les cartes de A, B et B - A dans la fenêtre (F4), sur une même échelle. Points noirs : la région ; ronds blancs sur la
    différence : p_electrodes < .05 ; cercles vides : les électrodes non visibles (trop peu mesurées). info : toutes les
    électrodes ; vue : les électrodes visibles."""
    import matplotlib.pyplot as plt
    import mne
    from matplotlib.colors import ListedColormap                 # une carte de couleurs transparente, pour les cercles vides

    noms = np.array(info.ch_names)
    limite = max(np.abs(ev.copy().crop(*fenetre).get_data(units="uV").mean(axis=1)).max() for ev in groupe.values())
    masques = [np.isin(noms[vue], region)] * 2 + [p_electrodes < 0.05]
    styles = [dict(marker="o", markerfacecolor="k", markersize=4)] * 2 + [dict(marker="o", markerfacecolor="w", markeredgecolor="k",
                                                                                markersize=8)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (nom, ev), masque, style in zip(axes, groupe.items(), masques, styles):
        ev.plot_topomap(times=np.mean(fenetre), average=fenetre[1] - fenetre[0], axes=ax, vlim=(-limite, limite), cmap="RdBu_r",
                        colorbar=False, mask=np.repeat(masque[:, None], len(ev.times), axis=1), mask_params=style, time_format="",
                        show=False)
        if not vue.all():
            mne.viz.plot_topomap(np.zeros(len(noms)), info, axes=ax, show=False, vlim=(-1, 1), cmap=ListedColormap([(0, 0, 0, 0)]),
                                 contours=0, sensors=False, mask=~vue,
                                 mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="0.35", markersize=6))
        ax.set_title(nom)
    fig.colorbar(axes[0].images[0], ax=axes, shrink=0.7, label="µV")
    fig.suptitle(titre)
    plt.show()


def t_mesures(X):
    """Le t de chaque instant et de chaque électrode, avec les seuls participants qui l'ont mesurée (les autres valent NaN) :
    la statistique du test par amas sur toute la tête (3.1)."""
    n = np.isfinite(X).sum(axis=0)
    return np.nanmean(X, axis=0) / (np.nanstd(X, axis=0, ddof=1) / np.sqrt(n))


def amas_le_plus_fort(amas, p_amas, temps, electrodes):
    """Les électrodes et la fenêtre (premier et dernier instant) de l'amas au p le plus petit (3.1). amas : des masques
    instants x électrodes ; temps : les instants testés ; electrodes : les noms des électrodes testées."""
    fort = amas[np.argmin(p_amas)]
    instants = temps[fort.any(axis=1)]
    return list(np.array(electrodes)[fort.any(axis=0)]), (instants[0], instants[-1])


def figure_amas_tete(t_obs, amas, p_amas, temps, info, difference, nom, zero):
    """Le test par amas sur toute la tête (F5). À gauche : le t de chaque électrode (une ligne, l'avant de la tête en haut) à
    chaque instant testé, en gris hors des amas à p < .05. À droite : la carte de la différence du groupe pendant l'amas le plus
    fort, ses électrodes en blanc. info : les électrodes testées ; difference : l'Evoked de la différence du groupe."""
    import matplotlib.pyplot as plt
    import mne

    significatif = np.zeros(t_obs.shape, dtype=bool)             # les points (instants x électrodes) des amas à p < .05
    for masque, p in zip(amas, p_amas):
        if p < 0.05:
            significatif |= masque
    ordre = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
    rang = [info.ch_names.index(c) for c in ordre[::-1] if c in info.ch_names]   # l'image : l'avant de la tête en haut
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), width_ratios=[2, 1])
    limite_t = np.abs(t_obs).max()
    carte_t = mne.EvokedArray(t_obs[:, rang].T, mne.pick_info(info, rang), tmin=temps[0])   # les t, dans une Evoked pour plot_image
    carte_t.plot_image(axes=axes[0], mask=significatif[:, rang].T, mask_style="mask", mask_alpha=0.4,
                       clim=dict(eeg=[-limite_t, limite_t]), scalings=dict(eeg=1), units=dict(eeg="t"), show_names="all", show=False)
    for texte in axes[0].texts:                                  # (plot_image écrit le nombre d'essais : sans objet pour des t)
        texte.remove()
    axes[0].set(title=f"t de {nom} ; en gris : hors des amas à p < .05", xlabel=f"Temps depuis {zero} (s)", ylabel="")
    if len(amas):
        fort = amas[np.argmin(p_amas)]
        instants = temps[fort.any(axis=1)]
        limite_c = np.abs(difference.copy().crop(instants[0], instants[-1]).get_data(units="uV").mean(axis=1)).max()
        difference.plot_topomap(times=np.mean((instants[0], instants[-1])),
                                average=instants[-1] - instants[0] + 1 / difference.info["sfreq"], axes=axes[1],
                                vlim=(-limite_c, limite_c), cmap="RdBu_r", colorbar=False, time_format="", show=False,
                                mask=np.repeat(fort.any(axis=0)[:, None], len(difference.times), axis=1),
                                mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k", markersize=8))
        fig.colorbar(axes[1].images[0], ax=axes[1], shrink=0.7, label="µV")
        axes[1].set_title(f"L'amas le plus fort : {nom}, {instants[0] * 1000:.0f} à {instants[-1] * 1000:.0f} ms")
    else:
        axes[1].set_axis_off()
    plt.show()


def figure_interaction(ondes, cartes, interaction, interaction_c, electrodes, fenetre, nom_effet, comportement, zero):
    """L'effet dans chaque posture et l'interaction (F7). En haut : l'effet assis et en marche, l'écart marche - assis (la
    moyenne des participants et son IC à 95 %), puis l'effet de chaque participant dans l'EEG (interaction) et dans le
    comportement (interaction_c). En bas : les cartes de l'effet assis, en marche, et de l'écart, dans la fenêtre (à chaque
    électrode, la moyenne des participants qui l'ont mesurée, s'ils sont au moins 3 et la moitié d'entre eux).
    ondes et cartes : {posture: liste d'Evoked, un par participant}, dans la zone et à chaque électrode."""
    import matplotlib.pyplot as plt
    import mne

    unite_c = comportement.split("(")[1].rstrip(")")              # 'ms' ou '%'
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), layout="constrained")
    mne.viz.plot_compare_evokeds(ondes, picks="zone", ci=0.95, colors={"assis": "#2f6db5", "marche": "#e0791d"}, axes=axes[0, 0],
                                 show_sensors=False, title=f"{nom_effet}, dans chaque posture", truncate_xaxis=False, show=False)
    ecarts = [mne.combine_evoked([m, a], weights=[1, -1]) for a, m in zip(ondes["assis"], ondes["marche"])]   # marche - assis
    mne.viz.plot_compare_evokeds({"marche - assis": ecarts}, picks="zone", ci=0.95, colors=["k"], axes=axes[0, 1],
                                 show_sensors=False, title="L'interaction : marche - assis", truncate_xaxis=False, show=False)
    for ax in axes[0, :2]:
        ax.axvspan(*fenetre, color="0.9", zorder=0)
        ax.set_xlabel(f"Temps depuis {zero} (s)")
    for ax, tableau, unite, titre in ((axes[0, 2], interaction, "µV", f"{fenetre[0] * 1000:.0f} à {fenetre[1] * 1000:.0f} ms"),
                                      (axes[0, 3], interaction_c, unite_c, comportement)):
        ax.bar([0, 1], tableau.mean(), color="0.85")
        ax.plot([0, 1], tableau.T.to_numpy(), color="0.2", marker="o")           # un trait par participant
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set(xticks=[0, 1], xticklabels=["assis", "marche"], ylabel=f"effet {nom_effet} ({unite})", title=titre)
    axes[1, 3].set_axis_off()
    cartes = {**cartes, "marche - assis": [mne.combine_evoked([m, a], weights=[1, -1])
                                           for a, m in zip(cartes["assis"], cartes["marche"])]}
    valeurs = {nom: np.array([c.copy().crop(*fenetre).get_data(units="uV").mean(axis=1) for c in liste])
               for nom, liste in cartes.items()}
    vue = np.isfinite(valeurs["marche - assis"]).sum(axis=0) >= max(3, len(interaction) / 2)
    info = mne.pick_info(cartes["assis"][0].info, np.flatnonzero(vue))
    info["bads"] = []
    limite = max(np.abs(np.nanmean(v[:, vue], axis=0)).max() for v in valeurs.values())
    for ax, (nom, v) in zip(axes[1], valeurs.items()):
        mne.viz.plot_topomap(np.nanmean(v[:, vue], axis=0), info, axes=ax, vlim=(-limite, limite), cmap="RdBu_r", show=False,
                             mask=np.isin(info.ch_names, electrodes), mask_params=dict(marker="o", markerfacecolor="k", markersize=4))
        ax.set_title(f"{nom}, {fenetre[0] * 1000:.0f} à {fenetre[1] * 1000:.0f} ms")
    fig.colorbar(axes[1, 0].images[0], ax=axes[1, :3], shrink=0.7, label=f"effet {nom_effet} (µV)")
    plt.show()


def figure_nuage(points, xlabel, ylabel):
    """Le nuage de points (F8) : un point par participant (colonnes 'comportement' et 'eeg' de points), son nom, et la droite
    qui passe au mieux par les points."""
    import matplotlib.pyplot as plt

    pente, ordonnee = np.polyfit(points["comportement"], points["eeg"], 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(points["comportement"], points["eeg"], color="k")
    for participant, ligne in points.iterrows():
        ax.annotate(participant, (ligne["comportement"], ligne["eeg"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    xs = np.array([points["comportement"].min(), points["comportement"].max()])
    ax.plot(xs, pente * xs + ordonnee, color="0.6")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.show()


# =====================================================================================================
# Notebook 2 (ERP) : préparer les époques, le comportement, l'effet dans chaque posture
# =====================================================================================================
REGION_P200 = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]     # la région de la P200 et de la P200-N250
REGION_CP = ["C3", "Cz", "C4", "P3", "Pz", "P4"]                        # la région centro-pariétale : N400, P3/LPC, CPP
COMPOSANTES_ERP = {   # composante : (le sens du pic, la fenêtre de recherche du pic en s, le zéro des temps) ; partie 4
    "P200": ("pos", (0.15, 0.30), "mot"),
    "P200-N250": ("pos", (0.15, 0.35), "mot"),
    "N400": ("neg", (0.25, 0.65), "mot"),
    "P3/LPC": ("pos", (0.45, 0.90), "mot"),
    "CPP": ("pos", (-0.50, 0.10), "tape"),
    "P3/LPC (tape)": ("pos", (-0.40, 0.10), "tape"),     # la P3/LPC alignée sur la réponse (partie 8, défi A)
}


def conditions_erp(hypothese, posture="les deux"):
    """Les deux conditions comparées, A et B (l'effet = B - A), et comment choisir leurs époques : une condition MNE
    ('assis/HF'...), ou, pour 'tr', une requête sur les métadonnées. posture : 'les deux', 'assis' ou 'marche' (sauf pour
    'posture')."""
    debut = "" if posture == "les deux" else f"{posture}/"
    if hypothese == "posture":
        return "assis", "marche", "assis", "marche"
    if hypothese == "frequence":
        return "HF", "LF", f"{debut}HF", f"{debut}LF"
    if hypothese == "tr":
        filtre = "" if posture == "les deux" else f' and posture == "{posture}"'
        return "rapides", "lents", f'vitesse == "rapides"{filtre}', f'vitesse == "lents"{filtre}'
    return "mots", "pseudo-mots", [f"{debut}HF", f"{debut}LF"], f"{debut}pseudo"        # 'lexicalite'


def region_mesuree(ep, region):
    """Le signal de la région dans chaque époque (µV) : la moyenne des seules électrodes de la région mesurées dans cette époque
    (pas dans la colonne mauvaises ; NaN si aucune ne l'est). Rend aussi le nombre d'électrodes mesurées dans chaque époque."""
    x = ep.get_data(picks=region, units="uV")                                    # époques x électrodes x instants
    mesuree = np.array([[c not in str(r).split() for c in region] for r in ep.metadata["mauvaises"]])
    n = mesuree.sum(axis=1)
    signal = (x * mesuree[:, :, None]).sum(axis=1) / np.maximum(n, 1)[:, None]
    return np.where(n[:, None] > 0, signal, np.nan), n


def corriger_ligne_de_base(mot, ep, methode="soustraction", min_epoques=20):
    """Les époques ep corrigées de leur ligne de base (la moyenne de -0,2 à 0 s avant le mot, prise dans l'époque du mot du même
    essai, mot, aussi pour la tape), et la pente de cette ligne de base à chaque instant (un Evoked ; ses électrodes non estimées
    sont dans bads). 'soustraction' : la pente vaut 1 partout. 'regression' : pour chaque électrode, mne.stats.linear_regression
    estime un ERP par condition et la pente, sur les seuls essais où l'électrode est mesurée (Alday, 2019)."""
    import mne

    lignes = pd.DataFrame(mot.copy().crop(tmax=0.0).get_data().mean(axis=-1), columns=mot.ch_names,
                          index=pd.MultiIndex.from_frame(mot.metadata[["posture", "essai"]]))
    base = lignes.loc[pd.MultiIndex.from_frame(ep.metadata[["posture", "essai"]])].to_numpy()   # essais x électrodes (V)
    pentes, estimees = np.ones((len(ep.ch_names), len(ep.times))), []
    if methode == "regression":
        conditions = (ep.metadata["posture"] + "/" + ep.metadata["type"]).to_numpy()
        for k, ch in enumerate(ep.ch_names):
            mesure = np.flatnonzero([ch not in str(r).split() for r in ep.metadata["mauvaises"]])
            if len(mesure) < min_epoques:                    # trop peu d'essais pour l'estimer : la pente reste 1
                continue
            X = pd.get_dummies(conditions[mesure], dtype=float)  # un ERP par condition...
            X["ligne de base"] = base[mesure, k] * 1e6           # ... et la ligne de base, en µV
            une = ep[mesure].pick([ch])
            une.info["bads"] = []
            modele = mne.stats.linear_regression(une, X.to_numpy(), names=list(X.columns))
            pentes[k] = modele["ligne de base"].beta.data[0] * 1e6
            estimees.append(ch)
    corrigees = mne.EpochsArray(ep.get_data() - base[:, :, None] * pentes, ep.info, events=ep.events, tmin=ep.tmin,
                                event_id=ep.event_id, metadata=ep.metadata, baseline=None, on_missing="ignore")
    pente = mne.EvokedArray(pentes * 1e-6, ep.info, tmin=ep.tmin)
    pente.info["bads"] = [ch for ch in ep.ch_names if ch not in estimees] if methode == "regression" else []
    return corrigees, pente


def preparer_epoques(epochs, epochs_tape, centre, region, hypothese, ligne_de_base="soustraction", sans_clignement=False,
                     minimum=1):
    """Les époques de chaque participant, prêtes pour les parties 1 à 6 (cellule 0.5) :
    1. autour du mot, ou de la tape (centre) ; seulement les essais sans clignement entre le mot et la tape, si sans_clignement ;
    2. sans les époques où la région a moins de minimum électrodes mesurées ;
    3. pour 'tr', une colonne vitesse : 'lents' si le TR dépasse la médiane du participant (dans sa posture et son type de mot) ;
    4. la ligne de base retirée (celle du mot du même essai, aussi pour la tape) ;
    5. le signal de la région : des époques à une seule électrode virtuelle, 'région'.
    Rend trois dictionnaires {participant: ...} : les époques, les époques de la région, la pente de la ligne de base."""
    import mne

    source = epochs_tape if centre == "tape" else epochs
    if not source:
        raise ValueError("Époques introuvables : relancez la cellule 0.1, elle télécharge la dernière version des données")
    epoques, regionales, pentes = {}, {}, {}
    for participant, ep in source.items():
        if sans_clignement:
            ep = ep.copy().drop(~ep.metadata["sans_clignement"].to_numpy(bool), reason="clignement")
        ep = ep.copy().drop(region_mesuree(ep, region)[1] < minimum, reason="région")
        if hypothese == "tr":
            mediane = ep.metadata.groupby(["posture", "type"])["tr_ms"].transform("median")
            ep.metadata = ep.metadata.assign(vitesse=np.where(ep.metadata["tr_ms"] > mediane, "lents", "rapides"))
        epoques[participant], pentes[participant] = corriger_ligne_de_base(epochs[participant], ep, ligne_de_base)
        e = epoques[participant]
        regionales[participant] = mne.EpochsArray(region_mesuree(e, region)[0][:, None] * 1e-6,   # (MNE travaille en volts)
                                                  mne.create_info(["région"], e.info["sfreq"], "eeg"), events=e.events, tmin=e.tmin,
                                                  event_id=e.event_id, metadata=e.metadata, baseline=None, on_missing="ignore")
    return epoques, regionales, pentes


def qualite_region(epoques, region, minimum=4):
    """La qualité de la région (cellule 0.6), pour chaque participant et posture : le nombre d'époques, les électrodes de la
    région jamais mesurées, le nombre moyen d'électrodes de la région mesurées par époque, et la part des époques qui en ont
    moins de minimum."""
    lignes = []
    for participant, ep in epoques.items():
        for posture in ("assis", "marche"):
            md = ep.metadata[ep.metadata["posture"] == posture]
            if md.empty:
                continue
            mesurees = np.array([[c not in str(r).split() for c in region] for r in md["mauvaises"]])   # époques x électrodes
            n = mesurees.sum(axis=1)
            lignes.append({"participant": participant, "posture": posture, "époques": len(md),
                           "jamais mesurées": " ".join(np.array(region)[~mesurees.any(axis=0)]) or "-",
                           f"mesurées par époque (sur {len(region)})": round(n.mean(), 1),
                           f"époques avec moins de {minimum} (%)": round(100 * (n < minimum).mean())})
    return pd.DataFrame(lignes).set_index(["participant", "posture"])


def comportement(essais, hypothese, posture="les deux"):
    """Le comportement de chaque participant dans chaque condition (partie 1) : le TR moyen des réponses valides (correctes,
    250 ms ou plus, pas aberrantes : la colonne tr_valide), l'exactitude (%) et l'IES = TR / exactitude. Pour 'tr', les essais
    sont coupés au TR médian du participant (dans sa posture et son type de mot) : seul le TR a un sens."""
    if hypothese == "posture":
        choisis = essais.assign(condition=essais["posture"])
    else:
        choisis = essais if posture == "les deux" else essais[essais["posture"] == posture]
        if hypothese == "frequence":
            choisis = choisis[choisis["type"].isin(["HF", "LF"])].assign(condition=lambda t: t["type"])
        elif hypothese == "tr":
            choisis = choisis[choisis["tr_valide"]]
            mediane = choisis.groupby(["participant", "posture", "type"])["tr_ms"].transform("median")
            choisis = choisis.assign(condition=np.where(choisis["tr_ms"] > mediane, "lents", "rapides"))
        else:                                                                    # 'lexicalite'
            choisis = choisis.assign(condition=np.where(choisis["type"] == "pseudo", "pseudo-mots", "mots"))
    groupes = ["participant", "condition"]
    comport = pd.DataFrame({"TR (ms)": choisis[choisis["tr_valide"]].groupby(groupes)["tr_ms"].mean(),
                            "Exactitude (%)": 100 * choisis.groupby(groupes)["correct"].mean()}).reset_index()
    comport["IES (ms)"] = comport["TR (ms)"] / (comport["Exactitude (%)"] / 100)
    return comport[["participant", "condition", "TR (ms)"]] if hypothese == "tr" else comport


def erp_electrodes(e, min_epoques=20):
    """L'ERP de chaque électrode : la moyenne des seuls essais où elle est mesurée (NaN s'ils sont moins de min_epoques)."""
    import mne

    mesuree = np.array([[c not in str(r).split() for c in e.ch_names] for r in e.metadata["mauvaises"]])   # essais x électrodes
    compte = mesuree.sum(axis=0)
    moyenne = np.where(mesuree[:, :, None], e.get_data(), 0.0).sum(axis=0) / np.maximum(compte, 1)[:, None]
    moyenne[compte < min_epoques] = np.nan
    return mne.EvokedArray(moyenne, e.info, tmin=e.tmin, nave=len(e))


def effet_par_posture(epoques, effet, electrodes, fenetre, min_epoques=20):
    """L'effet (partie 5) dans chaque posture, pour chaque participant qui a les deux postures : effet = 'frequence' (LF - HF) ou
    'lexicalite' (pseudo-mots - mots), dans la zone (la moyenne des électrodes mesurées) et à chaque électrode.
    Rend le tableau participants x [assis, marche] (l'amplitude moyenne de l'effet dans la fenêtre, en µV), et les ondes et les
    cartes de l'effet, {posture: liste d'Evoked}."""
    import mne

    types_a, types_b = (["HF"], ["LF"]) if effet == "frequence" else (["HF", "LF"], ["pseudo"])
    ondes, cartes, lignes = {"assis": [], "marche": []}, {"assis": [], "marche": []}, []
    for participant, ep in epoques.items():
        signal, n = region_mesuree(ep, electrodes)
        zone = mne.EpochsArray(signal[n > 0, None] * 1e-6, mne.create_info(["zone"], ep.info["sfreq"], "eeg"), events=ep.events[n > 0],
                               tmin=ep.tmin, event_id=ep.event_id, metadata=ep.metadata[n > 0], baseline=None, on_missing="ignore")
        onde, carte = {}, {}
        for posture in ("assis", "marche"):
            choix_a, choix_b = [f"{posture}/{t}" for t in types_a], [f"{posture}/{t}" for t in types_b]
            if min(len(zone[choix_a]), len(zone[choix_b])) < min_epoques:
                break
            onde[posture] = mne.combine_evoked([zone[choix_b].average(), zone[choix_a].average()], weights=[1, -1])
            carte[posture] = mne.combine_evoked([erp_electrodes(ep[choix_b]), erp_electrodes(ep[choix_a])], weights=[1, -1])
        else:                                                    # seulement les participants qui ont les deux postures
            for posture in onde:
                ondes[posture].append(onde[posture])
                cartes[posture].append(carte[posture])
            lignes.append({"participant": participant,
                           **{posture: e.copy().crop(*fenetre).get_data(units="uV").mean() for posture, e in onde.items()}})
    return pd.DataFrame(lignes).set_index("participant"), ondes, cartes


def effet_comportement_par_posture(essais, effet, comportement):
    """Le même effet (partie 5) sur le comportement : B - A dans chaque posture, pour chaque participant qui a les deux.
    comportement : 'TR (ms)', 'Exactitude (%)' ou 'IES (ms)'."""
    types_a, types_b = (["HF"], ["LF"]) if effet == "frequence" else (["HF", "LF"], ["pseudo"])
    choisis = essais[essais["type"].isin(types_a + types_b)]
    choisis = choisis.assign(condition=np.where(choisis["type"].isin(types_b), "B", "A"))
    groupes = ["participant", "posture", "condition"]
    t = pd.DataFrame({"TR (ms)": choisis[choisis["tr_valide"]].groupby(groupes)["tr_ms"].mean(),
                      "Exactitude (%)": 100 * choisis.groupby(groupes)["correct"].mean()})
    t["IES (ms)"] = t["TR (ms)"] / (t["Exactitude (%)"] / 100)
    par_condition = t[comportement].unstack("condition")
    return (par_condition["B"] - par_condition["A"]).unstack("posture")[["assis", "marche"]].dropna()


def figure_difference(difference, fenetre, amas, p_amas, zero, titre, passe_bas=20.0):
    """L'onde de différence B - A (F3) : chaque participant en gris clair, la moyenne du groupe en noir et son IC à 95 %, la
    fenêtre de l'hypothèse en bleu clair, et chaque amas (2.2) avec son p : en vert s'il est significatif (p < .05), en gris
    sinon. Pour l'affichage seulement, les ondes passent par un passe-bas à passe_bas Hz ; le test utilise les données telles
    quelles (1-40 Hz)."""
    import matplotlib.pyplot as plt
    import mne
    from scipy import stats

    temps = difference[0].times
    X = np.array([d.data[0] for d in difference]) * 1e6                       # participants x instants (µV)
    if passe_bas:
        X = mne.filter.filter_data(X, difference[0].info["sfreq"], None, passe_bas, verbose=False)
    n = len(X)
    moyenne, ic = X.mean(axis=0), stats.t.ppf(0.975, n - 1) * X.std(axis=0, ddof=1) / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(*fenetre, color="#DBEAFE", zorder=0, label="la fenêtre de l'hypothèse")
    for masque, p in zip(amas, p_amas):
        debut, fin = temps[masque][0], temps[masque][-1]
        ax.axvspan(debut, fin, color="#86EFAC" if p < 0.05 else "0.88", alpha=0.7, zorder=0)
        ax.text((debut + fin) / 2, 1.01, f"amas\np = {p:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color="#15803D" if p < 0.05 else "0.45")
    ax.plot(temps, X.T, color="0.82", lw=0.6, zorder=1)
    ax.fill_between(temps, moyenne - ic, moyenne + ic, color="0.25", alpha=0.25, lw=0, zorder=2)
    ax.plot(temps, moyenne, color="k", lw=2.2, zorder=3, label=f"la moyenne des {n} participants (IC à 95 %)")
    ax.plot([], [], color="0.82", lw=1, label="chaque participant")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    limite = np.percentile(np.abs(X), 99)
    ax.set(xlim=(temps[0], temps[-1]), ylim=(-limite, limite), xlabel=f"Temps depuis {zero} (s)", ylabel="B − A (µV)")
    ax.set_title(titre, loc="left", pad=22)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    if passe_bas:
        ax.text(1.0, -0.2, f"affichage : passe-bas à {passe_bas:.0f} Hz (le test utilise les données de 1 à 40 Hz)",
                transform=ax.transAxes, ha="right", fontsize=8, color="0.45")
    plt.tight_layout()
    plt.show()


# =====================================================================================================
# Notebook 3 (rythmes) : les calculs
# =====================================================================================================
REGION_FC = ["F3", "Fz", "F4", "C3", "Cz", "C4"]          # fronto-centrale : le thêta précoce
REGION_PO = ["P3", "Pz", "P4", "O1", "O2"]                 # pariéto-occipitale : l'alpha
REGION_C = ["C3", "Cz", "C4"]                              # centrale, au-dessus des mains : le bêta avant la tape
MESURES_TF = {   # mesure : (la bande en Hz, la fenêtre en s, la région) ; '(tape)' : autour de la tape
    "thêta précoce": ((4, 7), (0.1, 0.3), REGION_FC),
    "thêta": ((4, 7), (0.3, 0.5), REGION_CP),
    "thêta tardif": ((4, 7), (0.5, 0.8), REGION_CP),
    "alpha": ((8, 12), (0.3, 0.8), REGION_PO),
    "bêta (tape)": ((13, 30), (-0.5, 0.0), REGION_C),
}
FREQUENCES_TF = np.arange(3, 31)                           # de 3 à 30 Hz, par pas de 1 Hz
N_CYCLES_TF = 3 * (FREQUENCES_TF / 3) ** 0.523            # de 3 cycles à 3 Hz à 10 cycles à 30 Hz
DECIM_TF = 2                                               # une valeur toutes les 20 ms : la puissance change lentement
LARGEUR_CLOCHE = 2 * np.sqrt(2 * np.log(2))                # la largeur à mi-hauteur d'une cloche, en écarts-types
SITUATIONS = ["repos", "tâche assis", "marche seule", "tâche en marche"]
COMPARAISONS_SITUATIONS = {"tâche assis - repos": ("tâche assis", "repos"),
                           "tâche en marche - marche seule": ("tâche en marche", "marche seule"),
                           "marche seule - repos": ("marche seule", "repos")}


def conditions_tf(hypothese, posture="les deux"):
    """Les deux conditions, A et B (l'effet = B - A), et CONDITIONS : {condition: (ses postures, ses types de mots)}."""
    tous = ["HF", "LF", "pseudo"]
    if hypothese == "posture":
        return "assis", "marche", {"assis": (["assis"], tous), "marche": (["marche"], tous)}
    postures = ["assis", "marche"] if posture == "les deux" else [posture]
    a, b, types_a, types_b = {"frequence": ("HF", "LF", ["HF"], ["LF"]), "lexicalite": ("mots", "pseudo-mots", ["HF", "LF"], ["pseudo"]),
                              "tr": ("rapides", "lents", tous, tous)}[hypothese]
    return a, b, {a: (postures, types_a), b: (postures, types_b)}


def ajouter_vitesse(*dictionnaires):
    """Pour 'tr' : la colonne vitesse de chaque époque, 'lents' si le TR dépasse la médiane du participant (dans sa posture et son
    type de mot), 'rapides' sinon."""
    for epoques in dictionnaires:
        for ep in epoques.values():
            mediane = ep.metadata.groupby(["posture", "type"])["tr_ms"].transform("median")
            ep.metadata = ep.metadata.assign(vitesse=np.where(ep.metadata["tr_ms"] > mediane, "lents", "rapides"))


def mesurees_tf(ep):
    """Époques x électrodes : True là où l'électrode est une mesure dans l'époque (pas dans la colonne mauvaises)."""
    return np.array([[c not in str(r).split() for c in ep.ch_names] for r in ep.metadata["mauvaises"]]).reshape(len(ep), len(ep.ch_names))


def sans_tache_de(sans_tache, enregistrements, participant, posture):
    """Les segments de l'enregistrement sans tâche de la posture : la marche seule, ou le repos voisin des blocs assis (celui du début
    si les blocs assis viennent en premier, celui de la fin sinon ; l'autre s'il manque). None s'il n'y en a pas."""
    if posture == "marche":
        return sans_tache.get((participant, "marche_seule"))
    ordre = enregistrements.loc[enregistrements["participant"] == participant, "ordre"]
    voisin, autre = ("repos_debut", "repos_fin") if len(ordre) and ordre.iloc[0] == "assis d'abord" else ("repos_fin", "repos_debut")
    return sans_tache.get((participant, voisin), sans_tache.get((participant, autre)))


def spectre(ep, tmin=None, tmax=None, min_epoques=20):
    """Électrodes x fréquences (µV²/Hz) : le spectre de Welch de chaque segment de 2 s (fenêtre de Hann), moyenné sur les segments où
    chaque électrode est mesurée (NaN si elle l'est dans moins de min_epoques) ; et les fréquences."""
    psd = ep.compute_psd(method="welch", fmin=1.0, fmax=40.0, tmin=tmin, tmax=tmax, n_fft=200, n_per_seg=200, window="hann", exclude=[])
    mesure = mesurees_tf(ep)
    s = (psd.get_data(exclude=[]) * mesure[:, :, None]).sum(axis=0) / np.maximum(mesure.sum(axis=0), 1)[:, None] * 1e12
    s[mesure.sum(axis=0) < min_epoques] = np.nan
    return s, psd.freqs


def spectres_situations(epochs, sans_tache, enregistrements, min_epoques=20):
    """Le spectre de chaque participant dans les quatre situations (partie 1) : le repos et la marche seule (les enregistrements sans
    tâche), la tâche assis et la tâche en marche (2 s par essai, de -0,5 à 1,5 s autour du mot). Rend {(participant, situation):
    électrodes x fréquences (µV²/Hz)}, les fréquences, et les participants qui ont les quatre situations."""
    spectres, freqs = {}, None
    for participant, ep in epochs.items():
        for posture, sans, avec in (("assis", "repos", "tâche assis"), ("marche", "marche seule", "tâche en marche")):
            segments = sans_tache_de(sans_tache, enregistrements, participant, posture)
            if segments is not None:
                spectres[(participant, sans)], freqs = spectre(segments, min_epoques=min_epoques)
            if len(ep[posture]) >= min_epoques:
                spectres[(participant, avec)], _ = spectre(ep[posture], tmin=-0.5, tmax=1.49, min_epoques=min_epoques)
    quatre = [p for p in epochs if all((p, s) in spectres for s in SITUATIONS)]
    return spectres, freqs, quatre


def puissance_situations(spectres, participants, j, dans_bande):
    """Participants x situations (µV²/Hz) : la puissance de la bande dans la région (électrodes j), sans un participant dont la région
    n'a aucune électrode mesurée."""
    return pd.DataFrame({s: [np.nanmean(spectres[(p, s)][j][:, dans_bande]) for p in participants] for s in SITUATIONS},
                        index=participants).dropna()


def comparer_situations(valeurs):
    """Les comparaisons de la partie 1 (participants x situations) : chaque différence contre 0 (test t, dz), et l'interaction
    (ce que la tâche change en marche, moins ce qu'elle change assis). Une ligne par comparaison."""
    from scipy import stats

    differences = {nom: valeurs[a] - valeurs[b] for nom, (a, b) in COMPARAISONS_SITUATIONS.items()}
    differences["interaction (marche - assis)"] = differences["tâche en marche - marche seule"] - differences["tâche assis - repos"]
    lignes = []
    for nom, d in differences.items():
        t, p = stats.ttest_1samp(d, 0.0)
        lignes.append({"comparaison": nom, "moyenne": d.mean(), "t": t, "ddl": len(d) - 1, "p": p, "dz": d.mean() / d.std(ddof=1)})
    return pd.DataFrame(lignes).set_index("comparaison").round(3)


def pente_et_bosse(spectres, participants, freqs, j, dans_bande, ajustement=(3, 30)):
    """La pente et la bosse (1.3) : specparam ajuste à chaque spectre de la région une droite (en log-log) et des bosses. Une ligne par
    participant et situation : la pente, ce qui dépasse la droite dans la bande (dB), le pic alpha (Hz) et la qualité (R²)."""
    from specparam import SpectralGroupModel

    lignes = []
    for situation in SITUATIONS:
        spectres_region = np.array([np.nanmean(spectres[(p, situation)][j], axis=0) for p in participants])   # participants x fréquences
        modele = SpectralGroupModel(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1, aperiodic_mode="fixed", verbose=False)
        modele.fit(freqs, spectres_region, list(ajustement))
        decalage, pente = modele.get_params("aperiodic").T       # la droite : log10 P = décalage - pente x log10 f
        droite = decalage[:, None] - pente[:, None] * np.log10(freqs[dans_bande])
        au_dessus = 10 * (np.log10(spectres_region[:, dans_bande]) - droite).mean(axis=1)
        pics = modele.get_params("peak")                         # fréquence, hauteur, largeur de chaque bosse, puis le n° du spectre
        for k, participant in enumerate(participants):
            alpha = pics[(pics[:, -1] == k) & (pics[:, 0] >= 7) & (pics[:, 0] <= 14)]
            lignes.append({"participant": participant, "situation": situation, "pente": pente[k], "au-dessus de la pente (dB)": au_dessus[k],
                           "pic alpha (Hz)": alpha[np.argmax(alpha[:, 1]), 0] if len(alpha) else np.nan, "R²": modele.get_metrics("gof")[k]})
    return pd.DataFrame(lignes)


def repos_debut_fin(sans_tache, enregistrements, participants, noms_ch, region, dans_bande):
    """Le repos de la fin moins le repos du début (1.4), dans la bande et la région (les électrodes mesurées dans les deux)."""
    lignes = []
    for participant in participants:
        if (participant, "repos_debut") in sans_tache and (participant, "repos_fin") in sans_tache:
            s = {nom: spectre(sans_tache[(participant, nom)])[0] for nom in ("repos_debut", "repos_fin")}
            k = [noms_ch.index(c) for c in region if all(np.isfinite(s[nom][noms_ch.index(c)]).all() for nom in s)]
            if k:
                lignes.append({"participant": participant,
                               "ordre": enregistrements.loc[enregistrements["participant"] == participant, "ordre"].iloc[0],
                               "début (dB)": 10 * np.log10(s["repos_debut"][k][:, dans_bande].mean()),
                               "fin (dB)": 10 * np.log10(s["repos_fin"][k][:, dans_bande].mean())})
    repos = pd.DataFrame(lignes).set_index("participant")
    repos["fin - début (dB)"] = repos["fin (dB)"] - repos["début (dB)"]
    return repos


# ---- le temps-fréquence (partie 2)
def _morlet(x, sfreq, n_cycles):
    import mne
    return mne.time_frequency.tfr_array_morlet(x, sfreq, FREQUENCES_TF, n_cycles=n_cycles, output="complex", decim=DECIM_TF) * 1e6


def puissance_morlet(ep, n_cycles=N_CYCLES_TF):
    """Époques x électrodes x fréquences x instants (µV²) : la puissance de Morlet de chaque époque."""
    import mne
    return mne.time_frequency.tfr_array_morlet(ep.get_data(), ep.info["sfreq"], FREQUENCES_TF, n_cycles=n_cycles, output="power",
                                               decim=DECIM_TF) * 1e12


def _reference(segments, n_cycles, min_epoques):
    """Électrodes x fréquences (µV²) : la puissance de Morlet de l'enregistrement sans tâche, moyennée sur chaque segment de 2 s loin
    de ses bords (à plus de 3 écarts-types de l'enveloppe), puis sur les segments où chaque électrode est mesurée."""
    sigma_t = n_cycles / (2 * np.pi * FREQUENCES_TF)
    p_ = puissance_morlet(segments, n_cycles)
    t = segments.times[::DECIM_TF]
    loin = (t[None, :] >= t[0] + 3 * sigma_t[:, None]) & (t[None, :] <= t[-1] - 3 * sigma_t[:, None])
    milieu = (p_ * loin).sum(axis=-1) / loin.sum(axis=-1)
    mesure = mesurees_tf(segments)
    r = (milieu * mesure[:, :, None]).sum(axis=0) / np.maximum(mesure.sum(axis=0), 1)[:, None]
    r[mesure.sum(axis=0) < min_epoques] = np.nan
    return r


def calculer_tf(epoques, hypothese, sans_tache, enregistrements, n_cycles=N_CYCLES_TF, min_epoques=20):
    """Le temps-fréquence de chaque participant (2.1) : pour chaque participant, posture et type de mot (et vitesse, pour 'tr'), la
    somme des puissances de Morlet, le nombre d'époques où chaque électrode est mesurée, et la somme des phases (2.3) ; et la
    référence de chaque posture (l'enregistrement sans tâche). Rend un dictionnaire, TF, que lisent les fonctions suivantes."""
    groupes_de = ["posture", "type"] + (["vitesse"] if hypothese == "tr" else [])
    cellules, references = {}, {}
    for participant, ep in epoques.items():
        for posture in ("assis", "marche"):
            e = ep[posture]
            if len(e) < min_epoques:
                continue
            x, mesure = e.get_data(), mesurees_tf(e)
            for groupe, k in e.metadata.reset_index(drop=True).groupby(groupes_de).indices.items():
                w, m = _morlet(x[k], e.info["sfreq"], n_cycles), mesure[k][:, :, None, None]
                phase = w / np.maximum(np.abs(w), 1e-30)               # un nombre complexe de module 1 (2.3)
                cellules[(participant,) + groupe] = (((w * w.conj()).real * m).sum(axis=0), mesure[k].sum(axis=0), (phase * m).sum(axis=0))
            segments = sans_tache_de(sans_tache, enregistrements, participant, posture)
            if segments is not None:
                references[(participant, posture)] = _reference(segments, n_cycles, min_epoques)
    temps = epoques[next(iter(epoques))].times[::DECIM_TF]
    sigma_t = n_cycles / (2 * np.pi * FREQUENCES_TF)
    bord = (temps[None, :] < temps[0] + 3 * sigma_t[:, None]) | (temps[None, :] > temps[-1] - 3 * sigma_t[:, None])   # l'ondelette déborde
    return {"cellules": cellules, "references": references, "temps": temps, "sigma_t": sigma_t, "bord": bord, "n_cycles": n_cycles,
            "min_epoques": min_epoques, "hypothese": hypothese}


def carte_tf(TF, participant, posture, types, vitesse=None):
    """Électrodes x fréquences x instants (µV²) : la puissance moyenne des époques de la posture et des types (et de la vitesse),
    électrode par électrode sur les époques où elle est mesurée ; None s'il n'y a pas assez d'époques."""
    cellules, minimum = TF["cellules"], TF["min_epoques"]
    cles = [c for c in cellules if c[:2] == (participant, posture) and c[2] in types and (vitesse is None or c[3] == vitesse)]
    if not cles:
        return None
    somme, n = sum(cellules[c][0] for c in cles), sum(cellules[c][1] for c in cles)
    if n.max() < minimum:
        return None
    moyenne = somme / np.maximum(n, 1)[:, None, None]
    moyenne[n < minimum] = np.nan
    return moyenne


def _puissance_condition(TF, conditions, participant, posture, condition):
    return carte_tf(TF, participant, posture, conditions[condition][1], condition if TF["hypothese"] == "tr" else None)


def _en_db(TF, conditions, participant, condition):
    """La condition rapportée à l'enregistrement sans tâche de sa posture (dB ; la moyenne des postures de la condition)."""
    valeurs = []
    for posture in conditions[condition][0]:
        p_, r_ = _puissance_condition(TF, conditions, participant, posture, condition), TF["references"].get((participant, posture))
        if p_ is None or r_ is None:
            return None
        valeurs.append(10 * np.log10(p_ / r_[:, :, None]))
    return np.mean(valeurs, axis=0)


def _difference_db(TF, conditions, cond_a, cond_b, participant):
    """B - A (dB). Dans une même posture, le rapport des deux puissances (la référence s'annule) ; entre les postures, chacune est
    rapportée à son enregistrement sans tâche."""
    if TF["hypothese"] == "posture":
        a, b = _en_db(TF, conditions, participant, cond_a), _en_db(TF, conditions, participant, cond_b)
        return None if a is None or b is None else b - a
    valeurs = []
    for posture in conditions[cond_a][0]:
        a, b = _puissance_condition(TF, conditions, participant, posture, cond_a), _puissance_condition(TF, conditions, participant, posture, cond_b)
        if a is None or b is None:
            return None
        valeurs.append(10 * np.log10(b / a))
    return np.mean(valeurs, axis=0)


def cartes_groupe(TF, conditions, cond_a, cond_b):
    """Pour chaque participant : la carte de B - A (dB), et celles de A et de B (dB par rapport à la même posture sans la tâche)."""
    participants = sorted({c[0] for c in TF["cellules"]})
    differences = {p: d for p in participants if (d := _difference_db(TF, conditions, cond_a, cond_b, p)) is not None}
    cartes_ab = {p: (a, b) for p in participants if (a := _en_db(TF, conditions, p, cond_a)) is not None
                 and (b := _en_db(TF, conditions, p, cond_b)) is not None}
    return differences, cartes_ab


def _coherence(TF, conditions, participant, condition):
    """La cohérence de phase par paires (PPC) des époques de la condition, électrode par électrode (NaN sous min_epoques), moyennée
    sur les postures de la condition ; None s'il manque une posture."""
    cellules, valeurs = TF["cellules"], []
    for posture in conditions[condition][0]:
        cles = [c for c in cellules if c[:2] == (participant, posture) and c[2] in conditions[condition][1]
                and (TF["hypothese"] != "tr" or c[3] == condition)]
        if not cles:
            return None
        s, n = sum(cellules[c][2] for c in cles), sum(cellules[c][1] for c in cles)[:, None, None].astype(float)
        ppc = (np.abs(s) ** 2 - n) / np.maximum(n * (n - 1), 1)
        ppc[n[:, 0, 0] < TF["min_epoques"]] = np.nan
        valeurs.append(ppc)
    return np.mean(valeurs, axis=0)


def coherences_groupe(TF, conditions, cond_a, cond_b):
    """Pour chaque participant : la PPC de A et de B (2.3)."""
    participants = sorted({c[0] for c in TF["cellules"]})
    return {p: (a, b) for p in participants if (a := _coherence(TF, conditions, p, cond_a)) is not None
            and (b := _coherence(TF, conditions, p, cond_b)) is not None}


def dans_la_region(c, j):
    """Fréquences x instants : la moyenne des électrodes de la région (numéros j) qui ont une valeur."""
    return np.nanmean(c[j], axis=0)


def dans_le_cadre(c, dans_bande, dans_fenetre):
    """Électrodes : la moyenne d'une carte dans la bande et la fenêtre (NaN : pas mesurée)."""
    return c[:, dans_bande][:, :, dans_fenetre].mean(axis=(1, 2))


def amas_tf(D, bord, montres, seed=97):
    """Le test par amas sur des cartes participants x fréquences x instants (F3, 2.3) : sans les bords, sur les instants montrés.
    Rend les amas, leurs p, le masque des amas à p < .05, et les cartes testées."""
    from mne.stats import permutation_cluster_1samp_test
    from scipy import stats

    X = np.where(bord, 0.0, D)[:, :, montres]
    _, amas, p_amas, _ = permutation_cluster_1samp_test(X, threshold=stats.t.ppf(0.975, len(X) - 1), n_permutations=1024, tail=0,
                                                        out_type="mask", seed=seed)
    significatif = np.zeros(X.shape[1:], bool)
    for masque, p in zip(amas, p_amas):
        if p < 0.05:
            significatif |= masque
    return amas, p_amas, significatif, X


def decrire_amas(amas, p_amas, X, temps, cond_b, quoi="puissance"):
    """Les trois amas les plus nets, en une ligne : leurs fréquences, leurs instants, leur sens et leur p."""
    if not len(amas):
        return "Amas : aucun"
    plus_nets = sorted(range(len(amas)), key=lambda k: p_amas[k])[:3]
    return "Amas : " + "; ".join(f"{FREQUENCES_TF[amas[k].any(axis=1)].min()}-{FREQUENCES_TF[amas[k].any(axis=1)].max()} Hz, "
                                 f"{temps[amas[k].any(axis=0)][0] * 1000:.0f} à {temps[amas[k].any(axis=0)][-1] * 1000:.0f} ms, "
                                 f"{'plus' if X[:, amas[k]].mean() > 0 else 'moins'} de {quoi} pour {cond_b} (p = {p_amas[k]:.3f})"
                                 for k in plus_nets)


def mesures_participants_tf(TF, conditions, cond_a, cond_b, differences, cartes_ab, j, dans_bande, dans_fenetre):
    """Votre mesure pour chaque participant (partie 4) : A et B (dB par rapport à la même posture sans la tâche), B - A (dB), et
    B - A sans rapport (µV² ; pour la posture, chaque posture moins son enregistrement sans tâche)."""
    def mesure(c):
        return np.nanmean(dans_le_cadre(c, dans_bande, dans_fenetre)[j]) if c is not None else np.nan

    def difference_uv2(participant):
        valeurs = {}
        for condition in (cond_a, cond_b):
            par_posture = []
            for posture in conditions[condition][0]:
                p_ = _puissance_condition(TF, conditions, participant, posture, condition)
                if p_ is None:
                    return np.nan
                if TF["hypothese"] == "posture":
                    if (participant, posture) not in TF["references"]:
                        return np.nan
                    p_ = p_ - TF["references"][(participant, posture)][:, :, None]
                par_posture.append(mesure(p_))
            valeurs[condition] = np.mean(par_posture)
        return valeurs[cond_b] - valeurs[cond_a]

    participants = list(differences)
    return pd.DataFrame({cond_a: {p: mesure(cartes_ab[p][0]) if p in cartes_ab else np.nan for p in participants},
                         cond_b: {p: mesure(cartes_ab[p][1]) if p in cartes_ab else np.nan for p in participants},
                         f"{cond_b} - {cond_a}": {p: mesure(differences[p]) for p in participants},
                         f"{cond_b} - {cond_a} (µV²)": {p: difference_uv2(p) for p in participants}})


def effet_par_posture_tf(TF, effet, j, dans_bande, dans_fenetre):
    """L'effet (partie 5) dans chaque posture, pour chaque participant qui a les deux : le rapport des puissances des deux types de
    mots (dB), dans votre cadre. effet = 'frequence' (LF - HF) ou 'lexicalite' (pseudo-mots - mots)."""
    types_a, types_b = (["HF"], ["LF"]) if effet == "frequence" else (["HF", "LF"], ["pseudo"])
    lignes = []
    for participant in sorted({c[0] for c in TF["cellules"]}):
        effets = {}
        for posture in ("assis", "marche"):
            a, b = carte_tf(TF, participant, posture, types_a), carte_tf(TF, participant, posture, types_b)
            if a is not None and b is not None:
                effets[posture] = np.nanmean(dans_le_cadre(10 * np.log10(b / a), dans_bande, dans_fenetre)[j])
        if len(effets) == 2:
            lignes.append({"participant": participant, **effets})
    return pd.DataFrame(lignes).set_index("participant")


def figure_essais_tf(TF, epoques, conditions, cond_a, cond_b, participant, j, dans_bande, centre, zero, affichage, bande):
    """La puissance de la bande, essai par essai, chez un participant (2.2) : A en bas, B en haut, chacune du TR le plus court au plus
    long ; le trait noir : l'autre événement (la tape, ou le mot)."""
    import matplotlib.pyplot as plt
    import mne

    lignes, valeurs = [], []
    for condition, (postures, types) in conditions.items():
        for posture in postures:
            e = epoques[participant][posture]
            choix = np.isin(e.metadata["type"], types)
            if TF["hypothese"] == "tr":
                choix &= (e.metadata["vitesse"] == condition).to_numpy()
            e, r_ = e[choix], TF["references"].get((participant, posture))
            if r_ is None or len(e) == 0:
                continue
            db_ = 10 * np.log10(puissance_morlet(e, TF["n_cycles"])[:, j][:, :, dans_bande] / r_[j][:, dans_bande][None, :, :, None]).mean(axis=2)
            m = mesurees_tf(e)[:, j] & np.isfinite(r_[j]).all(axis=1)
            valeurs.append(np.where(m.any(axis=1)[:, None], np.where(m[:, :, None], db_, 0.0).sum(axis=1) / np.maximum(m.sum(axis=1), 1)[:, None],
                                    np.nan))
            lignes.append(e.metadata.assign(condition=condition))
    if not valeurs:
        raise ValueError(f"{participant} n'a pas d'essais ou d'enregistrement sans tâche pour ces conditions : choisissez un autre "
                         "PARTICIPANT (cellule 0.4)")
    X, md = np.concatenate(valeurs), pd.concat(lignes, ignore_index=True)
    garde = ~np.isnan(X).any(axis=1)
    info = mne.create_info(["région"], 100 / DECIM_TF, "misc")
    essais = mne.EpochsArray(X[garde][:, None, :], info, tmin=TF["temps"][0], metadata=md[garde].reset_index(drop=True))
    essais.crop(*affichage)
    tr = essais.metadata["tr_ms"].to_numpy() / 1000
    autre = tr if centre == "mot" else -tr
    autre = np.where((autre >= affichage[0]) & (autre <= affichage[1]), autre, np.nan)
    ordre = np.lexsort((tr, (essais.metadata["condition"] == cond_b).to_numpy()))
    lim = np.nanpercentile(np.abs(essais.get_data()), 95)
    figure = mne.viz.plot_epochs_image(essais, picks="région", order=ordre, overlay_times=autre, sigma=3, vmin=-lim, vmax=lim,
                                       units=dict(misc="dB"), scalings=dict(misc=1), cmap="RdBu_r",
                                       title=f"{participant}, {bande[0]}-{bande[1]} Hz : {cond_a} en bas, {cond_b} en haut", show=False)[0]
    figure.axes[0].set_ylabel("Essais")
    figure.axes[1].set(xlabel=f"Temps depuis {zero} (s)", ylabel="dB")
    plt.show()
    print(f"{participant} : {int((essais.metadata['condition'] == cond_a).sum())} essais {cond_a}, "
          f"{int((essais.metadata['condition'] == cond_b).sum())} essais {cond_b}")


# =====================================================================================================
# Figures du notebook 3 (rythmes)
# =====================================================================================================
def figure_spectres(spectres, participants, freqs, noms_ch):
    """F1 : le spectre moyen (dB) des quatre situations (repos, tâche assis, marche seule, tâche en marche), dans trois régions.
    spectres : (participant, situation) -> électrodes x fréquences (µV²/Hz)."""
    import matplotlib.pyplot as plt

    regions = {"frontale": ["F3", "Fz", "F4"], "centrale": ["C3", "Cz", "C4"], "pariéto-occipitale": ["P3", "Pz", "P4", "O1", "O2"]}
    styles = {"repos": ("#2f6db5", "--"), "tâche assis": ("#2f6db5", "-"), "marche seule": ("#e0791d", "--"),
              "tâche en marche": ("#e0791d", "-")}
    bandes = {"thêta": (4, 7), "alpha": (8, 12), "bêta": (13, 30)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, (nom_region, region) in zip(axes, regions.items()):
        j = [noms_ch.index(c) for c in region]
        for situation, (couleur, trait) in styles.items():
            db_ = np.array([10 * np.log10(np.nanmean(spectres[(p, situation)][j], axis=0)) for p in participants])
            ax.plot(freqs, np.nanmean(db_, axis=0), color=couleur, ls=trait, label=situation)
        for bande, (f0, f1) in bandes.items():
            ax.axvspan(f0, f1, color="0.93", zorder=0)
            ax.text((f0 + f1) / 2, 0.98, bande, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        ax.set(title=f'{nom_region} ({", ".join(region)})', xlabel="Fréquence (Hz)")
    axes[0].set_ylabel("Puissance (dB, 10 log10 µV²/Hz)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"F1 — Le spectre avec et sans la tâche ({len(participants)} participants ; tirets : sans la tâche)")
    plt.tight_layout()
    plt.show()


def figure_bande_tache(valeurs, bande, region):
    """F2 : votre bande avec et sans la tâche, dans chaque posture (valeurs : participants x situations, en dB), puis ce que la
    tâche change dans chaque posture."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for posture, (sans, avec), couleur in (("assis", ("repos", "tâche assis"), "#2f6db5"),
                                            ("en marche", ("marche seule", "tâche en marche"), "#e0791d")):
        axes[0].plot([0, 1], valeurs[[sans, avec]].T.to_numpy(), color=couleur, alpha=0.25, lw=0.8)   # un trait par participant
        axes[0].errorbar([0, 1], valeurs[[sans, avec]].mean(), yerr=valeurs[[sans, avec]].sem(), color=couleur, marker="o", lw=2,
                         capsize=3, label=posture)
    axes[0].set(xticks=[0, 1], xticklabels=["sans la tâche", "avec la tâche"], xlim=(-0.3, 1.3), ylabel="Puissance (dB)",
                title=f'{bande[0]}-{bande[1]} Hz, {", ".join(region)} (moyenne ± erreur type)')
    axes[0].legend()
    effets = pd.DataFrame({"assis": valeurs["tâche assis"] - valeurs["repos"],
                           "en marche": valeurs["tâche en marche"] - valeurs["marche seule"]})
    axes[1].bar([0, 1], effets.mean(), color=["#2f6db5", "#e0791d"], alpha=0.3)
    axes[1].plot([0, 1], effets.T.to_numpy(), color="0.2", marker="o", lw=0.8)
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set(xticks=[0, 1], xticklabels=["assis", "en marche"], ylabel="Avec - sans la tâche (dB)", title="Ce que la tâche change")
    plt.tight_layout()
    plt.show()


def figure_tf(cartes, titres, limites, masques, temps, freqs, info_region, nave, fenetre, bande, zero, tr_median, unite, titre):
    """Trois cartes temps-fréquence côte à côte (condition A, condition B, B - A) dans la région (F3 et 2.3) : le cadre de votre
    mesure, le zéro, l'autre événement au TR médian (tirets) et, sur la différence, le contour des amas à p < .05."""
    import matplotlib.pyplot as plt
    import mne

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    for ax, t_, valeurs, lim, masque in zip(axes, titres, cartes, limites, masques):
        deja = len(fig.axes)
        mne.time_frequency.AverageTFRArray(info_region, valeurs[None], temps, freqs.astype(float), nave=nave).plot(
            picks="région", vlim=(-lim, lim), cmap="RdBu_r", axes=ax, mask=masque, mask_style=None if masque is None else "contour",
            show=False)
        fig.axes[deja].set_ylabel(unite)                       # la barre de couleur que MNE vient d'ajouter
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(tr_median if zero == "le mot" else -tr_median, color="k", lw=0.8, ls="--")   # la tape (ou le mot) au TR médian
        ax.add_patch(plt.Rectangle((fenetre[0], bande[0] - 0.5), fenetre[1] - fenetre[0], bande[1] - bande[0] + 1, fill=False, ec="k",
                                   lw=1.2))
        ax.set(title=t_, xlabel=f"Temps depuis {zero} (s)", ylabel="Fréquence (Hz)")
    fig.suptitle(titre)
    plt.show()


def figure_topos_tf(cartes, info_vue, info_tout, vue, noms, region, p_electrodes, titres, titre):
    """F4 : trois cartes de la tête (condition A, condition B, B - A), en dB. Points noirs : la région ; ronds blancs : p < .05 ;
    cercles vides : électrodes mesurées chez trop peu de participants."""
    import matplotlib.pyplot as plt
    import mne
    from matplotlib.colors import ListedColormap

    limites = [max(np.abs(cartes[0]).max(), np.abs(cartes[1]).max())] * 2 + [np.abs(cartes[2]).max()]
    masques = [np.isin(noms[vue], region), np.isin(noms[vue], region), p_electrodes < 0.05]
    styles = [dict(marker="o", markerfacecolor="k", markersize=4)] * 2 + [dict(marker="o", markerfacecolor="w", markeredgecolor="k",
                                                                                markersize=8)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, carte_, lim, masque, style, t_ in zip(axes, cartes, limites, masques, styles, titres):
        image, _ = mne.viz.plot_topomap(carte_, info_vue, axes=ax, show=False, vlim=(-lim, lim), cmap="RdBu_r", mask=masque,
                                        mask_params=style)
        if not vue.all():                                    # les électrodes trop peu mesurées : des cercles vides, sans valeur
            mne.viz.plot_topomap(np.zeros(len(noms)), info_tout, axes=ax, show=False, vlim=(-1, 1), cmap=ListedColormap([(0, 0, 0, 0)]),
                                 contours=0, sensors=False, mask=~vue,
                                 mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="0.35", markersize=6))
        ax.set_title(t_)
        fig.colorbar(image, ax=ax, shrink=0.7, label="dB")
    fig.suptitle(titre)
    plt.show()


def figure_mesure_tf(mesures, cond_a, cond_b, couleurs, nom):
    """F5 : votre mesure dans les deux conditions (un trait par participant) et leur différence (un point par participant)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), width_ratios=[2, 1])
    deux = mesures[[cond_a, cond_b]].dropna()
    axes[0].bar([0, 1], deux.mean(), color=[couleurs[cond_a], couleurs[cond_b]], alpha=0.3)
    axes[0].plot([0, 1], deux.T.to_numpy(), color="0.2", marker="o", lw=0.8)
    axes[0].axhline(0, color="0.5", lw=0.8)                  # 0 dB : comme sans la tâche
    axes[0].set(xticks=[0, 1], xticklabels=[cond_a, cond_b], ylabel="dB (par rapport à la même posture sans la tâche)", title=nom)
    d = mesures[f"{cond_b} - {cond_a}"].dropna()
    axes[1].bar([0], d.mean(), color="0.85")
    axes[1].scatter(np.zeros(len(d)), d, color="0.2", zorder=3)
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set(xticks=[0], xticklabels=[f"{cond_b} - {cond_a}"], ylabel="dB", title="La différence")
    plt.tight_layout()
    plt.show()


def figure_effets_posture(tableaux, nom_effet):
    """F6 : l'effet dans chaque posture, un trait par participant ; tableaux : liste de (tableau assis/marche, titre, unité)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tableaux), figsize=(5 * len(tableaux), 4.5), squeeze=False)
    for ax, (tableau, t_, unite) in zip(axes[0], tableaux):
        ax.bar([0, 1], tableau.mean(), color="0.85")
        ax.plot([0, 1], tableau.T.to_numpy(), color="0.2", marker="o")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set(xticks=[0, 1], xticklabels=["assis", "marche"], ylabel=f"effet {nom_effet} ({unite})", title=f"{t_} : effet {nom_effet}")
    plt.tight_layout()
    plt.show()


# =====================================================================================================
# Rattrapage : les notebooks 2, 3 et 4 lisent pretraite_00/ ; il est dans les données téléchargées
# =====================================================================================================
def preparer_donnees(dossier=DOSSIER):
    """Vérifie que les données (et pretraite_00/) sont là ; les télécharge sinon."""
    dossier = Path(dossier)
    if not donnees_a_jour(dossier):
        telecharger_donnees(dossier)
    return dossier / "pretraite_00"


if __name__ == "__main__":
    # python outils_brainwalk.py P010 [sortie] : prétraite un participant dans les deux postures et exporte
    participant = sys.argv[1] if len(sys.argv) > 1 else "P010"
    sortie = Path(sys.argv[2]) if len(sys.argv) > 2 else DOSSIER / "derivatives" / "pretraite_00"
    telecharger_donnees()
    resultats = []
    for posture in POSTURES:
        if any(fichier_bids(participant, posture, n, "eeg.vhdr") for n in ETIQUETTES[posture] if n.startswith("bloc")):
            resultats.append(pretraiter(participant, posture))
    bilan = exporter(resultats, sortie)
    (sortie / "resumes").mkdir(parents=True, exist_ok=True)
    (sortie / "resumes" / f"sub-{participant}_resume.json").write_text(
        json.dumps([r["resume"] for r in resultats] + [{"export": bilan["epoques"]}], indent=2, ensure_ascii=False,
                   default=str), encoding="utf-8")
    bilan["sans_tache"].to_csv(sortie / "resumes" / f"sub-{participant}_sans_tache.tsv", sep="\t", index=False)
    print(bilan["epoques"])
