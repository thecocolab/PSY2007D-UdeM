"""
Vérification de l'environnement Python — PSY2007D, automne 2026

Utilisation (dans le terminal, environnement activé) :
    python verifier_installation.py

Le script vérifie la version de Python, la présence des paquets du cours,
puis exécute un petit test MNE (création d'un signal, filtrage, spectre, epochs).
"""

import importlib
import sys

PAQUETS = [
    # (nom à importer, nom pour pip)
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("mne", "mne"),
    ("mne_bids", "mne-bids"),
    ("mne_connectivity", "mne-connectivity"),
    ("mne_denoise", "mne-denoise"),
    ("specparam", "specparam"),
    ("antropy", "antropy"),
    ("sklearn", "scikit-learn"),
    ("ipykernel", "ipykernel"),
]


def verifier_python():
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}  ({sys.executable})")
    if version < (3, 10):
        print("  PROBLÈME : Python 3.10 ou plus récent est nécessaire pour MNE. Installez Python 3.13.")
        return False
    if version[:2] != (3, 13):
        print("  Note : le cours utilise Python 3.13 ; cette version devrait fonctionner, mais n'a pas été testée.")
    return True


def verifier_paquets():
    manquants = []
    for module, nom_pip in PAQUETS:
        try:
            mod = importlib.import_module(module)
            print(f"  OK        {nom_pip:<18} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"  MANQUANT  {nom_pip}")
            manquants.append(nom_pip)
    return manquants


def tester_mne():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # pas de fenêtre graphique pendant le test
    import mne

    mne.set_log_level("ERROR")
    sfreq = 250
    t = np.arange(0, 20, 1 / sfreq)
    data = 1e-6 * (np.sin(2 * np.pi * 10 * t) + np.random.default_rng(0).standard_normal((2, t.size)))
    raw = mne.io.RawArray(data, mne.create_info(["Cz", "Pz"], sfreq, "eeg"))
    raw.filter(1, 40)
    psd = raw.compute_psd(fmax=40)
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=0.9, baseline=None, preload=True)
    pic = psd.freqs[psd.get_data().mean(axis=0).argmax()]
    print(f"  Test MNE : {len(epochs)} epochs, pic du spectre à {pic:.1f} Hz (attendu : 10 Hz)")


def main():
    print("=" * 60)
    print("Vérification de l'environnement PSY2007D")
    print("=" * 60)
    ok_python = verifier_python()
    print("\nPaquets :")
    manquants = verifier_paquets()

    if manquants:
        print("\nPaquets manquants. Dans le terminal, avec l'environnement activé :")
        print("  python -m pip install " + " ".join(manquants))
        print("ou, depuis le dossier du dépôt :")
        print("  python -m pip install -r requirements.txt")
    else:
        print("\nTest :")
        try:
            tester_mne()
        except Exception as err:
            print(f"  ÉCHEC du test MNE : {type(err).__name__}: {err}")
            manquants.append("test")

    print("\n" + ("Environnement prêt." if ok_python and not manquants
                  else "L'environnement n'est pas prêt : voir les messages ci-dessus."))


if __name__ == "__main__":
    main()
