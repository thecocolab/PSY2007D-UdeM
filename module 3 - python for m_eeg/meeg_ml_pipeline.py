"""
Pipeline ML simple sur caractéristiques oscillatoires (synthétique)

Objectif :
- Générer des essais (époques) avec différence de puissance alpha entre 2 classes
- Extraire des caractéristiques (puissance de bande : thêta, alpha, bêta)
- Entraîner une régression logistique et évaluer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def gen_epoch(fs=250, duree=2.0, amp_alpha=1.0, bruit_sigma=0.5):
    t = np.arange(0, duree, 1/fs)
    alpha = amp_alpha * np.sin(2*np.pi*10*t)
    bruit = bruit_sigma * np.random.randn(t.size)
    return t, alpha + bruit


def band_power_welch(x, fs, fmin, fmax):
    f, pxx = signal.welch(x, fs=fs, nperseg=int(0.5*fs))
    masque = (f >= fmin) & (f <= fmax)
    return np.trapezoid(pxx[masque], f[masque])


def extraire_features(epochs, fs):
    feats = []
    for x in epochs:
        p_theta = band_power_welch(x, fs, 4, 7)
        p_alpha = band_power_welch(x, fs, 8, 12)
        p_beta  = band_power_welch(x, fs, 13, 30)
        feats.append([p_theta, p_alpha, p_beta, p_alpha/(p_beta+1e-9)])
    return np.array(feats)


def main():
    rng = np.random.default_rng(42)
    fs = 250
    n_epochs = 400

    # Classe 0 : alpha plus faible, Classe 1 : alpha plus fort
    epochs_c0 = [gen_epoch(fs=fs, amp_alpha=1.0 + 0.2*rng.standard_normal())[1] for _ in range(n_epochs//2)]
    epochs_c1 = [gen_epoch(fs=fs, amp_alpha=2.0 + 0.2*rng.standard_normal())[1] for _ in range(n_epochs//2)]

    X_epochs = np.array(epochs_c0 + epochs_c1)
    y = np.array([0]*(n_epochs//2) + [1]*(n_epochs//2))

    # Mélanger
    idx = rng.permutation(n_epochs)
    X_epochs = X_epochs[idx]
    y = y[idx]

    # Caractéristiques de puissance
    X = extraire_features(X_epochs, fs)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion')
    plt.xlabel('Prédit'); plt.ylabel('Vrai')

    # Distribution de la puissance alpha par classe
    plt.subplot(1,2,2)
    alpha_power = X[:,1]
    plt.hist(alpha_power[y==0], bins=20, alpha=0.7, label='Classe 0')
    plt.hist(alpha_power[y==1], bins=20, alpha=0.7, label='Classe 1')
    plt.title('Puissance alpha par classe')
    plt.xlabel('Puissance (Welch 8–12 Hz)'); plt.ylabel('Compte')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

