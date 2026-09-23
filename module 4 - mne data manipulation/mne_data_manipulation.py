# # AP1 - Manipulation de données électrophysiologiques dans MNE-Python
# Ce tutoriel est traduit et adapté de [ceux de Dr. Jas](https://jasmainak.github.io/mne-workshop-brown/readme.html). N'hésitez pas à les consulter pour aller plus loin. Les tutoriels disponibles dans la [documentation de MNE-Python](https://mne.tools/stable/auto_tutorials/index.html) sont également une excellente ressource.
#
# ## Introduction à l'activité
# À travers ces 4 activités pratiques, vous apprendrez à utiliser des outils de programmation pour traiter et analyser des données électrophysiologiques. À la fin de ces tutoriels, vous aurez exploré, nettoyé et analysé des données d'entraînement en utilisant différentes approches couramment utilisées en neurosciences cognitives.
# Le premier de ces outils est [MNE-Python](https://mne.tools/stable/index.html), une bibliothèque permettant la manipulation de jeux de données électrophysiologiques, incluant la MEG, l'EEG, l'EEG intracrânien et la polysomnographie (une combinaison d'EEG, d'ECG et d'EMG).
#
# MNE-Python est un outil polyvalent pour le traitement de données cérébrales.
#
# ![eeg_swissknife](https://i.ibb.co/y0Rnw9T/eeg-swissknfife.png)
#
# Au cours du traitement, les données prennent différentes formes, des données brutes (*raw*), des segments (*epochs*), des potentiels évoqués (*evoked potentials*), des cartes temps-fréquence, et d'autres encore.
# Les objets de MNE-Python permettent d'associer les données sous différentes formes à leurs métadonnées, c'est-à-dire aux informations complémentaires à l'enregistrement qui facilitent leur lecture. Chaque objet possède des [méthodes](https://docs.python.org/3/tutorial/classes.html#method-objects) qui lui sont propres, c'est-à-dire des fonctions qui s'appliquent sur ces objets pour les transformer ou pour tracer des graphiques afin de les explorer.
#
# Au cours de cette première activité, nous allons donc découvrir comment les données M/EEG sont couramment stockées et manipulées par MNE-Python, ainsi que comment nous pouvons les explorer de manière sommaire.

# # Imports
#
# Pour commencer, nous devons installer les bibliothèques nécessaires. Si vous utilisez le notebook via Binder (en ligne), vous devez exécuter la cellule suivante. Sinon, si vous exécutez le notebook localement sur votre ordinateur, vous pouvez utiliser le fichier requirements.txt pour installer les bibliothèques. Pour ce faire, activez votre environnement et exécutez la commande suivante dans votre terminal ou votre invite de commandes :
#
#     pip install -r requirements.txt
#

# !pip install mne
# !pip install matplotlib
# !pip install pooch
# !pip install tqdm

# Maintenant, importons MNE. Pour ce tutoriel, nous importerons l'intégralité de la bibliothèque et utiliserons les noms des fonctions dans leur intégralité afin de rendre explicites les différents modules de MNE.
#

import os
import mne
import matplotlib.pyplot as plt

# Pour activer les graphiques interactifs, remplacez "%matplotlib inline" par "%matplotlib qt".
# %matplotlib inline

# ## Charger les données

# Différents formats de fichiers existent pour stocker les données électrophysiologiques. Ils peuvent être reconnus à leur extension, c'est-à-dire à la suite de caractères qui succède au point dans le nom du fichier. Chaque format a une manière différente de représenter les données et leurs métadonnées, mais la plupart du temps, le fichier est composé d'un *en-tête* (contenant les métadonnées) et d'un *corps*, contenant les données elles-mêmes (par exemple, les signaux mesurés sur les différents capteurs, le plus souvent accompagnés d'un canal encodant les événements liés à l'expérience).
# Certains formats sont plus courants que d'autres, certains sont libres, d'autres sont propriétaires, mais tous ont leurs avantages et inconvénients. En général, chaque fabricant de M/EEG a son propre format, mais heureusement, MNE-Python a les moyens de lire la plupart d'entre eux. Quelques-uns des plus courants sont :
#
# - EDF ou EDF+ (.edf) : [European Data Format(+)](https://www.edfplus.info/index.html). Format open source assez standard et polyvalent. Permet d'encoder des données EEG et MEG, et même EKG/EMG dans sa version étendue (+).
# - FIF (.fif) : ?. Alternative au format EDF avec à peu près les mêmes avantages, c'est le plus utilisé par MNE-Python.
# - SET (.set) : Format propre à [EEGLAB](https://sccn.ucsd.edu/eeglab/index.php), une autre bibliothèque d'analyse M/EEG qui fonctionne sous Matlab.
# - CTF, CNT, BDF etc... : Autres formats. Notre conseil serait que vous transformiez vos données en l'un des formats ci-dessus (FIF ou EDF), et que vous vous éloigniez dès que possible de ces formats propriétaires démoniaques.
#
# Dans notre exemple, les données sont au format FIF. Nous utiliserons donc la fonction [mne.io.read_raw_fif](https://mne.tools/stable/generated/mne.io.read_raw_fif.html#mne.io.read_raw_fif). Prenez un moment pour consulter sa documentation. D'autres fonctions sont disponibles [ici](https://mne.tools/stable/reading_raw_data.html) pour lire les autres formats.
#
# En chargeant les données, un objet [Raw](https://mne.tools/stable/generated/mne.io.Raw.html) de la bibliothèque MNE-Python sera créé.

# Crée un chemin d'accès vers le jeu de données du tutoriel. 
# Ce jeu de données sera téléchargé automatiquement lors du premier appel du fichier.
chemin_donnees = os.path.join(mne.datasets.sample.data_path(), 'MEG', 'sample', 'sample_audvis_raw.fif')

# Charge les données et crée un objet mne.io.Raw()
raw = mne.io.read_raw_fif(chemin_donnees, preload=True) 
# L'option preload=False permet d'économiser de la mémoire lorsque le chargement immédiat des données n'est pas nécessaire, mais seulement l'accès aux métadonnées

# ## Sauvgarder les données

# N'importe quel objet de données de MNE-Python peut être sauvegardé en utilisant la méthode [.save()](https://mne.tools/stable/generated/mne.io.Raw.html?highlight=save#mne.io.Raw.save). Essayons avec notre objet Raw.
# L'attribut "overwrite" permet de forcer l'enregistrement des données même si le fichier existe déjà. Il est recommandé de l'utiliser avec précaution.

data_path = os.path.join(mne.datasets.sample.data_path(), 'MEG', 'sample', 'sample_audvis_raw.fif')
raw.save(data_path, overwrite=True)

# ## Objet mne.io.[Raw](https://mne.tools/stable/generated/mne.io.Raw.html)
#
# Les objets Raw contiennent des données continues, c'est-à-dire non segmentées. Ils se composent d'une [structure d'information](https://mne.tools/stable/generated/mne.Info.html) et d'un [tableau numpy](https://numpy.org/doc/stable/reference/generated/numpy.array.html) contenant les données.
# Lorsque vous appelez un objet raw sans spécifier de méthode, sa structure d'information est affichée. Il s'agit des métadonnées du fichier. Vous y trouverez diverses informations utiles pour la suite du processus, telles que la fréquence d'échantillonnage, le nombre et le type de canaux, ainsi que la présence de filtres matériels pendant l'enregistrement, etc.
#

raw # Équivalent à raw.info

# Il est également possible d'accéder aux métadonnées en les appelant directement via le [dictionnaire](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) raw.info ou à travers les [attributs](https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces) de l'objet Raw.

# Affichez la fréquence d'échantillonnage
print(f"Fréquence d'échantillonnage : {raw.info['sfreq']} Hz")
# Affichez la taille de la matrice de données
print(f'{len(raw.ch_names)} canaux x {len(raw.times)} échantillons')

## Question 1: Quelle est la durée totale de l'enregistrement en secondes?
print(f"Durée totale de l'enregistrement : {...:.2f} secondes")
## Question 2: Quelles sont les fréquences de filtrage des données?
print(f"Fréquence de coupure haute des données : {...:.2f} Hz")
print(f"Fréquence de coupure basse des données : {...:.2f} Hz")

# De manière similaire, vous pouvez accéder au tableau contenant les données.

raw.get_data()

# Il est également possible d'indexer directement les données de l'objet Raw, de la même manière qu'un array. Cela peut permettre, par exemple, de sélectionner une portion spécifique des données.

# Extrait les données des 5 premiers canaux, de 1 seconde à 3 secondes.
sfreq = raw.info['sfreq']
start_secondes = 1  # Début de la période en secondes
end_secondes = 3  # Fin de la période en secondes

# Calcule les indices correspondant à la période spécifiée
start_indices = int(sfreq * start_secondes)
end_indices = int(sfreq * end_secondes)

# Calcule les indices correspndant aux 5 premiers canaux
start_sensor = 0
end_sensor = 5

# Extrait les données et les temps correspondants
data, times = raw[start_sensor:end_sensor, start_indices:end_indices]

# Cette partie du code trace les données des 5 premiers canaux pour la plage horaire spécifiée.
_ = plt.plot(times, data.T)
_ = plt.title('Canaux d\'échantillonnage')
_ = plt.xlabel('Temps (s)')
_ = plt.ylabel('Amplitude')
plt.show()


#Question 3: Extrait les données des canaux 5 jusqu'à 10, de 2 seconde à 4.5 secondes.
...

# En général, il est préférable de sélectionner les données en utilisant les méthodes disponibles. Étant donné que ces fonctions modifient l'objet Raw, il est judicieux de les appliquer à une copie de celui-ci si l'on souhaite préserver l'intégrité de notre objet original. Pour ce faire, on utilise la méthode .copy().
# Par exemple :

meg_only = raw.copy().pick_types(meg=True, eeg=False)

# Ces fonctions sont plus flexibles et facilitent la manipulation des données. Par exemple, [Raw.pick_types()](https://mne.tools/stable/generated/mne.io.Raw.html?highlight=raw%20pick_types#mne.io.Raw.pick_types) permet de sélectionner uniquement certains types de capteurs MEG si on les spécifie sous forme de chaînes de caractères.

grad_only = raw.copy().pick_types(meg='grad') # grad est un type de capteur MEG

# Il est également possible de spécifier les noms des canaux directement. Ces noms se trouvent généralement dans l'attribut raw.ch_names.

pick_chans = ['MEG 0112', 'MEG 0111', 'MEG 0122', 'MEG 0123']
specific_chans = raw.copy().pick_channels(pick_chans)

print(meg_only)
print(grad_only)
print(specific_chans)

meg, times_meg = meg_only[:, :int(sfreq * 2)]
grad, times_grad = grad_only[:, :int(sfreq * 2)]
spec, times_spec = specific_chans[:, :int(sfreq * 2)]

# Crée un tracé basique avec matplotlib
f, (a1, a2, a3) = plt.subplots(3, 1, sharex=True)

# Plot pour meg_only
a1.plot(times_meg, meg[0])
a1.set_ylabel('Amplitude')
a1.set_title('MEG Data')
a1.grid(True)

# Plot pour grad_only
a2.plot(times_grad, grad[0])
a2.set_ylabel('Amplitude')
a2.set_title('Gradiometer Data')
a2.grid(True)

# Plot pour specific_chans
a3.plot(times_spec, spec[0])
a3.set_xlabel('Time')
a3.set_ylabel('Amplitude')
a3.set_title('Specific Channels Data')
a3.grid(True)

plt.tight_layout()  # Pour s'assurer que les titres et les étiquettes ne se chevauchent pas
plt.show()

# Après toutes ces opérations de copie, libérons un peu de mémoire0

del meg, meg_only, grad_only, data, specific_chans, spec, grad, times_meg, times_grad, times_spec

#Question 4: Créez un tracé similaire pour les canaux EEG seulement et pour canaux EEG: EEG 001, EEG 002, EEG 003, EEG 004.
pick_chans = ...
eeg_only = ...
specific_chans = ...

print(eeg_only)
print(specific_chans)

eeg, times_eeg = ...
spec, times_spec = ...

# Crée un tracé basique avec matplotlib
f, (a1, a2) = plt.subplots(2, 1, sharex=True)
a1.plot(...)
a1.set_xlabel('Time')
a1.set_ylabel('Amplitude')
a1.set_title('EEG Data')
a1.grid(True)

a2.plot(...)
a2.set_xlabel('Time')
a2.set_ylabel('Amplitude')
a2.set_title('Specific Channels Data')
a2.grid(True)

plt.tight_layout()  # Pour s'assurer que les titres et les étiquettes ne se chevauchent pas

# Vous pouvez aussi utiliser une méthode pour sélectionner un segment de données. Avec cette fonction, vous pouvez directement spécifier les bornes du segment en secondes.

# Recadrer les données brutes de 0 à 50 secondes
raw = raw.crop(0, 50)  # en secondes

# Afficher la nouvelle plage de temps
print(f'Nouvelle plage de temps : {raw.times.min():.2f}s à {raw.times.max():.2f}s')

# Pour supprimer un canal spécifique.
nchan = raw.info['nchan']
raw = raw.drop_channels(['MEG 0241', 'EEG 001'])
print(f'Nombre de canaux réduit de {nchan} à {raw.info["nchan"]}')

# Il est possible de "re-coller" plusieurs portions de signal entre elles. Il suffit que leurs métadonnées soient compatibles. Cela peut être utile si l'on veut retirer une partie du signal inutile pour l'analyse (par exemple des artefacts).

# Créez plusieurs objets Raw
raw1 = raw.copy().crop(0, 10)
raw2 = raw.copy().crop(10, 20)
raw3 = raw.copy().crop(20, 40)

# Concaténer dans le temps (fonctionne également sans préchargement)
raw1.append([raw2, raw3])
print(f"La plage de temps s'étend de {raw1.times.min():.2f}s à {raw1.times.max():.2f}s")

# Enfin, plusieurs méthodes permettent de visualiser les données. La plus basique est la fonction [Raw.plot()](https://mne.tools/stable/generated/mne.io.Raw.html?highlight=raw%20plot#mne.io.Raw.plot) (rappel : allez jeter un œil à la documentation, il y a beaucoup de paramètres utiles !). Elle permet de créer un tracé des données qui peut être interactif si vous avez activé le bon [*backend* de matplotlib](https://matplotlib.org/1.5.1/faq/usage_faq.html#what-is-a-backend) au début du notebook.
#
# Cette fonction est très puissante. Quelques choses à savoir :
# - Les canaux MEG sont affichés en bleu, les canaux EEG sont en noir.
# - Les canaux marqués comme *bad* sont en gris et peuvent être vus sur la barre de défilement verticale.
# - En cliquant sur les canaux, vous pouvez les marquer comme *bad* de manière interactive.
# - Les touches +/- permettent d'ajuster l'échelle des données.
# - Les paramètres d'échelle initiaux peuvent être ajustés dans les paramètres de la fonction.
# - Si vous ne connaissez pas l'échelle de votre signal, laissez les paramètres par défaut ou essayez `scalings='auto'`.
# - Avec les touches *pageup/pagedown* et *home/end*, vous pouvez ajuster la quantité de données affichées.
#

raw.plot();

#  Il est également possible de tracer le spectre du signal avec la fonction [Raw.compute_psd().plot()](https://mne.tools/stable/generated/mne.io.Raw.html?highlight=raw%20plot_psd#mne.io.Raw.plot_psd). C'est utile afin d'inspecter :
# - Le bruit de la ligne électrique
# - La présence de canaux bruités
# - Les canaux encodant les mouvements de la tête
# - Si les données ont été traitées (par exemple, filtrées)

# Créez une figure et un axe pour le tracé du spectre de puissance
fig, ax = plt.subplots(1, 1)

# Copiez les données brutes et sélectionnez uniquement les capteurs MEG de type 'mag' pour le tracé du spectre
raw.copy().pick(picks='mag').compute_psd().plot(spatial_colors=False, show=False,
                                                   axes=ax);

# Ajoutez des lignes verticales pour indiquer les fréquences de 60 Hz, 120 Hz et 180 Hz
for freq in [60., 120., 180.]:
    ax.axvline(freq, linestyle='--', alpha=0.6)

# Ajoutez des descriptions en français
ax.set_xlabel('Fréquence (Hz)')
ax.set_ylabel('Puissance du Signal (dB/Hz)')
ax.set_title('Spectre de Puissance des Capteurs MEG de Type "Mag"')

# Enfin, il est possible de sauvegarder les données à l'aide de la méthode [Raw.save()](https://mne.tools/stable/generated/mne.io.Raw.html?highlight=raw%20save#mne.io.Raw.save).
#

new_data_path = os.path.join(mne.datasets.sample.data_path(), 'MEG', 'sample', 'sample_audvis_raw_processed.fif')
raw.save(new_data_path, overwrite=True)

# ## Objet [mne.Epoch](https://mne.tools/stable/generated/mne.Epochs.html)
# Parfois, les données sont segmentées, on les appelle alors des "epochs". L'objet [mne.Epoch](https://mne.tools/stable/generated/mne.Epochs.html), tout comme l'objet Raw, est accompagné d'une structure d'information permettant de comprendre son contenu. Lorsqu'on ouvre un fichier .fif contenant des epochs avec [mne.read_epochs()](https://mne.tools/stable/generated/mne.read_epochs.html?highlight=read_epochs#mne.read_epochs), un objet Epoch est automatiquement créé.

kiloword_data_file = os.path.join(mne.datasets.kiloword.data_path(), 'kword_metadata-epo.fif')
epochs = mne.read_epochs(kiloword_data_file)

# Ici, nous utilisons le jeu de données "kiloword", car il a déjà été segmenté. Ce jeu de données a été enregistré lors d'une expérience auditive dans laquelle plusieurs mots ont été présentés à un sujet, tels que les mots "acide", "film", "froid", etc.

# Question 5: Affichez les informations sur les epochs.
...

# Question 6: Affichez la fréquence d'échantillonnage des epochs.
print(f"Fréquence d'échantillonnage : {...:.2f} Hz")
# Question 7: Affichez la taille de la matrice de données des epochs.
print(f'{...} canaux x {...} échantillons')
# Question 8: Quelle est la durée totale de l'enregistrement en secondes?
print(f"Durée totale de l'enregistrement : {...:.2f} secondes")
# Question 9: Quelles sont les fréquences de filtrage des données?
print(f"Fréquence de coupure haute des données : {...:.2f} Hz")
print(f"Fréquence de coupure basse des données : {...:.2f} Hz")

# De la même manière, il est possible de créer facilement un graphique interactif ou un spectre moyen à travers les epochs.

epochs.plot(events=False);

epochs.compute_psd().plot();

# Il est possible d'indexer les epochs individuellement ou en passant une liste à l'objet Epochs :

epochs[0]

epochs[[0,1,2,3,4]] # Équivalent à epochs[:5]

# Mais la véritable puissance des objets Epochs réside dans leur capacité à manipuler les epochs basées sur des métadonnées. Dans ce jeu de données, différents mots ont été présentés au sujet. Chaque mot est associé à une epoch, et on peut y accéder en spécifiant le mot qui nous intéresse à l'objet Epochs. Cela fonctionne également avec des conditions expérimentales, ce qui est donc très efficace pour séparer ses données avant de les comparer.

epochs['acid']

print(epochs['acid'].info)

# Question 10: Affichez les informations sur les epochs de 5 à 10.
epcohs[...]

# Question 10: Affichez les informations sur les epochs de l'événement 'cent'.
print(epochs[...].info)

# ## Objet [mne.Evoked](https://mne.tools/stable/generated/mne.Evoked.html?highlight=evoked#mne.Evoked)
# Lorsqu'on fait la moyenne des epochs, on obtient des réponses évoquées. Celles-ci sont stockées dans des objets [mne.Evoked](https://mne.tools/stable/generated/mne.Evoked.html?highlight=evoked#mne.Evoked), qui viennent également avec leurs propres méthodes. On peut lire les données évoquées depuis un fichier .fif avec [mne.read_evokeds()](https://mne.tools/stable/generated/mne.read_evokeds.html?highlight=read_evoked).

sample_data_evk_file = os.path.join(mne.datasets.sample.data_path(), 'MEG', 'sample', 'sample_audvis-ave.fif')
evokeds_list = mne.read_evokeds(sample_data_evk_file, verbose=False)
evokeds_list

# Cette fois-ci, la fonction nous retourne une liste d'objets Evoked : un par condition expérimentale, qui ont été moyennés séparément. On peut visualiser le premier potentiel évoqué (PE) de la liste (Audition Gauche) avec, encore une fois, la méthode [Evoked.plot()](https://mne.tools/stable/generated/mne.Evoked.html?highlight=evoked%20plot#mne.Evoked.plot). Notez que si vous avez activé le *backend* interactif, cet affichage vous permet de créer rapidement une topographie lorsque vous cliquez sur le signal. Essayez de sélectionner le moment auquel le potentiel est le plus visible et examinez sa topographie. Qu'en pensez-vous ?

evokeds_list[0].plot();

# La fonction [Evoked.plot_topomap()](https://mne.tools/stable/generated/mne.Evoked.html?highlight=plot_topomap#mne.Evoked.plot_topomap) permet de créer une carte topographique directement.

# Afficher la carte topographique du premier Evoked de la liste
evokeds_list[0].plot_topomap(times=[0.1, 0.2], # les temps de topo à afficher en secondes
                             ch_type='mag', # le type de capteur à afficher
                             time_unit='s', 
                             cmap='coolwarm', 
                             sensors=True,
                             sphere=0.185,
                             extrapolate='head');

# Question 11: Affichez la carte topographique du premier Evoked de la liste pour les capteurs EEG. 
# pour les temps -0.1, 0.1, 0.2, 0.3, 0.4, .
# Afficher la carte topographique du premier Evoked de la liste
evokeds_list[0].plot_topomap(times=..., 
                             ch_type=..., # le type de capteur à afficher
                             time_unit='s', 
                             cmap='coolwarm', 
                             sensors=True,
                             sphere=0.09,
                             extrapolate='head');

# # Conclusion
# Les objets Raw, Epochs et Evoked offrent de nombreuses autres options. Dans la prochaine activité, nous verrons comment prétraiter des données MEG afin de les nettoyer en vue d'une analyse des potentiels évoqués et des oscillations.

# #FIN
