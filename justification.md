# Journal des justifications — wmh-multisite

> Carnet de bord des choix du projet : **ce qu'on a décidé, pourquoi, sur quelles preuves, et ce qu'on a écarté**.
> Une entrée par décision, numérotée `J-xxx`, jamais supprimée : si une décision change, on ajoute une nouvelle entrée qui la **remplace** et on met à jour le statut de l'ancienne.
>
> **Statuts** : `Actée` · `Provisoire` (dépend d'un résultat attendu) · `Remplacée par J-xxx`

**Modèle d'entrée**

```
## J-xxx — Titre court
Date · Statut
Contexte : pourquoi la question se pose.
Décision : ce qu'on fait.
Alternatives écartées : quoi, et pourquoi.
Preuves : mesures, sources, chiffres.
Conséquences : ce que ça implique pour la suite.
```

---

## J-001 — Sujet : segmentation multi-sites des hyperintensités de la substance blanche (maladie des petits vaisseaux)
2026-10-02 · Actée

**Contexte.** Le projet doit appuyer une candidature d'ingénieur de recherche au CNRS (CREATIS, projet BrainTwin). L'offre exige « une première expérience en gestion et manipulation de logiciels de recherche pour l'analyse de données médicales (y compris image) ». Elle décrit 4 axes : (1) chaîne de prétraitement avec QC et harmonisation multi-sites, (2) intégration de modèles de segmentation de lésions, (3) biomarqueurs de charge lésionnelle, (4) modèles pronostiques multimodaux. Contexte pathologique : AVC et maladie des petits vaisseaux (SVD). Délai : 5 jours.

**Décision.** Chaîne complète sur les HSB (marqueur principal de la SVD) à partir d'IRM de 5 scanners, avec QC, segmentation, biomarqueurs et harmonisation.

**Alternatives écartées.**
- *Segmentation AVC + pronostic (ISLES'24)* : couvre l'axe 4, mais archive monobloc de 99 Go, 149 cas, un seul site → peu de matière pour l'axe 1. Gardé en extension.
- *Benchmark QC/harmonisation sur IXI (sujets sains)* : excellent sur l'axe 1, mais aucune lésion → axes 2-3 non couverts.
- *BraTS (tumeurs)* : données déjà prétraitées (crâne retiré, recalées, rééchantillonnées) → impossible de démontrer l'axe 1 ; pathologie hors cœur de l'offre ; sujet très saturé.

**Preuves.** Couverture des axes : HSB multi-sites = axes 1, 2, 3 + « benchmark sur cohortes publiques multi-sites » explicitement demandé.

**Conséquences.** Pas de volet pronostique (voir J-014).

---

## J-002 — Jeu de données : MICCAI WMH Segmentation Challenge 2017
2026-10-02 · Actée

**Contexte.** Il faut des IRM multi-scanners avec annotations de HSB, accessibles immédiatement et sans procédure d'accès longue.

**Décision.** Données publiées du WMH Challenge (DataverseNL, DOI [10.34894/AECRSD](https://doi.org/10.34894/AECRSD)).

**Preuves (vérifiées le 2026-10-02 via l'API et le readme).**
- 170 sujets annotés : 60 train (Utrecht Philips 3T, Singapour Siemens 3T, Amsterdam GE 3T ; 20 chacun) et 110 test (30/30/30 sur ces scanners + 10 Philips 3T Ingenuity + 10 GE 1.5T, **jamais vus en entraînement**).
- Par sujet : `orig/` (FLAIR, T1 3D brute défacée, T1 alignée sur la FLAIR par elastix, paramètres de recalage) et `pre/` (images corrigées du biais par SPM12) ; vérité terrain dans l'espace FLAIR (0 = fond, 1 = HSB, 2 = autre pathologie, ignorée à l'évaluation).
- Annotations selon les critères STRIVE, par un expert, relues par un second ; deux observateurs supplémentaires (O3, O4) sur les 60 cas du train.
- 1 791 fichiers, 8,8 Go, téléchargeables un par un via l'API sans compte.
- Licence **CC BY-NC 4.0** ; elle remplace les anciennes conditions de participation au challenge (qui interdisaient tout autre usage).
- Résolutions hétérogènes (7 combinaisons de voxels, FLAIR 2D à 3 mm sur deux sites d'entraînement), orientations différentes entre T1 3D (PIR) et FLAIR (LPS) → vraie matière pour le prétraitement.

**Alternatives écartées.** ADNI (accès sur dossier, plusieurs jours à plusieurs semaines) ; ATLAS v2.0 (AVC chronique, accès sur formulaire).

**Conséquences.** Attribution obligatoire, pas d'usage commercial, données jamais versionnées dans le dépôt.

---

## J-003 — Nom du projet : `wmh-multisite`
2026-10-03 · Actée

**Décision.** `wmh-multisite` (paquet Python `wmh_multisite`).
**Raison.** Descriptif et sobre : dit l'objet (HSB) et l'angle (multi-sites), sans promesse excessive.

---

## J-004 — Pas de données démographiques : conception du benchmark d'harmonisation
2026-10-02 · Actée

**Contexte.** Un benchmark d'harmonisation vérifie habituellement qu'on enlève l'effet du site **tout en conservant** un signal biologique (souvent l'âge). Le jeu WMH ne fournit ni âge ni sexe (rien dans le readme ni dans les fichiers).

**Décision.** Le signal biologique à conserver est la **charge lésionnelle manuelle** (vérité terrain). Mesures : précision d'un classifieur qui devine le site (doit baisser) et association des caractéristiques harmonisées avec la charge lésionnelle (doit rester).

**Limite assumée.** Le signal conservé est proche de ce qu'on mesure : c'est une validation partielle, à écrire comme telle dans le README.

---

## J-005 — Aucun calcul lourd sur la machine locale
2026-10-02 · Actée

**Contexte.** Machine locale : Windows 11, RTX 4060 Laptop (8 Go de VRAM), 24 Go de RAM, i5-12450HX, **26 Go de disque libre**, WSL2 non fonctionnel, Docker absent.

**Faits.** Deux plantages complets du PC pendant les tests de WMH-SynthSeg :
1. Sur GPU : « CUDA out of memory… 16.00 GiB is allocated » sur une carte de 8 Go → le pilote Windows déborde dans la RAM partagée au lieu d'échouer proprement.
2. Sur CPU sans recadrage : saturation probable des 24 Go de RAM (l'outil en demande ~32 Go, cf. J-007).

**Décision.**
- nnU-Net (entraînement, inférence) et WMH-SynthSeg : **jamais en local**, sur Kaggle.
- En local : au plus 2 traitements lourds en parallèle, toujours sous un watchdog mémoire (`utils/guard.py`, plafond 10 Go de RAM) ; GPU local seulement pour HD-BET, après validation de sa VRAM sur Kaggle.
- Docker : construit et testé dans GitHub Actions plutôt qu'en local.
- Téléchargement partiel pour économiser le disque (J-015).

---

## J-006 — Calcul distant : Kaggle plutôt que Colab
2026-10-02 · Actée

**Décision.** Kaggle Notebooks pour les calculs lourds (GPU T4 pour nnU-Net, CPU pour WMH-SynthSeg).

**Raisons.** Quota GPU de 30 h/semaine **garanti** (Colab : non publié, variable) ; sessions longues en arrière-plan (« Save & Run All ») sans coupure d'inactivité (Colab : 90 min) ; ~30 Go de RAM (Colab gratuit : ~12,7 Go) ; sorties conservées. Préférer T4 à P100 (architecture ancienne, support PyTorch récent incertain).

**Conséquences.** Les notebooks téléchargent les données directement depuis DataverseNL. Seuls nos prétraitements sont déposés en dataset Kaggle **privé**. Le découpage « développement local + calcul distant » ressemble à l'usage d'un supercalculateur comme Jean Zay.

**Sources.** [gpuperhour.com](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), [hivenet.com](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations) (consultés le 2026-10-02).

---

## J-007 — WMH-SynthSeg : méthode de comparaison secondaire, en mode CPU d'origine
2026-10-03 · Actée (périmètre précisé par J-020)

**Contexte.** WMH-SynthSeg (Laso et al., ISBI 2024, FreeSurfer dev) est un modèle pré-entraîné généraliste qui segmente les HSB et l'anatomie à partir de n'importe quel contraste. Il ne tient pas sur nos GPU.

**Analyse (documentation, article, code et poids inspectés).**
- Documentation officielle : « About 32GB of RAM memory » ; `--crop` est requis sur GPU (« long story ») ; périphérique par défaut : CPU.
- Article : entraînement sur des patches de 160³, mais inférence sur le volume entier rééchantillonné à 1 mm ; TTA par retournement gauche-droite.
- Poids : 66 M de paramètres, filtres 32/64 → 1024 (l'article dit « 64 feature maps per level », inexact par rapport au code), décodeur par concaténation (192 canaux à pleine résolution au dernier niveau), GroupNorm, 39 canaux de sortie. Le fichier de 790 Mo contient l'état de l'optimiseur Adam (~2/3).
- Avec `--crop` (192×224×192 = 8,3 M voxels, fp32) : pic estimé à ~15-16 Go → échec sur la T4 (~15 Go utiles). L'erreur observée (« Tried to allocate 1.97 GiB ») correspond exactement à un tenseur de 64 canaux à pleine résolution.
- Pourquoi pas de patches dans l'outil d'origine (déduction) : champ récepteur de ~200 mm (le réseau voit tout le cerveau), statistiques GroupNorm dépendantes de l'étendue de l'entrée, héritage SynthSeg, outil pensé pour le CPU.

**Décision.** Mode CPU **d'origine, sans modification**, sur Kaggle CPU, sur les 110 sujets de test (~9 h en 2 sessions, sans quota GPU). Rôle : modèle généraliste prêt à l'emploi comparé à nnU-Net.

**Alternatives écartées.**
- *fp16 + crop sur GPU* (~8 Go estimés) : seule la précision change ; gardé en extension, à valider contre la sortie CPU (Dice d'accord > 0,98).
- *Fenêtres glissantes* : change le contexte et la normalisation → pas équivalent au modèle publié ; écarté du périmètre de base.

**Preuves (rapport Kaggle du 2026-10-02).**
- GPU T4 (15,6 Go), `--crop` : échec, pic de VRAM relevé ≥ 12,3 Go (relevé chaque seconde, le pic réel est plus haut) → confirme l'analyse.
- CPU, `--crop` : ~300-320 s par sujet, pic de RAM **26,4 Go**.

| Sujet | Dice | Volume vérité terrain | Volume prédit |
|---|---|---|---|
| Utrecht 0 | 0,484 | 24,2 ml | 43,8 ml |
| Singapour 50 | 0,732 | 25,7 ml | 21,5 ml |
| Amsterdam GE3T 100 | 0,224 | 6,3 ml | 18,6 ml |
| Amsterdam GE1.5T 150 | — | — | arrêté par le watchdog (RAM > 28,6 Go) |
| Amsterdam Philips 160 | — | — | arrêté par le watchdog (RAM > 28,7 Go) |

Lecture : performance très variable selon le site, forte surestimation quand la charge est faible. C'est attendu pour un modèle généraliste et c'est utile pour l'axe 2. Les deux scanners « inconnus » ont des champs de vue plus grands, d'où le besoin de plus de RAM.

**Point technique.** PyTorch ≥ 2.6 charge en `weights_only=True` : on autorise explicitement les types numpy du checkpoint (`add_safe_globals`) plutôt que de désactiver la protection (`weights_only=False`), qui permettrait l'exécution de code arbitraire.

---

## J-008 — nnU-Net v2, entraîné à partir de zéro, comme méthode principale
2026-10-03 · Actée

**Contexte.** Il faut une méthode de segmentation de référence, reconnue et défendable.

**Décision.** nnU-Net v2 (Isensee et al., DKFZ ; Nature Methods 2021), **entraîné à partir de poids aléatoires** sur les 60 cas d'entraînement, avec la recette standard. Ce n'est pas du fine-tuning : nnU-Net est un framework qui se configure seul, pas un modèle pré-entraîné.

**Raisons.** Standard de fait en segmentation médicale ; correspond à l'axe 2 (intégrer et évaluer l'existant plutôt qu'inventer) ; 60 cas suffisent pour cette tâche (chaque volume compte des millions de voxels annotés et des dizaines de lésions, entraînement par patches, augmentation massive, tâche locale) ; **toutes les équipes du classement du challenge ont été entraînées sur ces mêmes 60 cas**.

**Repère.** Leaderboard officiel (readme, 57 équipes) : 1er « pgs » Dice 0,81 / HD95 5,63 mm / AVD 18,6 % / rappel 0,82 / F1 0,79 ; rangs 2-3 à 0,80. Cible réaliste : Dice 0,75-0,80.

**Alternatives écartées.** Architecture maison (hors esprit de l'offre, risquée en 5 jours) ; fine-tuning d'un modèle 3D pré-entraîné (dépendance et variable en plus, gain incertain → extension).

**Provisoire** : configuration (3d_fullres ou 2d) et nombre d'epochs, selon la durée d'epoch mesurée sur Kaggle.

---

## J-009 — Trois méthodes comparées et deux repères
2026-10-03 · Actée

| | Méthode | Apprentissage | Question |
|---|---|---|---|
| M1 | nnU-Net v2 | Entraîné sur nos 60 cas | Modèle **spécialisé** |
| M2 | WMH-SynthSeg | Pré-entraîné, aucune adaptation | Modèle **généraliste** publié |
| M3 | Seuillage FLAIR | Un seuil réglé sur le train | Méthode **classique**, sensible aux intensités |
| Repère | Observateurs O3/O4 | — | Plafond humain |
| Repère | Leaderboard officiel | — | Positionnement externe |

**Raison.** Reproduit la question pratique de l'axe 2 : utiliser un modèle publié tel quel ou entraîner sur ses données ? Où chaque approche décroche-t-elle quand le scanner change ?

---

## J-010 — Notre prétraitement, à partir des images brutes, alimente la segmentation
2026-10-03 · Actée (remplace une version antérieure du plan où nnU-Net utilisait `pre/`)

**Contexte.** Le challenge fournit des images déjà prétraitées (`pre/` : SPM12 + elastix). Les utiliser pour nnU-Net aurait réduit le prétraitement à un rôle annexe et masqué la compétence de l'axe 1.

**Décision.** La chaîne principale part de `orig/` : N4 (ANTs), recalage rigide T1 3D → FLAIR (ANTs), masque du cerveau (HD-BET), SyN → MNI. **nnU-Net est entraîné et appliqué sur nos images.** `pre/` sert de **référence de vérification** (qualité du recalage, comparaison des intensités) et pour une expérience « effet du prétraitement » (inférence de notre modèle sur `pre/`).

**Pourquoi la comparaison au leaderboard reste valide.** L'évaluation se fait dans l'espace natif de la FLAIR et **notre prétraitement ne rééchantillonne pas la FLAIR** (c'est la T1 qui est amenée dans sa grille) : prédictions et vérité terrain partagent la même grille. L'argument contraire avancé plus tôt était faux.

**Risque et repli.** L'entraînement dépend désormais de notre prétraitement des 60 cas d'entraînement (jalon J1 soir). Si ce n'est pas prêt, on lance l'entraînement sur `pre/` pour ne pas perdre la nuit de calcul, puis on bascule.

---

## J-011 — Contrôle qualité : métriques maison inspirées de MRIQC
2026-10-03 · Actée

**Contexte.** MRIQC est l'outil de référence du QC IRM, mais il ne gère que T1w, T2w, BOLD et diffusion : **pas de FLAIR**. Il faut en plus Docker, absent en local.

**Décision.** Module `qc/iqm.py` qui reprend les définitions MRIQC (SNR, CNR, CJV, EFC, FBER…) sur FLAIR et T1, puis détection d'images atypiques (z-scores robustes, IsolationForest), **validée par injection d'artefacts** (TorchIO) d'intensité connue.

**Source.** [Documentation MRIQC](https://mriqc.readthedocs.io/en/stable/running.html) (modalités supportées).

---

## J-012 — Évaluation avec le script officiel du challenge
2026-10-02 · Actée

**Décision.** Utiliser `evaluation.py` de [hjkuijf/wmhchallenge](https://github.com/hjkuijf/wmhchallenge) (licence MIT) : Dice, HD95, AVD (%), rappel et F1 par lésion (composantes connexes 3D), label 2 ignoré.
**Raison.** Résultats directement comparables au classement officiel ; aucune réimplémentation susceptible de diverger.

---

## J-013 — Aucun réglage sur le jeu de test
2026-10-03 · Actée

**Décision.** Tout choix de paramètre (seuil de M3, ajustement de ComBat, choix de configuration nnU-Net) se fait sur les 60 cas d'entraînement. Les 110 cas de test ne servent qu'à l'évaluation finale.

---

## J-014 — Pas de modèle pronostique
2026-10-02 · Actée

**Contexte.** L'axe 4 de l'offre porte sur la fusion image + données cliniques.
**Décision.** Hors périmètre : le jeu WMH ne contient aucune variable clinique ni aucun suivi. Mentionné comme extension (ex. ISLES'24 : 149 AVC avec mRS à 3 mois).

---

## J-015 — Téléchargement partiel des données
2026-10-03 · Actée

**Décision.** On ne télécharge pas `pre/3DT1.nii.gz` (T1 3D en float32, ~22 Mo par sujet, ~3,7 Go au total, inutile pour nous). Environ 5 Go au lieu de 8,8 Go.
**Raison.** 26 Go de disque libre en local (J-005).

---

## J-016 — Recalage rigide ANTs par défaut : suffisant
2026-10-03 · Actée

**Preuves.** Recalage rigide (information mutuelle) T1 3D → FLAIR sur 5 sujets de 5 scanners : corrélation avec la T1 alignée par elastix (référence du challenge) de **0,981 à 0,993** ; 14-16 s par sujet (48 s pour le Philips, dont la FLAIR compte 321 × 240 × 83 voxels).
**Décision.** Paramètres par défaut d'ANTsPy (`type_of_transform="Rigid"`). La corrélation avec elastix est conservée comme **indicateur QC du recalage** pour chaque sujet.

---

## J-017 — Recalage SyN vers MNI : en local
2026-10-03 · Actée

**Preuves.** 78,6 s sur Kaggle (4 cœurs). Le PC local a 12 threads, et ANTs SyN consomme peu de RAM.
**Décision.** En local, la nuit, sous watchdog, 2 sujets en parallèle au maximum : 170 sujets ≈ 2 à 4 h.

---

## J-018 — HD-BET : sur Kaggle GPU, et découplé de l'entrée nnU-Net
2026-10-03 · Actée

**Preuves.** HD-BET (GPU, sans TTA) : 67,8 s pour le premier sujet (chargement compris), **5,8 Go de VRAM, 10,5 Go de RAM** au pic.
**Décision.** Au-dessus du seuil fixé pour le PC (5 Go de VRAM, 10 Go de RAM, J-005) → HD-BET tourne sur Kaggle GPU, sur les 170 T1 en une seule session (~1 à 1,5 h de quota).
**Conséquence.** Pour ne pas retarder l'entraînement, la correction N4 des images d'entrée de nnU-Net utilise un **masque de premier plan simple** (seuillage d'Otsu, `skimage.filters.threshold_otsu`, puis nettoyage morphologique) et non le masque HD-BET. *Correction du 2026-10-03 : `ants.get_mask` avait été cité par erreur ; il seuille par défaut à la moyenne de l'image, pas par la méthode d'Otsu.* Le masque HD-BET sert au QC, au seuillage (M3), aux biomarqueurs et à l'harmonisation. nnU-Net n'a pas besoin d'images sans crâne (les images `pre/` du challenge ne l'étaient pas non plus).

---

## J-019 — nnU-Net : configuration 3d_fullres et budget d'entraînement
2026-10-03 · Provisoire (choix du nombre d'epochs à valider)

**Preuves (Kaggle T4, 4 cœurs CPU).**
- Plan automatique 3d_fullres : patch 48 × 224 × 192, batch 2, espacement 3,0 × 0,98 × 1,0 mm (médiane des FLAIR en coupes de 3 mm). Plan 2d : patch 256 × 224, batch 57.
- Entraînement 3d_fullres : **~194 s par epoch** (287 s pour la première), **6,8 Go de VRAM**, 14 Go de RAM. Pseudo-Dice de validation après 5 epochs : 0,844 (indicateur interne optimiste, calculé sur des patches, pas la métrique officielle).
- Extrapolation : 250 epochs ≈ **13,5 h**, 1000 epochs (défaut) ≈ 54 h → le défaut est hors budget.
- Prétraitement nnU-Net des 60 cas : 203 s.
- La configuration 2d n'apparaît pas dans le rapport (durée d'epoch inconnue).

**Mesure dédiée (benchmark Kaggle, 2026-10-03, notre Dataset001_WMH, fold 0).** Kaggle décompresse les `.nii.gz` d'une archive déposée : le notebook utilise donc l'extension réelle (`.nii`). Temps par epoch (médiane, hors 1re epoch) : A 1 GPU, 3 processus d'augmentation, compile : 219 s ; B 1 GPU, 4 processus, compile : 215 s ; C 1 GPU, 4 processus, sans compile : 251 s ; **D 2 GPU (DDP), 4 processus, compile : 110 s**. Sur 1 GPU, le GPU est utilisé à 98 % et le CPU à 43 % : l'entraînement est limité par le GPU, pas par l'augmentation de données. `torch.compile` fait gagner 14 %. Deux T4 divisent le temps par 1,96 (GPU à 94-96 %, CPU à 87 %, ~4 Go de mémoire par GPU). Projection D : 250 epochs ≈ 7,8 h, **dans une seule session de 12 h**. `nnUNetTrainer_150epochs` n'existe pas dans nnU-Net (100, 250, 500... seulement). En DDP, le lot de 2 est réparti à 1 image par GPU et les gradients sont moyennés ; nnU-Net utilisant l'InstanceNorm (et non la BatchNorm), c'est équivalent à un lot de 2 sur un GPU.
**Recommandation (révisée après mesure).** `nnUNetTrainer_250epochs`, 2 GPU, 4 processus d'augmentation, `torch.compile` actif : ~7,8 h en une session.
**Recommandation initiale (avant mesure).** 3d_fullres avec `nnUNetTrainer_250epochs`, réparti sur **2 sessions** grâce à la reprise sur point de contrôle (`--c`). La durée d'entraînement est un paramètre du trainer : le planning du taux d'apprentissage s'adapte au nombre total d'epochs, ce n'est donc pas un entraînement « coupé ».
**Repli.** `nnUNetTrainer_150epochs` (~8 h, une session) si le quota ou le calendrier se tendent.
**Budget GPU.** ~1,1 h (test) + 13,5 h (entraînement) + ~1,5 h (HD-BET) + ~1 h (inférence) ≈ **17 h sur 30**.
**À vérifier.** Si l'epoch est limitée par le CPU (augmentation de données sur 4 cœurs) : relever l'utilisation du GPU pendant l'entraînement.

---

## J-020 — WMH-SynthSeg : périmètre réduit à un sous-ensemble équilibré par scanner
2026-10-03 · Remplacée par J-043

**Contexte.** ~5 min et ~26-29 Go de RAM par sujet sur Kaggle CPU (33,7 Go). Les deux scanners inconnus dépassent le seuil du watchdog fixé à 85 % de la RAM.
**Décision.** 10 sujets par scanner (50 au total, ~4,5 h, une session CPU). Le seuil du watchdog est relevé à ~32 Go. Chaque sujet tourne dans un sous-processus, donc un manque de mémoire ne tue que ce sujet, pas la session. Les échecs éventuels sont **rapportés comme résultat** (coût mémoire de l'outil).
**Alternative écartée.** Les 110 sujets de test (~9,5 h) : coût élevé pour une méthode secondaire qui plafonne autour d'un Dice de 0,5.

---

## J-021 — Gestion de l'environnement avec uv et fichier de verrouillage
2026-10-03 · Actée

**Décision.** Environnement local géré par [uv](https://docs.astral.sh/uv/) en mode projet : `uv sync --extra local` crée `.venv` à partir de `pyproject.toml` et génère `uv.lock`, qui est **versionné**. Version de Python fixée par `.python-version` (3.12, celle de la machine).
**Raisons.** Le fichier de verrouillage fige la version exacte de chaque dépendance, y compris les dépendances indirectes, pour toutes les plateformes : c'est la reproductibilité demandée par l'offre, sans `pip freeze` manuel. uv est déjà installé (0.11.6) et nettement plus rapide que pip. `pyproject.toml` garde des bornes minimales lisibles ; les versions exactes sont dans `uv.lock`.
**Preuve.** `uv lock --dry-run` résout l'ensemble des extras sans conflit.
**Conséquence.** Les notebooks Kaggle continuent d'utiliser `pip` (environnement imposé par Kaggle), en installant les versions du verrou pour les paquets du projet.

---

## J-022 — neuroHarmonize : déclaration explicite de neuroCombat et statsmodels
2026-10-03 · Actée

**Contexte.** Après `uv sync`, `import neuroHarmonize` échoue : `No module named 'neuroCombat'`.
**Cause.** neuroHarmonize 2.5.1 déclare correctement ses dépendances mais impose `numpy<2.0`, incompatible avec le reste de l'environnement (numpy 2.3). Le solveur se rabat donc sur la 2.4.5, dont les métadonnées ne déclarent **aucune** dépendance : neuroCombat n'est jamais installé.
**Décision.** Garder neuroHarmonize 2.4.5 et déclarer ses dépendances d'exécution dans l'extra `harmonize` : `neuroCombat==0.2.12` (version exigée par la 2.5.1) et `statsmodels>=0.14`.
**Alternatives écartées.** Forcer `numpy<2` : rétrograde tout l'environnement pour une seule bibliothèque. Abandonner neuroHarmonize pour neuroCombat seul : on perdrait ComBat-GAM et l'application d'un modèle appris sur le train à de nouvelles données.
**Preuve.** Test sur données simulées (3 sites décalés, covariable à préserver), numpy 2.3.5 : écart entre sites 6,59 → 2,19 après harmonisation, corrélation avec la covariable 0,85 → 0,95. À revalider sur les vraies données en Phase 8.

---

## J-023 — Notebooks : sorties retirées au commit (nbstripout)
2026-10-03 · Actée

**Contexte.** Un notebook exécuté embarque ses figures (2,5 Mo pour `01_explore_nifti.ipynb`) : l'historique Git gonfle et chaque réexécution produit un diff illisible.
**Décision.** Filtre Git `nbstripout` (déclaré dans `.gitattributes`, outil dans l'extra `dev`) : les sorties sont retirées de la version commitée et conservées dans le fichier local. Les figures destinées au README sont exportées dans `results/figures/`, pas lues dans les notebooks.
**Preuve.** Version indexée : 0 sortie ; fichier local : 22 sorties conservées.
**Note.** Le filtre est enregistré dans la configuration Git locale (`.git/config`) : sur un autre clone, relancer `uv run nbstripout --install --attributes .gitattributes`.

---

## J-024 — `save_like` : pas de remise à zéro explicite de la mise à l'échelle
2026-10-03 · Actée

**Contexte.** Crainte qu'un en-tête copié d'une image mise à l'échelle (`scl_slope`, `scl_inter`) transforme les valeurs écrites.
**Preuve.** Test : avec un en-tête de référence à pente 2 et ordonnée 10, le constructeur `nib.Nifti1Image(data, affine, header=...)` remet déjà la mise à l'échelle à « aucune » ; les labels 0/1/2 sont relus à l'identique. Le risque n'existe que si l'en-tête est modifié après la construction (valeurs relues 10/12/14 dans ce cas).
**Décision.** Ligne `set_slope_inter(1, 0)` supprimée : le comportement de nibabel suffit, et `save_like` ne modifie pas l'en-tête après construction.

---

## J-025 — Nommage BIDS : `sub-{id:03d}` à partir de l'identifiant d'origine
2026-10-03 · Actée

**Décision.** `sub-000` à `sub-169`, l'identifiant numérique publié par le challenge, complété à 3 chiffres. Site, scanner et split sont des colonnes de `participants.tsv`, pas des éléments du nom.
**Raisons.** Les identifiants d'origine sont uniques sur tout le jeu (Utrecht 0-49, Singapour 50-99, Amsterdam 100-169) : traçabilité directe vers les fichiers publiés et le leaderboard, sans table de correspondance. BIDS interdit `_` et `-` dans les labels : un label numérique est le plus sûr. Mettre le site dans le nom aurait mélangé identité et métadonnée.

---

## J-026 — Téléchargement : SHA-1, miroir à noms assainis, liens physiques pour BIDS
2026-10-03 · Actée

- **Empreinte.** DataverseNL publie un **SHA-1** pour chaque fichier (pas de MD5, contrairement à ce que prévoyait la ROADMAP). Chaque fichier est haché pendant le téléchargement, écrit sous `<nom>.part`, puis renommé seulement si l'empreinte correspond : un téléchargement interrompu ne laisse jamais de fichier tronqué sous son nom final. Les fichiers déjà présents et valides ne sont pas retéléchargés (vérifié : les 10 fichiers téléchargés par le notebook ont été reconnus).
- **Noms de dossiers.** Le dossier publié `Amsterdam/Philips_VU .PETMR_01.` contient une espace et finit par un point ; **Windows supprime silencieusement les points et espaces finaux**, le miroir local divergerait du chemin publié. Règle déterministe : espace → `_`, point final → `_` (`Philips_VU_.PETMR_01_`). Le chemin publié exact reste dans le manifeste.
- **Manifeste** `data/raw/manifest.csv` : chemin publié, chemin local, identifiant, taille, SHA-1, statut, version du dataset (1.0), date. Source de vérité de l'étape BIDS.
- **BIDS sans duplication.** Les fichiers BIDS sont des **liens physiques** vers `data/raw/` (mêmes octets, aucun espace disque supplémentaire ; copie en repli). Vérifié : 1 310/1 310 images liées, `data/bids` n'occupe que les JSON/TSV.
- **Résultat.** 1 481 fichiers (4,07 Go), 0 échec, environ 3 minutes.

---

## J-027 — Inventaire QC de la conversion BIDS
2026-10-03 · Actée

**Contenu.** `results/tables/bids_inventory.csv` (versionné : géométrie et volumes, aucune donnée image) : formes, tailles de voxel et orientations de la FLAIR et de la T1, cohérence qform/sform, validité et grille de chaque masque (O1, O3, O4), grille de la T1 alignée par les organisateurs, volumes de HSB et d'autre pathologie.
**Constats sur les 170 sujets.**
- Aucune anomalie : qform/sform cohérentes, tous les masques valides (labels 0/1/2) et sur la grille de leur FLAIR.
- Une taille de voxel FLAIR unique par scanner ; toutes les FLAIR en `LPS`, toutes les T1 en `PIR`.
- **Exception : sub-017** (Utrecht, entraînement) a une T1 de 288 × 288 × 192 voxels de 0,89 mm au lieu de 256 × 256 × 192 à 1 mm : acquisition hors protocole, à suivre au QC (Phase 4).
- Charge lésionnelle O1 (médiane [min-max]) : Utrecht 13,9 ml [0,8-195,1], Singapour 15,9 [0,8-80,5], Amsterdam GE 3T 6,3 [0,8-54,2], GE 1,5T 8,5 [0,8-69,2], Philips 3T 7,6 [1,1-45,0].
- Observateurs sur le train : médiane O1 15,1 ml, O3 13,3 ml, O4 12,2 ml → O3 et O4 annotent moins que la référence (à quantifier en 6.3).

---

## J-028 — Défacement : masques FLAIR d'Amsterdam, noms explicites, calculs restreints au cerveau
2026-10-03 · Actée

> **CONSIGNE : RAPPELER CETTE ENTRÉE À L'UTILISATEUR AU MOMENT D'IMPLÉMENTER** les étapes 4.1 (métriques de qualité), 4.2 (détection d'atypiques), 5.5 (seuillage M3), 7.1 (volumes normalisés), 8.1 (normalisation d'intensité) et 8.2 (caractéristiques pour l'harmonisation), en lui citant les deux règles ci-dessous avant d'écrire le code.

**Constats (vérifiés sur les 170 sujets).**
- Masques publiés : `3DT1_mask` (grille T1) pour tous ; `FLAIR_mask` et `T1_mask` (grille FLAIR, identiques entre eux) pour les **70 sujets d'Amsterdam**, dont les FLAIR 3D ont été défacées (22 % du volume en médiane, de 11,6 à 46,1 %). Les FLAIR 2D d'Utrecht et de Singapour ne sont pas défacées, bien qu'elles couvrent une partie du visage (tissu dans la zone défacée de la T1 : Singapour médiane 40 ml, Utrecht 7 ml ; voir la note README de la ROADMAP).
- **Les `FLAIR_mask` ne sont pas binaires** : environ 1 % des voxels (au plus 3 %), sur la frontière, ont des valeurs interpolées dans ]0, 1[. Détecté par `io.load_labels`, qui a refusé ces fichiers. Sur les 70 sujets, la FLAIR vaut exactement 0 **là où le masque vaut exactement 0** : la zone défacée est définie par `mask == 0`. Les `3DT1_mask` sont, eux, strictement binaires (170/170).

**Décisions.**
1. Télécharger `orig/FLAIR_mask.nii.gz` (70 fichiers ; `T1_mask` est redondant). Total : 1 551 fichiers, 4,08 Go.
2. Noms BIDS qui disent la grille : `sub-XXX_space-T1w_desc-deface_mask.nii.gz` et `sub-XXX_space-FLAIR_desc-deface_mask.nii.gz` (Amsterdam). Les anciens fichiers `sub-XXX_desc-deface_mask.nii.gz` ont été supprimés par `prune_stale` (170 fichiers).
3. Pas de masque « maison » pour les FLAIR d'Utrecht et de Singapour : le projet ne redistribue aucune image (l'anonymisation n'est pas son rôle), et un masque transporté par affine laisserait des résidus. Il serait nécessaire seulement si on publiait des images dérivées (il faudrait alors un vrai outil de défacement).
4. **Deux règles pour que le défacement (présent à Amsterdam, absent ailleurs) ne crée pas un faux effet de site :**
   - **Règle A** : toute statistique d'intensité (normalisation, caractéristiques d'harmonisation, seuil de M3, métriques de qualité sur le tissu, volumes normalisés) se calcule **dans le masque du cerveau** (HD-BET), jamais sur la tête entière.
   - **Règle B** : les estimations de bruit sur l'arrière-plan **excluent les zéros exacts**, ce qui traite d'un coup les zéros du défacement et ceux que le scanner met dans l'air.
5. nnU-Net reçoit les images sans masquage (visage présent ou non selon le site, hors du cerveau), comme toutes les équipes du challenge : comparabilité avec le classement.

---

## J-029 — Stratégie de test : données synthétiques, aucun accès réseau
2026-10-03 · Actée

**Décision.** Tests pytest autonomes (`tests/`) : images NIfTI de quelques voxels construites avec numpy, fausse arborescence du challenge (3 sujets, dont un dossier au nom invalide sous Windows et un masque FLAIR à contour interpolé), faux serveur HTTP par `monkeypatch`. Aucun test ne lit les vraies données ni le réseau.
**Raisons.** La CI GitHub (9.3) n'aura pas les données (licence, 4 Go) ; un test doit être rapide et connaître la bonne réponse à l'avance ; chaque piège rencontré sur les vraies données (masque interpolé, NaN, grille décalée, mise à l'échelle, nom de dossier Windows, téléchargement tronqué) devient un test de non-régression.
**Résultat.** 70 tests, 5,6 s, couverture 91 % du code écrit (`io.py` 100 %, `bids.py` 95 %, `download.py` 88 %, `guard.py` 80 %). **Un bug trouvé et corrigé** : dans `bids.py`, le résumé final plantait (`min()` d'une liste vide) quand tous les masques O1 d'un scanner étaient invalides, c'est-à-dire précisément dans le cas qu'il devait signaler.

---

## J-030 — Correction N4 : validation sur les 60 cas d'entraînement
2026-10-03 · Actée (une question ouverte)

**Mise en oeuvre.** `preproc/bias.py` : N4 (ANTsPy, paramètres par défaut) sur la FLAIR et la T1 **brutes**, la T1 à 1 mm avant recalage ; masque de tête par seuil d'Otsu, plus grande composante connexe, remplissage des trous en 3D puis coupe par coupe sur les trois axes (le remplissage 3D seul laisse ouvertes les cavités qui touchent le bord du champ de vue : +3 656 voxels contre +305 882 coupe par coupe sur sub-000). Un sujet par processus surveillé, 2 en parallèle. Sorties `desc-n4`, `desc-biasfield`, `desc-head_mask` dans `data/derivatives/wmh-multisite`.
**Résultats** (`notebooks/03_n4_audit.ipynb`, `results/tables/n4_audit.csv`) : 120 images, 0 échec, ~3 s par FLAIR et ~14 s par T1. Cohérence géométrique vérifiée (image corrigée × champ = image brute à 1e-8). **Aucune image dégradée** : homogénéité du tissu et dispersion de luminosité entre coupes meilleures qu'avec SPM12 sur les trois sites (dispersion médiane N4 / SPM12 / brute : Amsterdam 0,024 / 0,034 / 0,043 ; Utrecht 0,017 / 0,024 / 0,038 ; Singapour 0,017 / 0,022 / 0,028). Champ T1 corrélé à celui de SPM12 (médianes 0,75 à 0,89).
**Anomalie : FLAIR de Singapour.** Corrélation avec le champ SPM12 inférieure à 0,3 pour 19 sujets sur 20 (environ 0,8 ailleurs). Hypothèse « visage et cou dans le masque » **réfutée** (masque érodé ou limité aux coupes hautes : pas d'amélioration). Constat : SPM12 estime à Singapour un biais plus faible (amplitude 0,065 contre 0,08 à 0,10) alors que N4 estime partout un champ d'amplitude ~0,12 ; quand le biais réel est faible, la part du champ N4 qui n'est pas du biais domine. Origine probable du biais faible (non vérifiable) : normalisation par le scanner. **Question ouverte** : refaire N4 avec le masque HD-BET (3.3) ; l'expérience « effet du prétraitement » (5.3) dira si cela change la segmentation.
**Autres signaux.** Champ T1 étendu à Singapour (p95 jusqu'à 1,6) : biais réel, confirmé par SPM12. Masques FLAIR sous 12 % du volume (sub-107, sub-112) : fausse alerte, masques corrects dans un grand champ de vue ; **au QC, raisonner en ml et non en fraction du volume**.
**Constat visuel (ajouté après inspection des masques).** La même règle (Otsu + remplissage) donne des masques de **définition différente selon le site**, à cause du contraste propre à chaque scanner : tête entière à Utrecht (le seuil retient surtout le cuir chevelu, le remplissage ajoute l'intérieur : 30 % du masque rempli en médiane), essentiellement le cerveau à Amsterdam (cuir chevelu sous le seuil), intermédiaire et irrégulier à Singapour (cavités remplies en rectangles par le remplissage coupe par coupe). L'air réel dans le masque reste faible partout (médiane < 4 %) et Utrecht, le plus « rempli », est le site où N4 et SPM12 s'accordent le mieux : ces différences n'expliquent pas l'anomalie de Singapour et ne dégradent pas N4. **Conséquence** : ce masque ne sert qu'à N4 ; aucun volume ni statistique ne doit en être tiré (le volume du masque n'est pas comparable entre sites). Le masque du cerveau HD-BET (3.3) sert à tout le reste (J-028).
**Défaut corrigé.** Les sujets déjà traités étaient réécrits dans `n4_report.csv` sans leurs statistiques ; elles sont maintenant recalculées depuis les fichiers.

---

## J-031 — Recalage rigide T1 → FLAIR : critère, contrôle qualité, déterminisme, lieu d'exécution
2026-10-03 · Actée

**Mise en oeuvre.** `preproc/register.py` : ANTsPy `registration(type_of_transform="Rigid")`, image fixe = FLAIR N4 (jamais rééchantillonnée), image mobile = T1 N4 à 1 mm ; critère information mutuelle de Mattes (défaut ANTs : 32 classes d'histogramme, 20 % des voxels échantillonnés au hasard, 4 niveaux de résolution, réduction 6/4/2/1, lissage 3/2/1/0 voxels). T1 rééchantillonnée en linéaire, masque de tête en `genericLabel`. Transformation sauvegardée au format ITK.
**Contrôle qualité.** (1) La transformation est appliquée à la T1 *brute* et corrélée à `orig/T1` (T1 brute alignée par elastix) : on compare deux images non corrigées par N4. (2) Écart géométrique avec la transformation elastix, mesuré en appliquant les deux transformations à 5 000 points de la tête (moyenne et maximum en mm) et angle de la rotation relative : comparer les paramètres eux-mêmes serait trompeur, les deux outils n'utilisant pas le même centre de rotation. Seuils d'alerte : corrélation < 0,95, rotation > 2°, déplacement > 2 mm (Kaggle : 0,981-0,993).
**Déterminisme.** Une graine aléatoire fixée (`ants.config._random_seed` ; l'argument `random_seed` de `ants.registration` est ignoré) ne suffit pas : en multi-thread, ITK additionne le critère dans un ordre variable et deux exécutions diffèrent légèrement (paramètres à 5e-4 près, image jusqu'à 10 unités d'intensité). Avec un seul thread (`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1`), deux exécutions sont **strictement identiques**. Coût : ~40 s au lieu de ~28 s par sujet, compensé en lançant 4 sujets en parallèle (0,73 Go de RAM par sujet au maximum).
**Lieu d'exécution : en local.** Le recalage n'utilise que le CPU (ANTs n'a pas de version GPU) et peu de mémoire : il ne relève pas des tâches lourdes de J-005. Le PC (12 threads) est plus rapide que Kaggle (4 cœurs) et évite de transférer les données.
**Résultats sur 3 sujets** (un par site) : corrélation 0,987 à 0,999, écart maximal avec elastix 0,14 à 0,93 mm, rotation relative 0,05 à 0,36°.
**À vérifier.** N4 utilise aussi ITK en multi-thread : son déterminisme n'a pas été testé.

---

## J-032 — Export nnU-Net : format, label ignoré, folds stratifiés par site et par charge lésionnelle
2026-10-03 · Actée

**Mise en oeuvre.** `seg/nnunet.py` construit `data/nnunet/nnUNet_raw/Dataset001_WMH/` à partir de **nos** images (J-010) : canal 0 = FLAIR N4, canal 1 = T1 N4 recalée sur la grille FLAIR, labels O1 réécrits en uint8 (le challenge les stocke en float32). Identifiants `WMH_XXX` reprenant le numéro d'origine, table `case_mapping.csv` (cas, sujet BIDS, scanner, volumes). Images en liens physiques (aucun espace disque), archive `Dataset001_WMH.zip` (841 Mo, stockage sans recompression : les NIfTI sont déjà compressés).
**Label 2 = `ignore`.** Les voxels « autre pathologie » ne comptent ni dans la perte ni dans la métrique de validation de nnU-Net, comme dans l'évaluation officielle du challenge. nnU-Net exige que le label ignoré soit le plus élevé : c'est le cas.
**Folds.** nnU-Net tire sinon 5 folds au hasard. Un premier découpage stratifié **par scanner seulement** donnait 4 sujets de chaque site par fold, mais des charges lésionnelles très déséquilibrées entre folds de validation (médianes de 4,6 à 26,1 ml pour une médiane globale de 15,1 ml) : le fold 0, celui qu'on entraîne, aurait été validé sur des cas surtout lourds. Découpage retenu : à l'intérieur de chaque scanner, cas triés par volume de HSB, découpés en blocs de 5 cas de charge voisine, chaque bloc réparti aléatoirement (graine 42) entre les 5 folds. Résultat : 4 sujets par site et par fold, médianes de validation de 10,9 à 16,5 ml. Fichier `splits_final.json`, à copier par le notebook Kaggle dans `nnUNet_preprocessed/Dataset001_WMH/`.
**Vérifications.** Contrôle maison (mêmes grilles pour les 3 fichiers de chaque cas, labels dans {0, 1, 2}) et **vérificateur officiel `verify_dataset_integrity` de nnU-Net** (exécuté dans un environnement temporaire `uv run --with nnunetv2`, sans l'ajouter au projet local) : aucune erreur. Tests : `tests/test_nnunet.py` (partition, stratification par scanner et par quintile de charge, reproductibilité, label ignoré).
**Point ouvert hors sujet.** Un test du watchdog (`test_call_guarded_reports_the_exception`) a échoué une fois sur cinq exécutions de la suite, lors d'une exécution anormalement lente (machine chargée) ; il passe systématiquement sinon. Probable condition de concurrence dans `call_guarded` sous charge, à investiguer.

---

## J-033 — Masque du cerveau HD-BET : T1 brute, exécution locale sur GPU, transport sur la grille FLAIR
2026-10-04 · Actée (révisée le jour même : exécution locale au lieu de Kaggle, voir la fin de l'entrée)

**Décision.** HD-BET v2 (Isensee et al. 2019, réseau nnU-Net) est appliqué à la **T1 3D brute à 1 mm**, la meilleure résolution anatomique disponible (la FLAIR n'a que des coupes de 3 mm), (J-018 : 5,8 Go de VRAM et 10,5 Go de RAM mesurés sur Kaggle avec les réglages par défaut). Augmentation au test activée (recommandée sur GPU par les auteurs). Les masques sont ensuite transportés sur la grille FLAIR avec la transformation rigide de l'étape 3.2, en interpolation `genericLabel`.
**Pourquoi la T1 brute et pas la T1 N4.** HD-BET est entraîné sur des images non corrigées, multi-centres ; son entrée doit ressembler à ses données d'entraînement. La T1 brute est aussi un fichier publié, vérifiable par empreinte.
**Pas d'envoi de données depuis le PC.** Le notebook `kaggle/02_hdbet.ipynb` télécharge lui-même les 170 T1 depuis DataverseNL et vérifie leur SHA-1. L'import local (`preproc/brainmask.py`) **refuse** un masque si : la T1 vue sur Kaggle n'a pas le même SHA-1 que celle de notre manifeste ; le masque n'est pas sur la grille de notre T1 ; il contient autre chose que 0/1.
**Contrôle qualité** (`results/tables/brainmask_qc.csv`) : volume du cerveau (alerte hors 900-2 000 ml), nombre de composantes connexes et part de la plus grande (alerte sous 99 %), part du cerveau contenue dans le masque de tête N4, et **couverture du cerveau par le champ de vue de la FLAIR** (volume sur la grille FLAIR / volume sur la grille T1 ; la FLAIR ne couvre que ~144 mm de hauteur et peut tronquer le haut ou le bas du cerveau).
**Rôle.** Ce masque définit « le cerveau » pour toutes les mesures (règle A de J-028) : normalisation d'intensité, métriques de qualité du tissu, seuillage M3, fraction du volume cérébral, harmonisation. Il n'entre pas dans nnU-Net (J-018).
**Tests.** `tests/test_brainmask.py` : import correct, refus d'une entrée différente (SHA-1), d'un masque sur une autre grille et d'un masque non binaire ; transport sur la grille FLAIR et contrôle qualité.
**Test local sur un sujet (2026-10-04).** Les 10,5 Go de RAM mesurés sur Kaggle venaient surtout des processus parallèles de HD-BET (4 de préparation, 8 d'écriture). En local, dans un environnement temporaire (`uv run --no-project --with hd-bet --with torch==2.14.1+cu130`, sans toucher au `.venv` du projet, dont le PyTorch est CPU), avec l'API Python de HD-BET, 1 processus de préparation et 1 d'écriture, TTA activée, un plafond de mémoire GPU imposé à PyTorch (`set_per_process_memory_fraction`, 5,5 Go : erreur nette au lieu d'un débordement du pilote Windows dans la RAM) et le watchdog (RAM 10 Go, VRAM 6 Go) : **pic de RAM 3,2 Go, pic de mémoire GPU 4,1 Go** (PyTorch : 3,8 Go alloués, 4,2 Go réservés), 38 s de prédiction pour sub-000 (après 20 s de téléchargement des poids). Masque sur la grille de la T1, binaire, une seule composante, 1 630 ml, entièrement contenu dans le masque de tête N4 ; contrôle visuel correct (crâne exclu, tronc cérébral coupé au trou occipital). **Conclusion : l'exécution locale est sûre avec ces garde-fous.**
**Révision : exécution locale.** Le notebook Kaggle `02_hdbet` est supprimé. `preproc/brainmask.py` lance HD-BET par lots (20 sujets, modèle chargé une fois par lot) via `preproc/hdbet_worker.py`, exécuté **par chemin** dans l'environnement temporaire CUDA (`uv run --no-project`, versions fixées dans `config.yaml` : `hd-bet==2.0.1`, `torch==2.14.1+cu130`), sous le watchdog (RAM 10 Go, mémoire GPU 6 Go, délai proportionnel au lot) avec le plafond PyTorch de 5,5 Go. Un sujet déjà traité est sauté (reprise possible). Chaque masque produit est vérifié (grille de la T1, valeurs 0/1, non vide) avant d'entrer dans les dérivés ; un lot incomplet arrête le traitement avec la fin du journal du worker. La vérification par SHA-1 de l'entrée n'est plus nécessaire : HD-BET lit directement notre fichier BIDS. Gains : pas de quota GPU Kaggle consommé, pas d'aller-retour d'archive. Pipeline réel validé sur sub-000 (63 s, RAM 3,2 Go, mémoire GPU 4,1 Go) ; contrôle visuel dans `notebooks/04_brainmask_check.ipynb`. Tests adaptés (faux worker, sans GPU) : contrôle des masques, lots et reprise, échec d'un lot, commande du worker, transport sur la grille FLAIR.
**Exécution complète (2026-10-04).** 170 masques, 0 échec, 0 alerte. 47 min (16,8 s par sujet, modèle chargé une fois par lot), pic de RAM 4,9 Go, pic de mémoire GPU 4,1 Go stable sur les 9 lots. Volume du cerveau : médiane 1 357 ml [1 020-1 810] ; médianes par scanner de 1 294 ml (Singapour) à 1 435 ml (Amsterdam Philips), écarts qui peuvent refléter les populations autant que les scanners. 169 masques d'un seul tenant ; sub-108 a une composante parasite de 4 voxels (négligeable). Couverture du cerveau par la FLAIR : 0,998 à 1,001 sur les 60 sujets d'entraînement (la FLAIR contient tout le cerveau). **Constat** : jusqu'à 5,5 % du cerveau HD-BET tombe hors du masque de tête N4 sur 3 sujets de Singapour (sub-055, 065, 068) : ce sont des zones sombres de la T1 (intensité médiane 81 contre 195 dans le cerveau), sous le seuil d'Otsu à cause du fort champ de biais de ces T1 (J-030, signal 2). N4 y a donc extrapolé le champ au lieu de l'estimer ; l'accord avec SPM12 reste élevé (0,89 à 0,94), mais cela renforce la vérification prévue : refaire N4 avec le masque HD-BET.

---

## J-034 — N4 : masque d'Otsu ou masque du cerveau HD-BET, comparés à SPM12
2026-10-04 · Mesurée (décision de bascule en attente de validation)

**Protocole.** Sur les 60 sujets d'entraînement, trois corrections comparées à l'image brute : N4 avec le masque de tête d'Otsu (actuel), N4 avec le masque du cerveau HD-BET (calculé en mémoire, rien d'écrasé), SPM12 des organisateurs. Mesures **dans le cerveau HD-BET**, hors HSB et « autre pathologie » : coefficient de variation (CoV) de l'intensité, et dispersion de la luminosité du tissu d'une coupe à l'autre. Pour la T1, les champs estimés à 1 mm sont ramenés sur la grille FLAIR par notre transformation rigide et comparés au champ SPM12 de la T1 alignée. Tables : `results/tables/n4_mask_comparison_*.csv`.
**Résultats.**
- *FLAIR* : gain faible avec HD-BET (CoV médian -0,09 % relatif, meilleur sur 36/60 ; dispersion entre coupes un peu plus basse sur les 3 sites). Champs Otsu et HD-BET très proches à Amsterdam et Utrecht (corrélation 0,99 et 0,95), moins à Singapour (0,88). Accord avec SPM12 à Singapour : 0,65 (Otsu) -> 0,74 (HD-BET). Les deux N4 battent SPM12 sur les 60 sujets.
- *T1* : gain net avec HD-BET (CoV médian -1,0 % relatif, jusqu'à -5,9 %, meilleur sur 52/60 ; dispersion entre coupes nettement plus basse : Amsterdam 0,068 -> 0,053, Singapour 0,057 -> 0,049, Utrecht 0,094 -> 0,086). N4-HD-BET bat SPM12 sur 60/60 sujets (N4-Otsu : 45/60). Accord avec SPM12 inchangé ou légèrement meilleur (0,94 à 0,98).
**Biais de la mesure.** L'homogénéité est mesurée dans le masque même qu'utilise N4-HD-BET : cela favorise légèrement cette variante. Le CoV mélange substance blanche, grise et liquide : seule la comparaison relative entre méthodes a un sens.
**Lecture.** Le masque du cerveau apporte surtout de la **cohérence** : même définition de la région d'estimation sur tous les sites (le masque d'Otsu changeait de nature selon le scanner, J-030), et des T1 mieux corrigées. Le gain sur la FLAIR, canal principal de nnU-Net, est faible.

---

## J-035 — Entraînement nnU-Net en local : mesures de faisabilité
2026-10-04 · Mesurée (décision en attente)

**Protocole.** Environnement temporaire CUDA (`uv run --no-project --with nnunetv2==2.8.1 --with torch==2.14.1+cu130`), données locales `Dataset001_WMH`, fold 0, `nnUNetTrainer_5epochs` arrêté après 3 epochs, sous surveillance (RAM du processus, mémoire GPU, échantillonnage de l'utilisation GPU/CPU, température, puissance). Plafond PyTorch `set_per_process_memory_fraction` à 6 Go (l'affichage Windows occupe ~1,9 Go des 8 Go).
**Prétraitement nnU-Net local.** Plan identique à Kaggle (patch 48 × 224 × 192, lot 2) ; 172 s ; pic de RAM 6,7 Go ; 873 Mo sur disque.
**Essai 1, plan standard, 6 processus d'augmentation.** Échec à la première epoch : `CUDA out of memory` propre au plafond de 5,6 Go autorisé (le plafond a empêché le débordement dans la RAM). RAM du système montée à 23,4 Go sur 24. **Le plan standard ne tient pas sur la carte.**
**Essai 2, plan à cible mémoire GPU de 5 Go (`-gpu_memory_target 5`, plans `nnUNetPlans_5GB`), 3 processus d'augmentation.** Patch réduit à 40 × 192 × 160 (même architecture, lot 2). **114 s par epoch**, stable sur 3 epochs ; mémoire GPU 4,2 Go ; RAM du processus 6,0 Go, du système jusqu'à 21 Go sur 24 ; GPU utilisé à 97 %, CPU à 35 % ; **température GPU jusqu'à 85 °C**, 78 W, horloge ~2,47 GHz. `torch.compile` est désactivé par nnU-Net sous Windows.
**Lecture.** Même vitesse que Kaggle en 2 × T4 (110 s par epoch), mais avec un patch plus petit (moins de contexte, différent du plan standard) ; 250 epochs ≈ 8 h de GPU à pleine charge sur un portable (85 °C dès les premières minutes, comportement thermique sur plusieurs heures non mesuré), machine inutilisable pour autre chose et RAM système proche de la saturation.

---

## J-036 — Chaîne de prétraitement réordonnée : N4 avec le masque du cerveau HD-BET
2026-10-04 · Actée (remplace l'ordre de J-010/J-030/J-031 ; la variante Otsu est archivée)

**Décision.** Après la comparaison J-034, N4 utilise le masque du cerveau HD-BET sur tous les sites. Pour que le masque soit disponible sur la grille FLAIR avant N4, l'ordre devient :
1. HD-BET sur la T1 brute (3.3) ;
2. **recalage rigide des images brutes** T1 -> FLAIR (3.2) ; l'information mutuelle est robuste au champ de biais (0,98-0,99 contre elastix sur images brutes au test de faisabilité) ;
3. masque du cerveau transporté sur la grille FLAIR (3.3, `genericLabel`) ;
4. N4 de la FLAIR et de la T1 avec leur masque du cerveau, puis T1 corrigée rééchantillonnée sur la grille FLAIR (linéaire) (3.1) ;
5. export nnU-Net (5.1).
Chaque étape ne dépend que des précédentes ; la transformation ne dépend plus de la correction de biais.
**Archive.** Les sorties de la première variante (N4 Otsu, transformations calculées sur les images N4 Otsu, masques du cerveau transportés avec elles : 600 fichiers, 5,2 Go) sont déplacées dans `data/derivatives/wmh-multisite-otsu/` pour le rapport de comparaison. `preproc.n4.mask: otsu` reproduit cette variante.
**Code.** `preproc/register.py` (entrées brutes, sortie : transformation seule, fonction `resample_onto_flair`), `preproc/bias.py` (masque selon le config, rééchantillonnage de la T1), constante `PIPELINE` déplacée dans `preproc/__init__.py`.

---

## J-037 — Réglages de l'entraînement nnU-Net
2026-10-04 · Actée

| Réglage | Choix | Raison |
|---|---|---|
| Architecture | ResEnc M (`nnUNetPlannerResEncM`, plans `nnUNetResEncUNetMPlans`, cible 8 Go de mémoire GPU) | recommandée par les auteurs depuis « nnU-Net Revisited » (Isensee et al., MICCAI 2024) ; tient sur une T4 (16 Go) |
| Configuration | 3d_fullres, fold 0 (48 entraînement / 12 validation, J-032) | budget d'un seul fold |
| Epochs | 250 (trainer personnalisé `nnUNetTrainerWMH_250`, sauvegarde toutes les 50 epochs) | le taux d'apprentissage décroît sur la durée totale : régler sur ce qu'on termine (1000 epochs ≈ 31 h, au-delà du quota et de l'échéance) |
| Perte | Dice + entropie croisée (défaut) | la plus classique, éprouvée |
| Sur-échantillonnage des lésions | 33 % des patches (défaut) | raisonnable pour des lésions petites et rares |
| Miroirs au test | activés (défaut) | |
| Canaux | FLAIR + T1 | |
| Exécution | Kaggle 2 × T4 (DDP), `torch.compile`, 4 processus d'augmentation | J-019 ; l'essai local (J-035) n'est pas retenu |
| 2e entraînement | même configuration avec `nnUNetTrainerWMH_DA5_250` (augmentation forte) | quantifier l'effet de l'augmentation sur la robustesse aux scanners inconnus, à budget égal |
**Durée.** Le temps par epoch de ResEnc M sur 2 × T4 n'est pas mesuré (benchmark fait avec le U-Net standard : 110 s) ; ResEnc M est plus lourd. Le notebook `kaggle/03_nnunet_train.ipynb` s'arrête proprement après une sauvegarde si la suivante dépasse le budget de session (11,3 h) et reprend dans une nouvelle session (`--c`). La décision de lancer le 2e entraînement se prendra sur le temps mesuré et le quota restant.

---

## J-038 — Entraînement principal : résultat de la session Kaggle
2026-10-04 · Constat

**Déroulement.** ResEnc M, 3d_fullres, fold 0, 250 epochs, 2 × T4. Plan ResEnc M : patch 40 × 224 × 192 (le planificateur réduit l'épaisseur pour sa cible de 8 Go), lot de 2. 1re epoch 336 s (compilation), puis **149 s par epoch, stable** (≈ +35 % par rapport au U-Net standard, 110 s). Sauvegardes aux epochs 50, 100, 150, 200 ; le garde-fou de session ne s'est pas déclenché (prévision correcte : 11,08 h < 11,3 h). Fin d'entraînement à 10,55 h, validation des 12 cas en ~3 min, session ≈ 10,7 h.
**Résultat.** Dice moyen de validation (fold 0, 12 cas, 4 par site, label 2 ignoré) : **0,815**. Ce chiffre n'est pas comparable au classement du challenge (110 cas de test, dont 20 de scanners inconnus) ; il montre que 250 epochs suffisent pour obtenir un modèle compétitif sur les scanners vus.
**Incident.** La cellule d'export des métriques a échoué (`KeyError: 'epoch'`) : l'expression régulière qui repère les lignes « Epoch N » exigeait une fin de ligne immédiate et avait été testée sur un journal fabriqué, pas sur un vrai. L'entraînement, le modèle et la validation ne sont pas affectés ; les métriques sont recalculables à partir de `training_log_*.txt`. Correctifs dans le notebook : expression tolérante aux espaces finaux, export qui ne peut plus faire échouer la session, pseudo-Dice ajouté à la ligne d'avancement.

---

## J-039 — Recalage signalé : sub-146
2026-10-04 · Actée (validée par l'utilisateur)

**Constat** (`notebooks/05_registration_sub146.ipynb`). Seul sujet signalé sur 170 (cas de test, Amsterdam GE 3T) : corrélation avec elastix 0,905, écart médian 2,0 mm dans le cerveau (max 3,3 mm), rotation relative 1,2°. Information mutuelle avec la FLAIR dans le cerveau : 0,245 pour la T1 d'elastix contre 0,162 pour la nôtre (sujet de référence sub-114 : 0,289 / 0,297). Relancé depuis la solution d'elastix, ANTs s'en éloigne (1,5 mm, MI 0,19) ; restreint au cerveau de la T1 (`moving_mask`), il échoue (20 mm). La FLAIR de ce sujet est très bruitée et de contraste atypique : faible lien statistique avec la T1, critère peu discriminant.
**Proposition.** Pour sub-146, utiliser la transformation d'elastix fournie par les organisateurs (exception documentée), et signaler la FLAIR atypique au contrôle qualité (Phase 4).
**Mise en oeuvre.** `preproc.rigid.use_elastix: [sub-146]` dans le config. `register.py` convertit les paramètres elastix en transformation ITK (`Euler3DTransform`, écrite par SimpleITK) à l'emplacement habituel ; ANTs l'applique comme la sienne (vérifié : T1 brute ainsi alignée contre `orig/T1`, corrélation 0,9998). La transformation ANTs est conservée sous `sub-146_from-T1w_to-FLAIR_mode-image_desc-ants_xfm.mat`. Le rapport de recalage indique `status = elastix` et ne signale plus le sujet (comparaison triviale). Refaits pour ce sujet : masque du cerveau sur la grille FLAIR, N4 de la FLAIR (son masque a changé), T1 corrigée sur la grille FLAIR. Le N4 de la T1 (grille T1) n'en dépend pas. Le sujet est un cas de test : l'entraînement n'est pas concerné.
**Résultat de l'entraînement principal** (complément à J-038, lu dans `validation/summary.json`) : Dice par cas de 0,668 (sub-023, 0,9 ml) à 0,911 (sub-037, 75 ml) ; moyenne par site 0,816 (Amsterdam), 0,814 (Singapour), 0,815 (Utrecht). Pseudo-Dice (moyenne sur 10 epochs) 0,829 / 0,836 / 0,848 / 0,846 / 0,839 aux epochs 50 / 100 / 150 / 200 / 250 : plateau à partir de ~150 epochs. Entraînement confirmé sur le dataset N4 HD-BET (version Kaggle vérifiée par l'utilisateur).

---

## J-040 — Inférence nnU-Net en local sur GPU
2026-10-04 · Actée

**Mise en oeuvre.** `seg/nnunet_infer.py` (orchestration) et `seg/nnunet_infer_worker.py` (exécuté par chemin dans l'environnement temporaire CUDA, `nnunetv2==2.8.1`, `torch==2.14.1+cu130`), même schéma que HD-BET (J-033) : lots de 37 sujets sous watchdog (RAM 10 Go, mémoire GPU 6 Go), plafond PyTorch 5,5 Go, reprise, contrôle de chaque prédiction (grille FLAIR, valeurs 0/1). Le trainer personnalisé `nnUNetTrainerWMH_250` n'existe que sur Kaggle : le réseau est reconstruit depuis `plans.json` avec `nnUNetTrainer.build_network_architecture` (le trainer personnalisé ne change que la durée et la période de sauvegarde) et les poids de `checkpoint_final.pth` sont chargés (`nnUNetPredictor.manual_initialization`). Miroirs au test activés, pas de fenêtre de 0,5. Sorties : `sub-XXX_space-FLAIR_desc-resencm_dseg.nii.gz`.
**Validation.** Sur deux cas de validation du fold 0 (sub-106, sub-023), prédiction locale **identique au voxel près** à celle de Kaggle (0 voxel différent ; Dice 0,823 et 0,668 dans les deux cas).
**Exécution (110 cas de test).** 16,2 min (~9 s par sujet), pic de mémoire GPU 1,9 Go, RAM 4,9 Go, 0 échec. Contrôle rapide (Dice calculé par nous, label 2 ignoré ; **pas encore le script officiel**, Phase 6) : moyenne 0,803 ; par scanner : Singapour 0,834, Amsterdam GE 1,5T 0,804 (inconnu), Amsterdam GE 3T 0,796, Utrecht 0,786, Amsterdam Philips 3T 0,776 (inconnu). Cas les plus faibles : de petites charges lésionnelles (≤ 2 ml). sub-146 (transformation elastix, J-039) : Dice 0,78.

---

## J-042 — Évaluation officielle du modèle principal (110 cas de test)
2026-10-04 · Constat

**Méthode.** `evaluate/wmh_challenge.py` : fonctions du script officiel (`hjkuijf/wmhchallenge`, commit 2e628cb, MIT) copiées sans modification ; `evaluate/metrics.py` les applique à chaque sujet (Dice, HD95, AVD, rappel et F1 par lésion ; label 2 retiré de la prédiction), puis moyenne par groupe comme le classement. Sorties : `results/tables/evaluation_resencm.csv` (par sujet) et `_summary.csv`.
**Piège trouvé.** `getImages` isole les HSB avec `BinaryThreshold(test, 0.5, 1.5)` : sur une vérité terrain stockée en entiers, SimpleITK convertit ces seuils en 0 et 1, tout le volume devient « HSB » et le masque « autre pathologie » efface la prédiction (toutes les métriques à 0, sans erreur). Les fichiers du challenge sont en float32, donc le classement n'est pas concerné ; notre fonction passe toujours au code officiel une copie float32 de la référence. Test unitaire avec une référence en uint8.
**Résultats (ResEnc M, fold 0, 250 epochs, 110 cas).** Dice 0,803 ; HD95 6,12 mm ; AVD 15,4 % ; rappel lésionnel 0,728 ; F1 lésionnel 0,783. Scanners vus (90) : Dice 0,805, HD95 5,60 ; **scanners inconnus (20)** : Dice 0,790, HD95 8,48, rappel 0,754, F1 0,802. Par scanner, Dice de 0,776 (Amsterdam Philips, inconnu) à 0,834 (Singapour).
**Position au classement** (formule officielle, 57 équipes publiées + nous, valeurs du classement arrondies à 2 décimales) : **9e sur 58**. Rang par métrique : Dice 2e, HD95 6e, AVD 1er, F1 2e, **rappel 22e**. Point faible : la détection des petites lésions (rappel), point fort : le volume total (AVD) et la précision des contours.
**Limites.** Un seul fold sans ensemble (les meilleures équipes utilisent souvent des ensembles) ; comparaison sur des valeurs du classement arrondies ; écarts entre scanners à confirmer par intervalles de confiance (6.2).

---

## J-043 — WMH-SynthSeg sur les 110 cas de test (remplace le sous-ensemble de J-020)
2026-10-04 · Actée

**Raison.** Comparer deux méthodes exige le **même ensemble de sujets** : comparaison appariée sujet par sujet (plus puissante statistiquement) et comparaison au classement, calculé sur les 110 cas. Le sous-ensemble de 50 sujets de J-020 n'était justifié que par le coût. Or une session Kaggle CPU ne consomme pas le quota GPU et 110 sujets (~5 min chacun, ~10 h) tiennent dans les 12 h d'une session.
**Mise en oeuvre.** `kaggle/05_wmhsynthseg.ipynb`, 110 sujets répartis en deux moitiés équilibrées par scanner (`PART = "A"` / `"B"`, 55 sujets, ~5 h chacune, alternance des sujets à l'intérieur de chaque scanner ; `"ALL"` en une session). Archives `wmhsynthseg_predictions_<PART>.zip`, à décompresser à la racine du projet ; évaluation avec `--model wmhsynthseg`.

---

## J-044 — Seuillage FLAIR (M3) : z-score dans le cerveau, profondeur minimale, taille minimale, réglage global sur le train
2026-10-04 · Actée

**Contexte.** M3 est la méthode classique de J-009 : les HSB sont brillantes en FLAIR, donc « les voxels plus clairs qu'un seuil ». Elle sert de référence basse et mesure la sensibilité d'une règle d'intensité fixe au scanner.
**Décision.** `seg/threshold.py`, sur la FLAIR N4 (grille FLAIR) :
1. z-score robuste **dans le masque du cerveau HD-BET** (médiane, 1,4826 x MAD ; règle A de J-028) ;
2. seuil t sur z ;
3. exclusion des voxels à moins de e mm du bord du cerveau ;
4. suppression des composantes connexes (26-connexité) de moins de m mm³.
Un seul triplet (t, e, m) pour tous les scanners, choisi par recherche sur grille sur les 60 cas d'entraînement (Dice moyen par sujet, label 2 ignoré ; J-013). Le meilleur seuil par scanner est publié **pour information** (mesure de la dépendance au scanner), jamais utilisé.
**Preuves (60 cas d'entraînement, exploration avant implémentation).** 100 % des voxels de lésion sont dans le masque HD-BET. z médian des lésions 2,8 (Amsterdam 3,0, Utrecht 3,3, **Singapour 2,2**) ; 10e centile des lésions 1,6, contre un 99e centile du tissu non lésionnel de 1,9 : les distributions se recouvrent, un seuil seul ne peut pas être parfait. Les lésions sont profondes : 5 % des voxels de lésion à moins de 16 mm du bord du cerveau en médiane (8 mm au minimum), alors que 11 % (jusqu'à 57 %) des voxels non lésionnels avec z > 3 sont à moins de 3 mm du bord. Sur deux sujets, l'optimum individuel tombait au bord de la première grille (e = 10 mm, m = 40 mm³) : grille élargie (e jusqu'à 16 mm, m jusqu'à 160 mm³) ; le réglage signale tout optimum resté au bord.
**Alternatives écartées.** Restriction à la substance blanche par une segmentation de la T1 : les HSB sont iso- ou hypo-intenses en T1, comme la substance grise, donc une substance blanche définie sur la T1 exclurait une partie des lésions ; une restriction par atlas dépend de SyN (3.6), peu fiable chez les sujets atrophiés (notebook 06). Seuil par scanner : contraire à l'objet de M3 (règle unique) et réglé sur 20 sujets seulement. Moyenne et écart type au lieu de médiane et MAD : tirés par les lésions elles-mêmes, donc par la charge lésionnelle.
**Conséquences.** Le critère est le Dice : il favorise les grandes valeurs de e et m, au détriment des petites lésions (rappel lésionnel). Effet à lire dans l'évaluation officielle. Prédictions `desc-threshold_dseg`, évaluées par `--model threshold`.
**Résultat du réglage (60 cas d'entraînement, 1 674 combinaisons, 19 min).** t = 2,2 ; e = 14 mm ; m = 40 mm³ (aucun au bord de la grille). Dice d'entraînement 0,585 (Amsterdam GE 3T 0,615, Utrecht 0,595, Singapour 0,546). Meilleur réglage par scanner, pour information : Singapour veut un seuil bien plus bas (t = 1,7 contre 2,4 à 2,6 ailleurs), conforme au contraste plus faible de ses lésions (z médian 2,2) : la règle unique pénalise Singapour, comme attendu.
**Évaluation officielle (110 cas de test).** Dice 0,595 ; HD95 22,7 mm ; AVD 72,7 % ; rappel lésionnel 0,229 ; F1 lésionnel 0,284. Par scanner, Dice de 0,565 (Utrecht, AVD 142 % : forte surestimation du volume) à 0,634 (Amsterdam GE 1,5T). Scanners inconnus 0,622 contre connus 0,589 : pas de chute hors domaine, la normalisation dans le cerveau absorbe l'essentiel de l'écart d'intensité ; l'écart vient surtout d'Utrecht. Le rappel lésionnel très bas confirme le coût du critère Dice (e et m élevés éliminent les petites lésions). Référence : nnU-Net (M1) Dice 0,803, rappel 0,728 (J-042).


---

## J-045 — Ventricules latéraux pour 7.2 : segmentation anatomique de WMH-SynthSeg
2026-10-04 · Actée (confirmée par le contrôle `CHECK` sur 5 sujets, notebook 06 section 10)

**Contexte.** 7.2 classe les HSB en périventriculaires ou profondes selon leur distance aux ventricules latéraux : il faut des ventricules exacts, surtout chez les sujets atrophiés, qui ont le plus de lésions. Notebook `06_atrophy_vs_template`, sections 7 et 8 : l'ajustement C de l'atlas (3.6) **s'arrête sur la bande de 10 mm** autour des ventricules de l'atlas chez les sujets très atrophiés (18 et 27 ml collés à la limite pour sub-021 et sub-000 ; corps des ventricules couverts à moitié sur les coupes hautes de sub-000). Élargir la bande fait fuir l'ajustement dans les citernes et la fissure inter-hémisphérique (sub-049 : 4 → 27 ml à 30 mm). Un prototype d'extension par les seules zones de liquide épaisses (« C2 » : ouverture morphologique de 3 mm, bande de 30 mm, profondeur > 16 mm) récupère les ventricules oubliés (sub-000 : 109 → 152 ml) sans rien changer aux sujets corrects, mais fuit encore chez sub-021 (13,7 ml ajoutés sur la ligne médiane : sillon calleux, 3e ventricule, citernes).
**Décision.** Utiliser la **segmentation anatomique de WMH-SynthSeg** : en plus des HSB (label 77), le modèle segmente les ventricules latéraux gauche et droit (labels 4 et 43, vérifié dans `inference.py`), le 3e ventricule et le liquide extracérébral sous des labels distincts. Le notebook Kaggle 05 conserve désormais la segmentation complète ; il est relancé sur les **170 sujets** (3 parties de 57, 57 et 56 sujets, équilibrées par scanner) après un contrôle `CHECK` sur les 5 sujets du notebook 06. Les sessions A et B de la version précédente (HSB seulement, 110 sujets de test) sont arrêtées : leurs sorties ne sont écrites qu'en fin de session, et la nouvelle version refait les mêmes HSB.
**Raisons.** Réseau entraîné à reconnaître l'anatomie sur des formes très variées : ne dépend ni de SyN ni d'une bande ; sépare ventricule latéral, 3e ventricule et citernes (nos fuites) ; donne un label propre aux HSB, donc une lésion collée à la paroi n'est pas absorbée par le ventricule (ce qu'un seuil sur la FLAIR ne sait pas faire).
**Alternatives écartées.** C seul (sous-estime les grands ventricules) ; C2 (fuites) ; SyN CC pour tous (50 min par sujet avec 12 threads, ~140 h) ; ventricules sur la T1 (les HSB sont sombres en T1 et se confondraient avec le liquide) ; WMH-SynthSeg en local (26 à 29 Go de RAM par sujet pour 25,5 Go sur le PC, J-007).
**Vérification (`CHECK`, 5 sujets d'entraînement, 2 scanners).** Notebook Kaggle 05 : 5 sujets sur 5, 7,5 min et 26,1 Go de RAM par sujet ; masques sur la grille FLAIR (même forme et même affine que la FLAIR N4), binaires, cohérents avec la segmentation complète. Notebook 06, section 10, volumes en ml (C / C2 / WMH-SynthSeg) et Dice C2 contre WMH-SynthSeg : sub-049 3,6 / 3,7 / 16,4 (0,29) ; sub-002 38,7 / 39,0 / 53,6 (0,82) ; sub-053 9,9 / 9,9 / 18,2 (0,67) ; sub-021 95,7 / 131,1 / 119,2 (0,81) ; sub-000 109,1 / 152,1 / 162,2 (0,91). Ce que C2 ajoute à C, classé selon le label WMH-SynthSeg des mêmes voxels : sub-000, 37,9 ml de ventricule latéral sur 43 (vraie récupération) ; sub-021, 21,3 ml de LCS extra-cérébral et 1,2 ml de 3e ventricule pour 12,0 ml de ventricule latéral (la fuite de la section 8c, confirmée).
**Lecture.** WMH-SynthSeg est plus grand que C chez tous les sujets (9 à 19 ml), pour deux raisons visibles sur les figures : (1) un liseré d'environ un voxel autour des ventricules, probablement les voxels de paroi en volume partiel (coupes de 3 mm) que le seuil de C exclut ; écart systématique, négligeable devant le seuil de 10 mm de 7.2, mais à garder en tête si l'on compare des volumes ventriculaires ; (2) des cornes occipitales et temporales que C manque chez les sujets à ventricules fins (sub-049, sub-053) : C sous-estime donc aussi les petits ventricules, pas seulement les grands. Réserve : 5 sujets, tous d'entraînement, 2 scanners sur 5 ; d'où le contrôle sur les 170 sujets prévu en 7.2.
**Conséquences.** 7.2 utilise WMH-SynthSeg (labels 4 + 43) pour tous les sujets. L'atlas (3.6) est **arrêté au prototype** : `atlas.py` (SyN par défaut + C) et le notebook 06 sont conservés comme trace de la démarche, mais SyN n'est pas lancé sur les 170 sujets (il n'a tourné que pour sub-000, sub-002 et les 3 sujets du benchmark). Un sujet sur lequel WMH-SynthSeg échouerait serait relancé sur Kaggle, ou traité par atlas + C pour ce sujet seul. C2 reste un prototype du notebook 06, jamais intégré à `atlas.py`. 7.5 (carte MNI, P2) supposerait SyN sur tous les sujets : non prévu dans le délai.

---

## J-046 — Accord inter-observateurs : plafond humain, et comparaison appariée avec nnU-Net sur 12 cas
2026-10-04 · Actée

**Contexte.** La référence O1 est elle-même la délimitation d'un expert : aucune méthode ne peut viser un Dice de 1. D'après le `readme.pdf` du jeu de données : O1 a délimité toutes les images, **O2 n'a rien délimité** mais a relu toutes les délimitations de O1 (O1 corrigeant les erreurs signalées) ; la référence est donc « O1 après relecture par O2 ». **O3 et O4 ont délimité les seuls 60 scans d'entraînement** (labels 0/1, sans label 2), eux aussi relus : l'accord inter-observateurs ne peut pas être mesuré sur le test.
**Décision.** `evaluate/interobserver.py` : O3 et O4 notés contre O1 avec **les mêmes métriques officielles** que les méthodes (label 2 de O1 ignoré) ; O4 contre O3, avec le label 2 de O1 reporté sur O3 pour que les zones « autre pathologie » soient ignorées dans les deux comparaisons. Dice, HD95 et F1 lésionnel sont symétriques ; AVD et rappel lésionnel dépendent de l'observateur pris comme référence et ne sont donnés qu'à titre indicatif pour O4 contre O3. **Comparaison appariée** : les 12 cas de validation du fold 0 (vérifiés absents de l'entraînement de ce modèle, `splits_final.json`), prédits par nnU-Net en fin d'entraînement avec le modèle final utilisé sur le test, sont les seuls scans où M1 et les humains sont comparables sur les mêmes images.
**Résultats (60 scans d'entraînement).** Dice contre O1 : O3 0,770, O4 0,785 ; O4 contre O3 : 0,759. HD95 de 6,8 à 8,2 mm, rappel lésionnel de 0,65 à 0,67. Sur les 12 scans partagés : M1 0,815 (HD95 4,3 mm, F1 0,762), O3 0,757, O4 0,781. Sur le test (J-042), M1 atteint 0,803.
**Lecture.** M1 est **au niveau du plafond humain**, et le dépasse légèrement contre O1. Ce n'est pas « mieux qu'un expert » : M1 a appris le style de délimitation de O1, alors que O3 et O4 sont des experts indépendants avec leurs propres conventions (bord des lésions, petites lésions). Réserves : 12 scans appariés seulement, tous de scanners vus à l'entraînement ; les tests statistiques et intervalles de confiance relèvent de 6.2.

---

## J-047 — Statistiques : bootstrap stratifié par scanner, Wilcoxon + Holm, Mann-Whitney
2026-10-04 · Actée

**Contexte.** Une moyenne seule ne dit pas si un écart est réel ou dû aux sujets tirés (10 à 30 par scanner).
**Décision.** `evaluate/stats.py`, sur les tableaux de 6.1 (test seulement) : (1) moyenne et IC à 95 % par bootstrap **percentile, stratifié par scanner** (tirage avec remise à l'intérieur de chaque scanner, effectifs conservés), 2 000 tirages, graine fixée ; (2) méthodes comparées **sujet par sujet** : différence moyenne, IC bootstrap stratifié, **test de Wilcoxon** signé, **correction de Holm** sur l'ensemble des tests appariés ; (3) scanners inconnus contre connus (sujets différents) : différence des moyennes, IC bootstrap (chaque groupe tiré séparément), **test de Mann-Whitney**. HD95 indéfini (rien de prédit) : sujet exclu pour cette métrique seulement, compté dans `n_missing`. Tests unitaires : Holm contre un calcul à la main, stratification (moyenne bootstrap constante quand les strates sont conservées), moyennes identiques aux moyennes simples, reproductibilité, différence appariée connue, HD95 manquant.
**Alternatives écartées.** IC par loi normale (Dice bornés et asymétriques) ; bootstrap non stratifié (le mélange de scanners varierait d'un tirage à l'autre) ; test t (suppose la normalité, sensible aux cas extrêmes) ; Bonferroni (plus conservateur que Holm, sans gain).
**Résultats (110 cas de test, M1 et M3).** M1 Dice 0,803 [0,784-0,820] ; M3 0,595 [0,560-0,629] ; M1 - M3 = +0,208 [0,185-0,231], p Holm < 1e-18, et M1 meilleur sur les 5 métriques. Scanners inconnus contre connus, M1 : Dice -0,015 [-0,066 ; +0,030], p = 0,45 ; HD95 +2,9 mm [-2,1 ; +8,7], p = 0,96 : **aucune dégradation démontrée** hors domaine ; l'écart de HD95 des moyennes (8,5 contre 5,6 mm, J-042) vient de quelques sujets extrêmes, pas de l'ensemble. M3 : l'AVD est plus faible sur les inconnus (-42 points, IC excluant 0) mais Mann-Whitney ne conclut pas (p = 0,16) : effet porté par les fortes surestimations de quelques sujets d'Utrecht, pas par un décalage général. Réserve : 20 sujets inconnus issus de 2 scanners ; un écart peut refléter la population autant que le scanner (J-004).

---

## J-048 — Biomarqueurs : volumes (7.1) et structure des lésions (7.3)
2026-10-05 · Actée (normalisation par l'ICV à ajouter, voir Conséquences)

**Décision.** `biomarkers/volumes.py`, `biomarkers/lesions.py`, `biomarkers/table.py` -> `results/tables/biomarkers.csv` (une ligne par sujet et source). Volume HSB en ml (voxels x volume exact du voxel, qui varie de 1,75 à 4,72 mm³ selon le scanner) ; fraction du masque du cerveau HD-BET (règle A de J-028) ; zones de label 2 de O1 retirées de toutes les segmentations (comme l'évaluation officielle). Lésions = composantes connexes en 26-connexité, **comme le script officiel** (`SetFullyConnected(True)`, test unitaire contre SimpleITK) : nombre brut, nombre de lésions d'au moins 10 mm³ (comparable entre sites : ~2 voxels sur la grille la plus grossière, ~6 sur la plus fine), taille médiane, plus grande lésion et sa part du volume (confluence). Sources : test = O1 + chaque modèle présent (M1, M3, puis M2 et DA5) ; entraînement = O1, O3, O4 (dispersion humaine des biomarqueurs, pour 7.4 ; charge lésionnelle des sujets d'ajustement de ComBat, 8.3) + M1 sur les 12 cas de validation du fold 0. M1 (hors validation) et M3 ne sont pas mesurés sur l'entraînement (appris ou réglés dessus).
**Contrôle.** Volumes du test identiques à `true_ml` / `pred_ml` de l'évaluation officielle à 5e-5 ml près (arrondi des tableaux). Un premier passage donnait jusqu'à 0,014 ml d'écart : `io.describe` arrondit la taille des voxels à 4 décimales (affichage) ; ajout de `io.voxel_volume_mm3` (valeur exacte, testée), utilisé pour les volumes. `threshold.py` et `atlas.py` utilisent encore la valeur arrondie (écart relatif ~1e-4 sur une taille minimale ou une distance : sans effet pratique).
**Premiers constats (médianes).** Test : volume O1 9,5 ml, M1 10,4, M3 10,7 ; lésions O1 57, M1 47, M3 20 (M3 élimine les petites lésions, taille médiane 108 mm³ contre 17). Entraînement : O1 15,1 ml, O3 13,0, O4 12,0 ; lésions O1 53,5, O3 38, O4 41,5 : **O1 délimite systématiquement plus que les deux autres experts**. Par scanner (O1, 170 sujets), Amsterdam Philips, aux voxels les plus fins (1,75 mm³), compte 91,5 lésions brutes contre 47 à 65 ailleurs, pour un volume moyen ; avec le seuil de 10 mm³, 54 contre 31 à 47 : l'écart se réduit, cohérent avec l'effet de résolution (sans exclure une différence de population).
**Conséquences.** Normalisation par le **volume intracrânien (ICV)** à ajouter dès que la segmentation complète de WMH-SynthSeg est disponible (rappel ROADMAP 7.1) : la fraction du masque HD-BET augmente avec l'atrophie. **Ajoutée le 2026-10-05** (colonnes `icv_ml`, `wmh_pct_icv`, J-053) : ICV médian 1 450 ml ; charge médiane (O1) de 0,44 % de l'ICV (Amsterdam GE 3T) à 1,18 % (Singapour).

---

## J-049 — Accord des biomarqueurs prédits avec la référence, global et par site (7.4)
2026-10-05 · Actée

**Contexte.** Un clinicien ou une étude multicentrique utilise un chiffre par patient (volume de HSB). Un biais qui dépend du site créerait un faux effet de site. Les volumes O1 du test vont de 0,8 à 195 ml ; l'écart absolu de M1 croît avec le volume (Spearman 0,71 en ml, -0,31 sur l'échelle logarithmique).
**Décision.** `biomarkers/agreement.py` : pour chaque source contre O1 (test : M1, M3, puis M2 et DA5 ; entraînement : O3, O4 sur 60 sujets et M1 sur les 12 cas de validation, référence humaine), d = ln(source / O1) par sujet. (1) Bland-Altman : biais et limites d'accord à 95 %, rapportés en rapports ; (2) ICC(A,1) (accord absolu, McGraw et Wong) sur ln(volume), IC bootstrap stratifié par scanner ; (3) biais par scanner (IC bootstrap, Wilcoxon contre 0, Holm) et test de Kruskal-Wallis entre scanners. Biomarqueur secondaire : nombre de lésions >= 10 mm³ sur ln(x + 1). Volume nul exclu de l'échelle log et compté. Figures `results/figures/bland_altman_<split>_<source>.png`. 7 tests (ICC contre un calcul à la main, Bland-Altman, strates conservées, biais de site retrouvé sur données simulées, exclusion des volumes nuls, appariement, figure).
**Alternatives écartées.** Corrélation (mesure l'association, pas l'accord : une méthode qui double tous les volumes aurait r = 1) ; Bland-Altman en ml (dominé par les fortes charges) ; ICC de cohérence (ignore un biais constant).
**Résultats (volume, test, 110 sujets).** M1 : rapport moyen 0,98, limites d'accord 0,62 à 1,55, ICC 0,982 [0,975-0,988]. M3 : 1,18, limites 0,33 à 4,14, ICC 0,814 [0,733-0,873]. **Référence humaine (entraînement)** : O3 0,97 [0,60-1,57], ICC 0,981 ; O4 0,91 [0,55-1,51], ICC 0,977 : **l'accord de M1 sur le volume est au niveau de celui de deux experts**. Le biais dépend du site pour toutes les sources (Kruskal-Wallis p < 0,001), humains compris. M1 : Utrecht 0,87 [0,80-0,95] (p Holm 0,026), Amsterdam GE 1,5T 1,10 [1,06-1,14] (p Holm 0,086), autres sites compatibles avec 1. M3 : Utrecht 1,72 [1,31-2,25] (p Holm 0,003), surtout des faibles charges surestimées 3 à 8 fois. Les humains aussi : O4 à Singapour 0,80 [0,75-0,85] (p Holm 0,001). Nombre de lésions >= 10 mm³ : ICC M1 0,89, O3 0,85, O4 0,88, M3 0,31.
**Lecture.** Un biais de site existe même entre experts : une partie du « biais de site » d'une méthode peut refléter le style de O1 sur certains sites plutôt que le scanner. Un biais associé au site n'est pas forcément causé par le scanner (population, J-004).

---

## J-050 — HSB périventriculaires et profondes : règle des 10 mm, ventricules de WMH-SynthSeg (7.2)
2026-10-05 · Actée

**Contexte.** Les échelles cliniques (Fazekas) notent séparément les HSB périventriculaires et profondes. Le découpage dépend entièrement de la qualité des ventricules (J-045).
**Contrôle CHECK (5 sujets du notebook 06).** Ventricules WMH-SynthSeg contre ajustement C de l'atlas : toujours plus grands (sub-000 162 contre 109 ml, sub-021 119 contre 96, sub-002 54 contre 39, sub-053 18 contre 10, sub-049 16 contre 4), contenant 80 à 97 % de C ; **aucun voxel de HSB (label 77) classé ventricule**. Contrôle visuel par l'utilisateur : SynthSeg retenu.
**Décision.** `biomarkers/location.py` : un voxel de HSB est périventriculaire s'il est à **10 mm ou moins** des ventricules latéraux (labels 4 et 43 de WMH-SynthSeg), profond sinon (règle voxel par voxel : une plaque confluente est partagée). Distance en mm avec la taille réelle des voxels (coupes de 3 mm). Mêmes ventricules pour toutes les sources d'un sujet (la comparaison entre sources mesure les HSB, pas les ventricules). Priorité des masques dans `config.yaml` (`biomarkers.ventricles`) : WMH-SynthSeg, puis l'ajustement C de l'atlas en secours ; colonne `ventricle_source` ; pas de découpe sans masque. Volumes périventriculaire et profond ajoutés à l'analyse d'accord de 7.4 (une méthode mesure-t-elle aussi bien les lésions profondes, petites et dispersées ?). Tests : distances anisotropes (3 coupes = 9 mm), règle des 10 mm (inclusive), plaque confluente partagée, masques vides, priorité des sources de ventricules.
**Alternatives écartées.** Règle de continuité (lésion touchant le ventricule) en plus : second chiffre à expliquer, gain secondaire, la confluence étant déjà mesurée en 7.3 (part de la plus grande lésion) ; mentionnée comme extension. Analyse de sensibilité SynthSeg contre C : comparer à une méthode déjà connue pour sous-estimer les grands ventricules n'apprend rien, et imposerait l'atlas sur les 170 sujets. **Conséquence : l'atlas par défaut ne sera lancé que pour les sujets où WMH-SynthSeg échouerait** (7.5 utilise SyN CC sur un sous-ensemble).
**Premiers résultats (5 sujets CHECK, entraînement).** Part périventriculaire de 0,39 (sub-053, faible charge) à 0,95 (sub-000) ; les trois experts concordent sur cette part (écarts de 0,01 à 0,04, sauf sub-053 : 0,39 à 0,60 pour 1 ml de lésions).

---

## J-051 — Contrôle qualité des images : IQM MRIQC, atypiques par scanner, artefacts simulés, rapport local (Phase 4)
2026-10-05 · Actée

**Décision.** `qc/iqm.py` (4.1) : sur les images **brutes** (N4 masquerait le défaut de biais, mesuré à part), indicateurs définis comme MRIQC : SNR, SNR de Dietrich, CNR, CJV, rapport SB/SG, EFC, FBER, étendue du champ de biais N4. Tissus de WMH-SynthSeg (SB 2/41 érodée d'un voxel, cortex 3/42 ; les HSB ont leur label 77 et ne sont pas dans la SB). **Règle A** (J-028) : statistiques de tissu dans le masque HD-BET. **Règle B** : fond = air hors de la tête dilatée, **zéros exacts exclus** ; EFC calculée sur les voxels non nuls. T1 mesurée **sur sa grille d'origine** (labels 1 mm transportés par la transformation rigide inverse de 3.2, plus proche voisin) : l'interpolation vers la grille FLAIR lisserait le bruit. `qc/outliers.py` (4.2) : z robuste (médiane, MAD) **par scanner** et par modalité, signé (positif = pire) ; Isolation Forest sur ces z ; classes `exclude` (z >= 5), `check` (z >= 3 ou Isolation Forest), `usable`, et `not_assessed` pour un scanner de moins de 8 images. Recommandation de contrôle visuel, jamais d'exclusion automatique. Validation : Dice de M1 par classe sur le test. `qc/artifacts.py` (4.3) : bruit, mouvement, champ de biais, images fantômes (TorchIO, graine fixe), 5 niveaux (0 = image intacte : taux de fausses alertes), sur 2 FLAIR « utilisables » par scanner ; z contre les images intactes du même scanner, Isolation Forest ajusté sur elles ; courbe de détection. `viz/report.py` (4.4) : HTML autonome (tableau, vignettes des cas signalés) dans `results/qc/`, **exclu de git** (images de patients, données CC BY-NC).
**Constats en cours de route.** Les FLAIR 2D d'Utrecht ont un contraste SB/SG presque nul (rapport 0,92 à 1,05 sur l'image brute, 0,90 à 1,00 après N4, contre 0,84 à 0,88 à Singapour) : CNR faible et CJV très grand, propres au site, d'où le jugement par scanner. Un premier essai sur les 5 sujets de CHECK classait 3 sujets d'Utrecht sur 4 en `exclude` : une médiane et une MAD de 4 valeurs n'ont pas de sens, d'où le minimum de 8 images par scanner. Sorties de cet essai supprimées.
**Correction annexe.** Condition de concurrence du watchdog (`utils/guard.py`, ROADMAP « Points à étudier ») : un sous-processus rapide pouvait se terminer entre deux appels psutil (`children()` ou `Process(pid)` levant `NoSuchProcess`), d'où les échecs intermittents des tests de `call_guarded`. Corrigée, test de non-régression ajouté ; 7 passages complets de la suite sans échec ensuite.
**Correction après le calcul complet.** Le CJV de la FLAIR a dominé un premier passage (z jusqu'à 209, 31 sujets de test signalés, avec un Dice **meilleur** que les autres) : en FLAIR, SB et SG ont presque la même intensité (rapport 0,87 à 0,99 selon le scanner), et le CJV divise par leur écart. Le CJV et le rapport SB/SG sont désormais mesurés mais **jugés pour la T1 seulement** (`iqm.JUDGED`), comme dans MRIQC, conçu pour des contrastes T1/T2.
**Résultats (170 sujets).** 4.1 en 15 min (3 sujets en parallèle). 4.2 : 136 `usable`, 30 `check`, 4 `exclude` (sub-003, sub-009, sub-044 d'Utrecht ; sub-160 d'Amsterdam Philips, rapport SB/SG de la T1, z = 11,5), répartis sur tous les scanners. **Validation : aucun lien avec la qualité de segmentation** : Dice de M1 0,813 pour les 19 sujets de test signalés contre 0,826 (Mann-Whitney p = 0,90 ; Spearman score / Dice -0,15). Les images du challenge ont été sélectionnées, et nnU-Net absorbe les variations de qualité présentes. 4.3 (10 FLAIR, 2 par scanner) : aucune fausse alerte au niveau 0 ; **champ de biais** détecté de 60 à 100 % aux niveaux 2 à 4 ; **bruit** de 30 à 70 % ; images fantômes surtout au niveau le plus fort (60 %) ; **mouvement non concluant** : les 2 sujets « détectés » (sub-162, sub-163) le sont dès le niveau 1 et à tous les niveaux, parce que la simulation TorchIO (recomposition dans l'espace de Fourier) change le bruit de fond (FBER médian 365 -> 26 000 dès le niveau 1), pas parce que le mouvement s'aggrave. Les indicateurs choisis sont donc sensibles au bruit et au biais, peu au mouvement simulé ; un indicateur dédié au mouvement manque. 4.4 : rapport local de 28 sujets avec vignettes.
**Alternatives écartées.** MRIQC lui-même (conteneur, pas de Docker en local, J-005 ; et ses tissus FSL sont moins adaptés aux FLAIR 2D que SynthSeg) ; z global (détecterait les scanners) ; moyenne et écart type (un très mauvais cas masque son propre écart) ; T1 rééchantillonnée sur la grille FLAIR (SNR gonflé).

---

## J-052 — Second entraînement (augmentation forte DA5) : pas de gain, modèle principal conservé
2026-10-05 · Actée

**Déroulement.** `nnUNetTrainerWMH_DA5_250`, mêmes réglages que J-037 sauf l'augmentation forte DA5 ; 250 epochs, 143 s par epoch (149 pour le premier), plateau du pseudo-Dice dès ~100 epochs (moyenne sur 10 epochs : 0,840 / 0,850 / 0,849 / 0,851 / 0,851 aux epochs 50 à 250). Modèle extrait de `kaggle/results.zip` (exclu de git) vers `data/nnunet/nnUNet_results/` ; inférence locale des 110 cas de test en 18,5 min, pic de mémoire GPU 2,0 Go, 0 échec.
**Résultats.** Validation (12 cas) : 0,813 contre 0,815 (DA5 meilleur sur 5 cas sur 12). Test (évaluation officielle) : Dice 0,798 contre 0,803 ; HD95 6,26 contre 6,12 mm ; AVD 16,3 contre 15,4 % ; rappel lésionnel 0,727 contre 0,728 ; F1 0,786 contre 0,783. Comparaison appariée (6.2) : différence de Dice +0,005 en faveur du premier modèle [0,001 ; 0,009], Wilcoxon p = 0,12, p Holm = 0,59 ; aucune métrique ne diffère significativement après correction. DA5 meilleur sur 48 sujets sur 110 ; la petite perte est concentrée à Utrecht (-0,025), **aucun gain sur les scanners inconnus** (Amsterdam GE 1,5T +0,002, Philips 3T -0,001 ; Dice inconnus 0,790 pour les deux modèles).
**Volumes (7.4).** DA5 sous-estime davantage le volume : rapport moyen 0,90 [limites 0,54 à 1,49] contre 0,98 [0,62 à 1,55], ICC 0,976 contre 0,982 ; à Utrecht 0,77 [0,70-0,85] (p Holm 0,001) contre 0,87. Hypothèse, non vérifiée visuellement : l'augmentation forte rend le modèle plus conservateur sur les bords des lésions, ce qui change peu le Dice mais biaise le biomarqueur.
**Décision.** Le premier modèle (ResEnc M, augmentation par défaut) reste le modèle principal M1 : rien ne justifie de changer, et c'est le plus simple. DA5 est conservé comme résultat négatif documenté : sur ces données, une augmentation plus forte n'améliore pas la généralisation aux scanners inconnus, qui ne montraient déjà pas de dégradation démontrée (J-047).

---

## J-053 — WMH-SynthSeg (M2) sur les 170 sujets : exécution, évaluation, ICV et ventricules
2026-10-05 · Actée

**Exécution.** Notebook Kaggle 05 v2, 3 parties (57, 57, 56 sujets) : **170 sujets sur 170**, aucun arrêt par le watchdog ; 350 s par sujet en médiane (507 au plus), pic de RAM 26,1 Go en médiane, **29,9 Go au plus** (scanners inconnus d'Amsterdam, sous le plafond de 32 Go) ; sessions de 5,2 à 7,0 h. Archives extraites à la racine (680 fichiers), rapports dans `results/tables/wmhsynthseg/`. **Déterminisme** : les 5 sujets passés deux fois (CHECK puis leur partie) donnent des segmentations et ventricules identiques au voxel près.
**Évaluation officielle (110 cas de test).** Dice **0,402** [0,363-0,444], HD95 14,0 mm, AVD 293 %, rappel lésionnel 0,43, F1 0,45 ; scanners connus 0,414, inconnus 0,350. Inférieur au seuillage (0,595) et à nnU-Net (0,802), différences significatives (p Holm < 1e-14). **Cause : sur-segmentation** : volume prédit 2,05 fois la référence en médiane (90 % des sujets surestimés ; rapport moyen 2,44, limites 0,41 à 14,5, ICC 0,37). Vérifié que ce n'est pas un artefact de notre chaîne : volume identique sur la grille 1 mm de l'outil et après notre rééchantillonnage (sub-001 : 11,4 contre 10,8 ml, référence 4,1 ml), label 77 correct ; sur sub-001, 54 % des voxels prédits sont à plus de 3 mm d'une lésion manuelle. Déjà visible au test de faisabilité (J-007). Hypothèse non vérifiée : sa définition des HSB, apprise sur d'autres annotations, est plus large que le protocole STRIVE de O1. Fait notable : son biais de volume **ne dépend pas du site** (Kruskal-Wallis p = 0,10), contrairement aux autres sources : un modèle généraliste robuste au contraste mais mal calibré sur ce protocole d'annotation.
**Usages anatomiques (7.1, 7.2).** ICV = voxels étiquetés de la segmentation 1 mm : médiane 1 450 ml [1 110-1 871], aucun manquant ; masque HD-BET / ICV = 0,94 en médiane. Ventricules latéraux pour la découpe de 7.2 sur les 170 sujets.
**Découpe périventriculaire / profond (7.2, test, M1 contre O1).** Périventriculaire : rapport 1,00 [0,64-1,56], ICC 0,982. **Profond : 0,87 [0,37-2,04], ICC 0,964** : les lésions profondes, petites et dispersées, sont sous-estimées et moins bien mesurées. Les experts aussi sous-estiment le profond par rapport à O1 (O3 0,86, O4 0,78) : O1 délimite davantage de lésions profondes. Part périventriculaire médiane (O1) : 0,80 (Utrecht) à 0,87 (Amsterdam GE).

---

## J-054 — Carte des lésions dans l'espace MNI (7.5) : SyN CC sur le modèle à 2 mm, 170 sujets
2026-10-05 · Actée (calcul de nuit lancé par l'utilisateur)

**Contexte.** Seul SyN CC suit les grands ventricules des sujets atrophiés (notebook 06, J-045), mais à 1 mm il coûte 50 min par sujet avec 12 threads : 10 sujets seulement en une nuit, trop peu pour une carte par site. Une carte de fréquence n'a pas besoin de 1 mm (elle est lissée, et les FLAIR ont des coupes de 3 mm).
**Test (sub-021, cas atrophié de référence).** CC sur le modèle MNI152NLin2009cAsym **à 2 mm** (`res-02`, rayon CC de 2 voxels soit ~4 mm, mêmes itérations 100/70/50/20) : **363 s au lieu de 3 011 (x8)**, RAM 0,95 Go au lieu de 2,96 ; ventricules du modèle tombant dans le liquide du sujet 0,94 (0,88 à 1 mm), volume ventriculaire estimé par la déformation 86 ml (95 à 1 mm, 26 avec le réglage par défaut) ; contrôle visuel (notebook 06, section 6) : ventricules ramenés dans le contour du modèle presque aussi bien qu'à 1 mm. Le jacobien d'une déformation à 2 mm est rééchantillonné sur la grille de 1 mm pour l'affichage.
**Décision.** Les 170 sujets en CC à 2 mm, en local pendant une nuit (`scripts/syn_benchmark.py --subjects all --variants cc2mm_100-70-50-20`, scanners alternés pour qu'un arrêt anticipé couvre tous les sites). La carte de fréquence par site et par source (O1, M1, M2, M3) sera calculée à 2 mm.
**Alternatives écartées.** CC à 1 mm sur 10 sujets (trop peu par site) ; Kaggle CPU (4 cœurs : estimé à 1,5-2,5 h par sujet à 1 mm, non mesuré).

---

## J-056 — Cartes de fréquence lésionnelle dans l'espace MNI (7.5)
2026-10-06 · Actée

**Mise en oeuvre.** `biomarkers/mni.py` : masque de lésions de chaque source (label 2 de O1 retiré) transporté en deux rééchantillonnages linéaires successifs, FLAIR -> T1 brute (rigide inverse de 3.2, convention vérifiée en 4.1) puis T1 -> MNI152NLin2009cAsym 2 mm ([déformation, affine] de la nuit, J-054 ; convention vérifiée dans le notebook 06). Interpolation linéaire : la fraction de voxel couverte est conservée. Carte de fréquence = moyenne des fractions par source et par site ; accord d'une méthode avec O1 = corrélation de Pearson voxel à voxel des cartes du test, dans le cerveau du modèle. Sources : O1 sur les 170 sujets, M1, M2 et M3 sur les 110 sujets de test. 170 sujets en ~3 min (3 processus). Tests : transport identité (même lésion, même place), corrélation des cartes.
**Contrôle.** Volume dans l'espace MNI / volume natif : médiane 1,18 (O1), cohérent avec le rapport de taille modèle / cerveaux (masque du modèle 1 887 ml contre 1 357 ml en médiane pour HD-BET). Cartes O1 anatomiquement attendues : bonnets frontaux et occipitaux, bandes le long des corps ventriculaires, symétrie, rien hors de la substance blanche ; même motif sur les 5 sites.
**Résultats (test).** Corrélation avec la carte O1 : **M1 0,995** (0,97 à 0,99 par site), M3 0,90 (0,85 à 0,89), M2 0,88 (**0,67** à Amsterdam GE 1,5T à 0,88). Cartes de différence : M1 sans écart notable ; **M2 : excès concentré dans une bande fine qui longe la paroi des ventricules** (sa sur-segmentation n'est pas diffuse : elle étiquette le liseré hyperintense périventriculaire, que O1 n'annote pas ; fréquence moyenne 0,015 contre 0,009) ; M3 : faux positifs sur la ligne médiane entre les ventricules et lésions profondes manquées. Figures `results/figures/mni_frequency_*.png` (cartes agrégées sur le modèle, aucune image de patient) ; cartes NIfTI dans `results/maps/` (exclu de git).

---

## J-055 — Normalisation d'intensité (8.1) et son effet sur le seuillage, par site (8.4)
2026-10-06 · Actée

**Rappel J-028** : règle A appliquée (tous les paramètres calculés dans le masque HD-BET) ; règle B sans objet (aucun calcul sur le fond).
**Décision.** `harmonize/intensity.py` : quatre normalisations de la FLAIR N4 : `none` (I / médiane des cerveaux d'entraînement, une seule constante) ; `zscore` (médiane, MAD du cerveau : le M3 de 5.5) ; `whitestripe` (moyenne et écart type de la substance blanche saine, tissus de WMH-SynthSeg ; variante de WhiteStripe, Shinohara 2014, qui trouve le pic de la SB dans l'histogramme) ; `histmatch` (Nyúl et Udupa 2000 : centiles 1, 10, ..., 90, 99 du cerveau alignés par morceaux linéaires sur une échelle standard apprise sur l'entraînement). `seg/threshold.py --normalization` : même recherche sur grille (J-044) sur l'entraînement, grille de seuils propre à chaque échelle (`t_grid_by_normalization`, choisie d'après la distribution des lésions d'entraînement), application et évaluation officielle sur le test (`desc-threshold<méthode>`). Grille de profondeur et de taille élargie (jusqu'à 22 mm et 320 mm³) après des optimums au bord ; l'optimum de 5.5 (14 mm, 40 mm³) n'est pas concerné. 6 tests (gain et décalage de scanner annulés par zscore et WhiteStripe, SB à moyenne nulle, règle A, appariement des centiles, constante de `none`, erreurs).
**Résultats (test, Dice officiel).**
| Normalisation | Dice | Connus / inconnus | Écart entre sites | Kruskal-Wallis (site) | Seuil optimal par site (Ams. GE 3T / Sing. / Utr.) |
|---|---|---|---|---|---|
| zscore (M3) | **0,595** | 0,589 / 0,622 | 0,07 | p = 0,90 | 2,4 / 1,7 / 2,6 |
| whitestripe | 0,564 | 0,570 / 0,541 | 0,08 | p = 0,94 | 4,5 / 2,75 / 3,75 |
| histmatch | 0,522 | 0,531 / 0,479 | 0,11 | p = 0,52 | 104 / 90 / 88 |
| none | **0,203** | 0,209 / 0,173 | 0,50 | p < 0,001 | 2,2 / 0,8 / 1,4 |
**Lecture.** Sans normalisation propre à chaque image, le seuillage **s'effondre** : Dice 0,00 à Singapour et à Amsterdam Philips, 0,13 à Amsterdam GE 3T (un seuil appris sur des échelles d'intensité différentes ne vaut rien ailleurs, y compris sur un site vu à l'entraînement). Les trois normalisations par image rendent le Dice **indépendant du site** (Kruskal-Wallis non significatif). zscore est la meilleure (écarts appariés : +0,031 contre WhiteStripe, p Holm 0,15 ; +0,073 contre histmatch, p Holm < 1e-7) : la plus simple suffit. L'appariement d'histogrammes aligne le mieux les seuils optimaux entre sites mais donne le moins bon Dice et sur-segmente (AVD 173 %) : en forçant tous les cerveaux sur la même distribution, il déforme le contraste des lésions, dont la part varie d'un sujet à l'autre (de 0,8 à 75 ml). Limite : pour histmatch, l'optimum de taille reste au bord (320 mm³), sans changement du Dice de test entre 160 et 320 mm³ (0,522).

---

## J-057 — Effet du prétraitement : nnU-Net sur les images `pre/` du challenge
2026-10-06 · Actée

**Question (J-010).** Le modèle principal (entraîné sur notre prétraitement : N4 avec le masque HD-BET, recalage rigide ANTs) dépend-il de notre chaîne ? On l'applique tel quel aux images `pre/` des organisateurs (correction de biais SPM12, T1 recalée avec elastix), sur les 110 cas de test. `seg/nnunet_infer.py --inputs challenge` (prédictions `desc-resencmpre`), même modèle et mêmes réglages ; le module d'inférence accepte désormais des cas quelconques (`predict_cases`), test de nommage ajouté.
**Résultat.** Dice **0,803 contre 0,803** (différence appariée +0,0007, Wilcoxon p = 0,91) ; HD95 6,45 contre 6,12 mm (p = 0,07) ; AVD, rappel et F1 sans différence (p > 0,7) ; aucune différence par scanner (au plus +0,003). 11 sujets sur 110 changent de plus de 0,02 de Dice, dans les deux sens. **Le modèle ne dépend pas de notre chaîne de prétraitement**, y compris sur les scanners inconnus : l'intérêt de notre chaîne est la maîtrise et la traçabilité (prétraitement depuis les données brutes), pas un gain de performance.
**Constat secondaire.** Les deux plus forts gains avec `pre/` sont des charges très élevées (sub-135 : 0,70 -> 0,88, 51 ml ; sub-009 : 0,65 -> 0,75, 195 ml), sous-segmentées à partir de nos images. Recalage hors de cause (sub-135 : 0,3 mm d'écart au plus avec elastix). Le contraste des lésions (médiane lésions / substance blanche saine) après notre N4, rapporté à celui après SPM12, **diminue avec la charge lésionnelle** (Spearman -0,57, 110 sujets ; médiane 0,95 au-delà de 30 ml contre 1,01 sous 5 ml ; sub-009 0,89). Interprétation : N4 absorbe en partie les grandes plaques confluentes dans le champ de biais lisse qu'il estime. Amélioration possible, non faite : exclure les lésions du masque de N4 (estimation itérative) ou pondérer N4 par la substance blanche. À écrire dans le README (limite).

---

## J-058 — Robustesse de nnU-Net aux artefacts (extension de 4.3) ; incident de disque plein
2026-10-06 · Actée

**Question.** 4.3 mesurait si le contrôle qualité **détecte** une image dégradée ; la question utile est **à partir de quel niveau d'artefact la segmentation se dégrade**. Un artefact non détecté mais sans effet est sans gravité ; un artefact qui dégrade la segmentation avant d'être détecté est un vrai risque.
**Mise en oeuvre.** `qc/robustness.py` : 10 sujets **de test** classés « utilisable » (2 par scanner, jamais vus par le modèle), FLAIR N4 (l'entrée du modèle) dégradée par les mêmes artefacts TorchIO que 4.3 (graine fixe), 4 niveaux par artefact, T1 intacte ; inférence nnU-Net (même modèle, mêmes réglages, `predict_cases`), métriques officielles contre O1. Niveau 0 = prédiction existante sur l'image intacte. 160 inférences, 5 lots, 2,0 à 3,5 Go de GPU. Test : choix des sujets (utilisables et de test seulement).
**Résultats (moyennes sur 10 sujets ; image intacte : Dice 0,787, rappel lésionnel 0,72).**
| Niveau | Bruit | Champ de biais | Images fantômes | **Mouvement** |
|---|---|---|---|---|
| 1 | 0,781 | 0,780 | 0,784 | **0,736** (rappel 0,53) |
| 2 | 0,766 | 0,761 | 0,770 | 0,699 (0,44) |
| 3 | 0,741 | 0,680 | 0,729 | 0,595 (0,34) |
| 4 | 0,699 | 0,442 | 0,637 | 0,591 (0,31) |
**Lecture, croisée avec la détection de 4.3** (comparaison qualitative : 4.3 dégradait les FLAIR brutes de 10 autres sujets). Champ de biais : la segmentation ne décroche qu'aux niveaux 3-4 (0,68 puis 0,44), alors que le QC le détecte dès le niveau 2 (60 %) puis à 100 % : **détecté avant de nuire**. Bruit : dégradation progressive et modérée (0,70 au niveau 4), détection partielle (30 à 70 %). Images fantômes : la segmentation baisse au niveau 3-4, le QC ne détecte qu'au niveau 4 (60 %). **Mouvement : c'est le seul artefact qui nuit dès le plus faible niveau** (1° et 1 mm : Dice -0,05, rappel lésionnel 0,72 -> 0,53, les petites lésions floutées disparaissent), **et le QC ne le détecte pas** (4.3) : c'est le risque principal, et la limite à écrire dans le README ; un indicateur dédié au mouvement manque au QC.
**Incident (2026-10-06).** Pendant le premier passage, le disque C: s'est rempli (0 Go libre) : 149 images dégradées écrites sur 160, aucune prédiction. Cause principale : `ants.registration` écrit chaque transformation dans le dossier temporaire du système et ne la supprime jamais (807 fichiers, 5,4 Go, des recalages de 3.2, 3.6 et 7.5). Actions : fichiers temporaires d'ANTs du projet supprimés (vérifiés par leurs dates, du 3 au 5 octobre), `uv cache prune` ; 11,6 Go libres. **Correction** : `preproc.register.remove_registration_files`, appelée après chaque recalage (3.2, 3.6, script SyN), supprime les fichiers d'ANTs du dossier temporaire après leur copie (test : seuls les fichiers du dossier temporaire sont supprimés). Dossier incomplet supprimé et étude relancée en entier.

---

## J-059 — Signature de site des caractéristiques d'image (8.2) et harmonisation ComBat (8.3)
2026-10-06 · Actée

**Rappel J-028** : règle A appliquée (statistiques dans la substance blanche saine à l'intérieur du masque HD-BET) ; règle B sans objet (aucune caractéristique de fond).
**8.2.** `harmonize/features.py` : par sujet, 10 caractéristiques en FLAIR et 10 en T1 (N4, T1 sur la grille FLAIR) : centiles 5 à 95, écart interquartile, asymétrie, aplatissement de la substance blanche saine (labels 2/41 de WMH-SynthSeg, qui donne aux HSB leur propre label), médiane du cortex et contraste cortex - SB ; deux échelles : `none` (une constante par modalité, médiane des cerveaux d'entraînement) et `zscore` (par image). `harmonize/combat.py` : classifieur de site (standardisation + régression logistique multinomiale), validation croisée en 5 plis stratifiés par scanner, précision équilibrée (hasard : 0,20).
**8.3.** ComBat (neuroHarmonize, Bayes empirique ; J-022) avec covariables biologiques ln(1 + volume HSB de O1) et ICV. **Appris sur les plis d'entraînement et appliqué au pli de test** : le classifieur n'est jamais évalué sur des données harmonisées avec ses propres sujets de test. Écart à la ROADMAP (« ajusté sur le train ») : les 2 scanners inconnus n'ont aucun sujet d'entraînement et ComBat ne peut pas harmoniser un site qu'il n'a jamais vu ; les plis de validation croisée contiennent tous les sites. Préservation de la biologie : corrélation de Spearman de chaque caractéristique avec la charge lésionnelle, et part de variance expliquée par le scanner (eta²), avant et après ComBat (ajusté sur tous les sujets). 4 tests sur données simulées (effet de site connu, signal biologique conservé, eta², caractéristiques).
**Résultats (170 sujets, 5 scanners).**
| Échelle | Site deviné (avant ComBat) | Après ComBat | Part de variance due au site (avant -> après) | Lien avec la charge lésionnelle, \|rho\| moyen (avant -> après) |
|---|---|---|---|---|
| none | **0,90** | 0,29 | 0,71 -> 0,02 | 0,24 -> 0,28 |
| zscore | **0,73** | 0,28 | 0,21 -> 0,02 | 0,40 -> 0,44 |
**Lecture.** Les intensités de la substance blanche saine portent une forte signature du scanner : 90 % des sujets sont attribués au bon scanner sans normalisation, et encore **73 % après la normalisation par image** (z robuste), qui ne retire qu'une partie de l'effet de site (eta² 0,71 -> 0,21) : forme de l'histogramme et contraste restent propres au site. ComBat ramène le classifieur près du hasard (0,28 pour 0,20) et la part de variance due au site à 2 %, **sans perdre la biologie** : le lien avec la charge lésionnelle se renforce légèrement (0,40 -> 0,44), le bruit de site masquant une partie du signal. Limite : la covariable de charge vient de O1 (manuel) ; dans une étude réelle, elle viendrait d'une segmentation automatique.

---

## J-060 — Figures de synthèse et position au classement du challenge (6.4)
2026-10-06 · Actée

**Classement.** Source : tableau du classement (57 équipes, état de décembre 2022) page 20 du `readme.pdf` officiel du jeu de données, extrait dans `results/tables/leaderboard_wmh2017.csv` (le site du challenge, wmh.isi.uu.nl, ne répondait plus le 2026-10-06). Formule officielle (page 23) dans `evaluate/leaderboard.py` : pour chaque métrique, meilleure équipe 0, pire 1, linéaire entre les deux ; moyenne des 5 rangs. **Contrôle** : appliquée aux 57 équipes seules, la formule retrouve les scores publiés à 0,005 près (valeurs sources arrondies à 2 décimales) ; tests sur l'exemple chiffré du readme. Chaque méthode est insérée **seule** parmi les 57 équipes (ajouter une équipe peut changer les bornes d'une métrique).
**Positions (sur 58).** M1 nnU-Net **9e** (Dice 2e, HD95 6e, AVD 1er, rappel 22e, F1 2e : J-042 confirmé) ; M1 + DA5 10e ; M1 sur les images `pre/` 10e ; M3 seuillage 55e ; M2 WMH-SynthSeg 55e.
**Figures** (`viz/figures.py`, chiffres agrégés seulement) : `dice_by_method_and_scanner.png` (IC bootstrap de 6.2), `seen_vs_unseen.png`, `leaderboard.png`, `human_ceiling.png` ; tableau `leaderboard_positions.csv`.

---

## J-061 — Workflow Snakemake (9.1) et intégration continue (9.3)
2026-10-06 · Actée

**9.1.** `workflow/Snakefile` **au niveau des étapes** : chaque règle appelle un module du paquet sur tous les sujets (les modules sautent le travail déjà fait), de `download` à `figures`, en passant par le QC, les biomarqueurs, les cartes MNI et l'harmonisation (24 tâches pour la cible `all`). Les deux étapes faites sur Kaggle (entraînements nnU-Net, WMH-SynthSeg) sont des **règles externes** : elles échouent avec les instructions (notebook à lancer) si leurs fichiers manquent, et sont ignorées s'ils sont présents. `ancient()` sur les entrées des étapes coûteuses ou externes : seule leur existence compte, pour qu'une réécriture de rapport ne déclenche jamais un nouvel entraînement Kaggle ni 8 h de recalage. Le projet ayant été calculé étape par étape, `snakemake --touch` a enregistré l'état existant : `snakemake -n` répond « Nothing to be done ». Alternative écartée : règles par sujet (170 x ~20 fichiers par étape) : DAG lourd, et les modules gèrent déjà le parallélisme et la reprise sous watchdog. Le linter de Snakemake recommande une directive `log:` et un environnement par règle : non fait (environnement unique verrouillé par uv).
**9.3.** `.github/workflows/ci.yml` : à chaque push, environnement verrouillé (`uv sync --frozen --extra local`), `ruff check` et `ruff format --check`, linter Snakemake (indicatif), pytest avec couverture. Vérifié localement avant le premier passage : verrou à jour, 66 fichiers formatés, lint sans erreur, **155 tests** passent. La construction de l'image Docker sera ajoutée avec 9.4.
---

## Décisions en attente

Aucune pour l'instant.
