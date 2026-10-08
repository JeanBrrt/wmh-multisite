# wmh-multisite

[![ci](https://github.com/JeanBrrt/wmh-multisite/actions/workflows/ci.yml/badge.svg)](https://github.com/JeanBrrt/wmh-multisite/actions/workflows/ci.yml)

**Chaîne reproductible d'analyse multicentrique des hyperintensités de la substance blanche (HSB) en IRM** : des images brutes aux biomarqueurs, avec contrôle qualité, segmentation, évaluation statistique et harmonisation inter-sites.

170 IRM FLAIR + T1 issues de **5 scanners** (3 constructeurs, 1,5 T et 3 T, 3 centres), jeu de données public du challenge MICCAI WMH 2017.

> **Question.** Quand les images viennent de centres, d'appareils et de protocoles différents, peut-on
> (1) repérer les images problématiques, (2) segmenter les lésions de façon fiable, y compris sur un scanner jamais vu,
> (3) en tirer des biomarqueurs comparables d'un site à l'autre, et (4) retirer l'empreinte du scanner sans effacer la biologie ?

Chaque choix est justifié, avec ses alternatives et ses preuves, dans le journal de décisions [`justification.md`](justification.md) (J-001 à J-064). L'avancement est suivi dans [`ROADMAP.md`](ROADMAP.md).

---

## Résultats en bref

| | Résultat |
|---|---|
| **Segmentation** | nnU-Net (ResEnc M) : **Dice 0,80** sur 110 cas de test, **9e sur 58** au classement officiel (2e au Dice, 1er sur l'erreur de volume) |
| **Plafond humain** | Deux experts indépendants s'accordent à Dice 0,76 ; nnU-Net atteint 0,815 contre la référence sur les mêmes cas |
| **Scanners inconnus** | Aucune dégradation démontrée (Dice −0,015, IC 95 % [−0,066 ; +0,030], 20 cas de 2 scanners jamais vus) |
| **Biomarqueur** | Volume de HSB : ICC 0,982 avec la référence, **au niveau de deux experts** (0,981 et 0,977) ; un biais de site subsiste, experts compris |
| **Harmonisation** | Le scanner est deviné à 90 % à partir de la substance blanche saine, 73 % après normalisation, **28 % après ComBat** (hasard 20 %), sans perte du signal biologique |
| **Contrôle qualité** | Champ de biais détecté avant qu'il ne nuise à la segmentation ; **le mouvement dégrade la segmentation dès le plus faible niveau sans être détecté en FLAIR** (limite principale) |

---

## Données

- **MICCAI WMH Segmentation Challenge 2017** (Kuijf et al., *IEEE TMI* 2019), [doi:10.34894/AECRSD](https://doi.org/10.34894/AECRSD), licence **CC BY-NC 4.0**. Les données ne sont **pas** versionnées : elles sont téléchargées par l'API DataverseNL, empreintes SHA-1 vérifiées.
- 60 sujets d'entraînement (3 scanners) et 110 de test, dont **20 issus de 2 scanners absents de l'entraînement** : un test de généralisation hors domaine prévu par le challenge.
- Référence : délimitation manuelle d'un expert (O1, relue par un second) ; deux autres experts (O3, O4) ont délimité les 60 cas d'entraînement, ce qui donne le plafond humain.

| Scanner | Entraînement | Test | FLAIR |
|---|---|---|---|
| Utrecht, Philips 3T | 20 | 30 | 2D, coupes de 3 mm |
| Singapour, Siemens 3T | 20 | 30 | 2D, coupes de 3 mm |
| Amsterdam, GE 3T | 20 | 30 | 3D |
| Amsterdam, GE 1,5T | 0 | 10 | 3D, **scanner inconnu** |
| Amsterdam, Philips 3T | 0 | 10 | 3D, **scanner inconnu** |

---

## La chaîne de traitement

```
DataverseNL ─> BIDS ─> recalage rigide T1→FLAIR ─> masque du cerveau (HD-BET) ─> correction de biais N4
                                                                                        │
         ┌──────────────────────────────────────────────────────────────────────────────┤
         v                                    v                                          v
  contrôle qualité               segmentation : M1 nnU-Net,                   normalisation d'intensité
  (indicateurs MRIQC,            M2 WMH-SynthSeg, M3 seuillage                (z-score, WhiteStripe,
   atypiques par site)                       │                                 appariement d'histogrammes)
                                             v                                          │
                              évaluation officielle + statistiques                      v
                                             │                               signature de site, ComBat
                                             v
                          biomarqueurs (volume / ICV, périventriculaire / profond,
                          nombre de lésions, cartes MNI), accord par site
```

### 1. Données et prétraitement (Phases 2-3)
Téléchargement vérifié et conversion en **BIDS 1.10** ; recalage rigide de la T1 sur la FLAIR (**ANTs**, information mutuelle, contrôlé contre le recalage elastix des organisateurs) ; masque du cerveau **HD-BET** (GPU local) ; correction de biais **N4** estimée dans le masque du cerveau, identique sur tous les sites. Recalage non linéaire **SyN** vers le modèle MNI152 pour les cartes de groupe.

### 2. Contrôle qualité (Phase 4)
Indicateurs de qualité calculés **comme MRIQC** (SNR, CNR, CJV, EFC, FBER, champ de biais), adaptés à la FLAIR que MRIQC ne gère pas ; z robustes **par scanner** et Isolation Forest, classement **utilisable / à vérifier / à exclure** ; validation sur des **artefacts simulés** (TorchIO) ; rapport HTML local ; robustesse de nnU-Net aux mêmes artefacts.

### 3. Segmentation (Phase 5)
| | Méthode | Rôle |
|---|---|---|
| **M1** | **nnU-Net v2, ResEnc M**, 3d_fullres, FLAIR + T1, entraîné sur Kaggle (2 × T4), plis stratifiés par scanner et par charge lésionnelle | Méthode principale |
| M2 | **WMH-SynthSeg** (FreeSurfer), modèle publié, utilisé sans modification | Modèle généraliste prêt à l'emploi |
| M3 | Seuillage de la FLAIR : z-score robuste dans le cerveau, profondeur et taille minimales, réglé par recherche sur grille sur l'entraînement | Référence classique |

### 4. Évaluation (Phase 6)
Script **officiel** du challenge (copié sans modification) : Dice, HD95, AVD, rappel et F1 lésionnels. IC à 95 % par **bootstrap stratifié par scanner**, comparaisons appariées (**Wilcoxon**, correction de **Holm** sur 140 tests), scanners connus contre inconnus (**Mann-Whitney**), plafond inter-observateurs, position au classement par la formule officielle.

### 5. Biomarqueurs (Phase 7)
Volume de HSB (ml et % du volume intracrânien), découpe **périventriculaire / profonde** (règle des 10 mm autour des ventricules), nombre et taille des lésions, **cartes de fréquence lésionnelle dans l'espace MNI** par site. Accord avec la référence : **Bland-Altman** sur l'échelle logarithmique, **ICC** d'accord absolu, biais par site.

### 6. Harmonisation (Phase 8)
Quatre normalisations d'intensité, testées par leur effet sur le seuillage, site par site ; 20 caractéristiques de la substance blanche saine ; **classifieur de site** en validation croisée ; **ComBat** (neuroHarmonize) appris dans les plis d'entraînement, avec la charge lésionnelle et l'ICV comme covariables protégées.

### 7. Reproductibilité (Phase 9)
**Snakemake** (chaîne complète, étapes Kaggle décrites comme externes), environnement verrouillé (**uv**), **155 tests** pytest sur données synthétiques, **GitHub Actions** (lint, tests, construction de l'image Docker), **Docker** et **Apptainer**, script **SLURM** pour Jean Zay (5 plis en parallèle, non testé), empreinte carbone (**CodeCarbon**).

---

## Résultats

### Segmentation (110 cas de test, moyenne et IC à 95 %)

| Méthode | Dice ↑ | HD95 (mm) ↓ | AVD (%) ↓ | Rappel lésionnel ↑ | F1 lésionnel ↑ | Classement |
|---|---|---|---|---|---|---|
| **M1 nnU-Net** | **0,802** [0,784-0,820] | **6,1** [4,8-7,7] | **15,4** [13,1-18,0] | **0,728** [0,709-0,746] | 0,783 [0,768-0,798] | **9e / 58** |
| M1 + augmentation forte (DA5) | 0,798 [0,779-0,815] | 6,3 | 16,3 | 0,727 | 0,786 | 10e / 58 |
| M1 sur le prétraitement du challenge | 0,803 [0,783-0,821] | 6,4 | 16,1 | 0,724 | 0,780 | 10e / 58 |
| M3 seuillage | 0,595 [0,558-0,628] | 22,7 | 72,7 | 0,229 | 0,284 | 55e / 58 |
| M2 WMH-SynthSeg | 0,402 [0,361-0,442] | 14,0 | 293 | 0,428 | 0,451 | 55e / 58 |

- M1 surpasse M3 et M2 sur les 5 métriques (p Holm < 1e-11). Ni l'augmentation forte ni le prétraitement des organisateurs ne changent le résultat (p Holm = 1) : **le modèle ne dépend pas de notre chaîne de prétraitement**.
- Point faible de M1 : les **petites lésions** (rappel 22e sur 58). WMH-SynthSeg **sur-segmente** (volume médian 2 fois la référence), surtout le long des ventricules.

![Dice par méthode et par scanner](results/figures/dice_by_method_and_scanner.png)

### Scanners connus contre inconnus (M1)

| | Connus (90) | Inconnus (20) | Inconnus − connus [IC 95 %] | p (Mann-Whitney) |
|---|---|---|---|---|
| Dice | 0,805 | 0,790 | −0,015 [−0,066 ; +0,030] | 0,45 |
| HD95 (mm) | 5,6 | 8,5 | +2,9 [−2,2 ; +8,7] | 0,96 |
| Rappel lésionnel | 0,723 | 0,754 | +0,031 [−0,017 ; +0,077] | 0,17 |

### Plafond humain (12 cas de validation, non vus par M1)

| Contre la référence O1 | Dice | HD95 (mm) | Rappel | F1 |
|---|---|---|---|---|
| **M1 nnU-Net** | **0,815** | **4,3** | 0,686 | 0,762 |
| Expert O3 | 0,757 | 5,7 | 0,641 | 0,721 |
| Expert O4 | 0,781 | 4,9 | 0,686 | 0,760 |
| *O4 contre O3 (deux experts indépendants)* | *0,740* | *8,0* | | *0,763* |

M1 est au niveau des experts. Il a appris le style de O1, contre lequel il est jugé : ce n'est pas « mieux qu'un expert ».

### Biomarqueurs : volume de HSB contre la référence

| Source | Rapport moyen [limites d'accord 95 %] | ICC [IC 95 %] |
|---|---|---|
| **M1 nnU-Net** (test, 110) | **0,98** [0,62-1,55] | **0,982** [0,975-0,988] |
| Expert O3 (entraînement, 60) | 0,97 [0,60-1,57] | 0,981 |
| Expert O4 (entraînement, 60) | 0,91 [0,55-1,51] | 0,977 |
| M3 seuillage | 1,18 [0,33-4,14] | 0,814 |
| M2 WMH-SynthSeg | 2,44 [0,41-14,5] | 0,372 |

- Lésions **profondes** moins bien mesurées que périventriculaires (M1 : ICC 0,964 contre 0,982).
- Le biais dépend du site pour toutes les sources, **experts compris** (Kruskal-Wallis p < 0,001) : une partie du « biais de site » reflète le style de la référence sur certains sites.
- Cartes MNI : corrélation avec la carte de la référence de 0,995 pour M1, 0,90 pour M3, 0,88 pour M2.

### Contrôle qualité

- 170 sujets : **136 utilisables, 30 à vérifier, 4 à exclure**, répartis sur tous les scanners ; sans lien avec la qualité de segmentation de M1 (les images du challenge ont été sélectionnées).
- Artefacts simulés (10 sujets, 4 niveaux), part détectée au niveau le plus fort :

| Artefact | Détecté en FLAIR | Détecté en T1 | Effet sur nnU-Net (Dice, niveau 1 → 4 ; image intacte 0,787) |
|---|---|---|---|
| Champ de biais | 100 % | 100 % | 0,780 → **0,442** (détecté **avant** de nuire) |
| Bruit | 70 % | 80 % | 0,781 → 0,699 |
| Images fantômes | 60 % | 100 % | 0,784 → 0,637 |
| **Mouvement** | **non concluant** | 70 % | **0,736** → 0,591 (rappel 0,72 → 0,53 dès le niveau 1) |

### Harmonisation

| Normalisation de la FLAIR | Dice du seuillage | Effet du site sur le Dice |
|---|---|---|
| Aucune | 0,203 | fort (p < 0,001) |
| **z-score robuste** | **0,595** | aucun (p = 0,90) |
| WhiteStripe | 0,564 | aucun (p = 0,94) |
| Appariement d'histogrammes | 0,522 | aucun (p = 0,52) |

| Caractéristiques de la substance blanche saine | Scanner deviné (hasard 20 %) | Variance due au site | Lien avec la charge lésionnelle (\|ρ\|) |
|---|---|---|---|
| Sans normalisation | 90 % | 71 % | 0,24 |
| z-score robuste | 73 % | 21 % | 0,40 |
| z-score + **ComBat** | **28 %** | **2 %** | **0,44** |

### Coût de calcul

Environ **5,5 kWh et 2,3 kg CO₂eq** au total, dont 98 % pour les calculs sur Kaggle (deux entraînements nnU-Net de ~10 h sur 2 × T4, WMH-SynthSeg sur CPU) ; mesuré avec CodeCarbon pour les étapes locales, estimé pour les autres ([`carbon_footprint.csv`](results/tables/carbon_footprint.csv)).

---

## Reproduire

```bash
uv sync --extra local                                    # environnement verrouillé (Python 3.12)
uv run python -m wmh_multisite.data.download             # ~4 Go depuis DataverseNL, SHA-1 vérifiés
uv run snakemake -s workflow/Snakefile --cores 4         # toute la chaîne locale
```

- Deux étapes tournent sur **Kaggle** (GPU / RAM insuffisants en local) : l'entraînement nnU-Net ([`kaggle/03_nnunet_train.ipynb`](kaggle/03_nnunet_train.ipynb)) et WMH-SynthSeg ([`kaggle/05_wmhsynthseg.ipynb`](kaggle/05_wmhsynthseg.ipynb)). Le workflow les décrit comme des étapes externes et indique quoi lancer si leurs sorties manquent.
- **Conteneurs** : `docker build -f containers/Dockerfile -t wmh-multisite .` (image CPU, construite à chaque push par l'intégration continue) ; [`containers/apptainer.def`](containers/apptainer.def) pour le calcul intensif.
- **Jean Zay** : [`hpc/slurm_jeanzay.sh`](hpc/slurm_jeanzay.sh) entraîne les 5 plis en parallèle (**non testé**, faute d'accès).
- Tests : `uv run pytest` (155 tests, données synthétiques, sans réseau).

---

## Outils

| Domaine | Outils |
|---|---|
| Imagerie | nibabel, SimpleITK, ANTsPy (N4, recalages rigide et SyN), HD-BET, TemplateFlow (MNI152NLin2009cAsym), BIDS |
| Segmentation | nnU-Net v2 (ResEnc M), WMH-SynthSeg (FreeSurfer), PyTorch |
| QC | indicateurs de type MRIQC, scikit-learn (Isolation Forest), TorchIO (artefacts simulés) |
| Statistiques | NumPy, SciPy, pandas (bootstrap, Wilcoxon, Mann-Whitney, Holm, ICC, Bland-Altman) |
| Harmonisation | neuroHarmonize / neuroCombat (ComBat), scikit-learn |
| Reproductibilité | uv, Snakemake, pytest, ruff, GitHub Actions, Docker, Apptainer, SLURM, CodeCarbon, nbstripout |
| Calcul | GPU local (RTX 4060 laptop, sous watchdog mémoire), Kaggle (2 × T4) |

## Organisation du dépôt

```
config/config.yaml          tous les paramètres (chemins, seuils, grilles, graines)
src/wmh_multisite/
  data/                     téléchargement DataverseNL, conversion BIDS
  preproc/                  recalage, masque du cerveau, N4, atlas
  qc/                       indicateurs, atypiques, artefacts simulés, robustesse
  seg/                      export et inférence nnU-Net, seuillage
  evaluate/                 script officiel, métriques, statistiques, plafond humain, classement
  biomarkers/               volumes, localisation, lésions, accord, cartes MNI
  harmonize/                normalisations, caractéristiques, ComBat
  viz/                      figures, rapport QC
workflow/Snakefile          chaîne complète
kaggle/                     notebooks des étapes sur GPU distant
notebooks/                  explorations et contrôles visuels (sorties retirées)
containers/, hpc/           Docker, Apptainer, SLURM
tests/                      155 tests
results/tables, figures     résultats agrégés (aucune image de patient)
justification.md            journal des décisions (J-001 à J-064)
```

## Limites

- **Un seul pli** d'entraînement nnU-Net, sans ensemble (budget GPU) ; le script SLURM prévoit les 5 plis.
- **20 cas** de scanners inconnus, tous d'Amsterdam : « aucune dégradation démontrée » ne veut pas dire « aucune dégradation ».
- Aucune donnée démographique : un écart entre sites peut venir de la population autant que du scanner ; le signal biologique protégé par ComBat est la charge lésionnelle elle-même.
- Le **mouvement** n'est pas détecté par le QC en FLAIR, alors qu'il dégrade la segmentation dès un faible niveau.
- N4 atténue un peu le contraste des lésions chez les sujets à très forte charge (pistes : exclure les lésions du masque de N4).
- Plafond humain mesuré sur 12 cas, de scanners vus.

## Références

- Kuijf H.J. et al. *Standardized Assessment of Automatic Segmentation of White Matter Hyperintensities and Results of the WMH Segmentation Challenge.* IEEE TMI, 2019.
- Isensee F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods, 2021 ; *nnU-Net Revisited.* MICCAI, 2024.
- Laso P. et al. *Quantifying white matter hyperintensity and brain volumes in heterogeneous clinical and low-field portable MRI* (WMH-SynthSeg). ISBI, 2024.
- Isensee F. et al. *Automated brain extraction of multisequence MRI using artificial neural networks* (HD-BET). Human Brain Mapping, 2019.
- Tustison N.J. et al. *N4ITK: improved N3 bias correction.* IEEE TMI, 2010.
- Esteban O. et al. *MRIQC.* PLoS ONE, 2017.
- Johnson W.E. et al. *Adjusting batch effects in microarray expression data using empirical Bayes methods* (ComBat). Biostatistics, 2007 ; Pomponio R. et al. *Harmonization of large MRI datasets* (neuroHarmonize). NeuroImage, 2020.

## Licence

Code sous licence MIT ([`LICENSE`](LICENSE)). Les données du challenge restent sous licence CC BY-NC 4.0 et ne sont pas redistribuées.
