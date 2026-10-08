# wmh-multisite

[![ci](https://github.com/JeanBrrt/wmh-multisite/actions/workflows/ci.yml/badge.svg)](https://github.com/JeanBrrt/wmh-multisite/actions/workflows/ci.yml)

**Chaîne reproductible d'analyse multicentrique des hyperintensités de la substance blanche (HSB) en IRM** : des images brutes aux biomarqueurs, avec contrôle qualité, segmentation, évaluation statistique et harmonisation inter-sites.

Les données proviennent du challenge MICCAI WMH 2017 (170 IRM FLAIR + T1 issues de **5 scanners** sur 3 sites différents).

Ce projet s'articule autour de quatre questions :

> **Question.** Quand les images viennent de centres, d'appareils et de protocoles différents, peut-on
> (1) repérer les images problématiques, (2) segmenter les lésions de façon fiable, y compris sur un scanner jamais vu,
> (3) en tirer des biomarqueurs comparables d'un site à l'autre, et (4) retirer l'empreinte du scanner sans effacer la biologie ?

Chaque choix est justifié, avec ses alternatives et ses preuves, dans le journal de décisions [`justification.md`](justification.md) (J-001 à J-065). L'avancement est suivi dans [`ROADMAP.md`](ROADMAP.md).

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

- **MICCAI WMH Segmentation Challenge 2017** (Kuijf et al., *IEEE TMI* 2019), [doi:10.34894/AECRSD](https://doi.org/10.34894/AECRSD), licence **CC BY-NC 4.0**. Les données ne sont **pas** versionnées et sont téléchargées par l'API DataverseNL (1 551 fichiers, 4,1 Go, empreintes SHA-1 vérifiées).
- 60 sujets d'entraînement (3 scanners) et 110 de test, dont **20 issus de 2 scanners absents de l'entraînement** (généralisation hors domaine prévue par le challenge).
- Labellisation manuelle d'un expert (O1, relue par un second, O2) ainsi que de deux autres experts (O3, O4) sur les 60 cas d'entraînement, ce qui permet de calculer un plafond humain.

| Centre | Scanner | Champ | Entraînement | Test | FLAIR acquise | T1 acquise |
|---|---|---|---|---|---|---|
| UMC Utrecht | Philips Achieva | 3 T | 20 | 30 | 2D axiale, 0,96 × 0,95 × 3 mm | 3D sagittale, 1 × 1 × 1 mm |
| NUHS Singapour | Siemens TrioTim | 3 T | 20 | 30 | 2D axiale, 1 × 1 × 3 mm | 3D sagittale, 1 × 1 × 1 mm |
| VU Amsterdam | GE Signa HDxt | 3 T | 20 | 30 | 3D sagittale, 0,98 × 0,98 × 1,2 mm | 3D sagittale, 0,94 × 0,94 × 1 mm |
| VU Amsterdam | Philips Ingenuity | 3 T | 0 | 10 | 3D sagittale, 1,04 × 1,04 × 0,56 mm | 3D sagittale, 0,87 × 0,87 × 1 mm |
| VU Amsterdam | GE Signa HDxt | 1,5 T | 0 | 10 | 3D sagittale, 1,21 × 1,21 × 1,3 mm | 3D sagittale, 0,98 × 0,98 × 1,5 mm |
| **Total** | | | **60** | **110** | | |

Les organisateurs fournissent toutes les FLAIR **reconstruites en coupes axiales de 3 mm** (grilles de 128 × 256 × 103 à 321 × 240 × 83 voxels selon le scanner), et défacées pour Amsterdam seulement. La FLAIR est l'image de référence : la vérité terrain et l'évaluation sont dans sa grille.

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

### 1. Données et prétraitement

Les données sont téléchargées via l'API Dataverse et converties au format **BIDS 1.10**. Les organisateurs fournissent un prétraitement des images brutes basé sur un recalage **elastix** et une correction du champ de biais par **SPM12**. Le prétraitement est refait autrement ici : la T1 est recalée sur la FLAIR par un recalage rigide (**ANTs**, information mutuelle comme critère, exécution déterministe) et le champ de biais est corrigé par **N4**, estimé dans un masque du cerveau produit par **HD-BET** (sur GPU local), identique sur tous les sites.

**Contrôle de notre recalage contre celui des organisateurs** (169 sujets) : la T1 brute alignée par chacun des deux recalages est comparée (corrélation), et les deux transformations sont appliquées à 5 000 points de la tête (écart en mm, rotation relative).

| | Corrélation avec elastix | Écart moyen (mm) | Écart maximal (mm) | Rotation relative (°) |
|---|---|---|---|---|
| Médiane | 0,998 | 0,14 | 0,29 | 0,11 |
| Pire cas | 0,974 | 0,69 | 1,36 | 0,75 |
| Médiane par scanner | 0,988 (GE 1,5T) à 0,999 (Utrecht) | 0,11 à 0,46 | 0,18 à 0,84 | 0,06 à 0,35 |

Les deux recalages sont équivalents à une fraction de voxel près. **Un seul sujet sur 170 est signalé** (sub-146, corrélation 0,905, écart de 2 mm) : sa FLAIR, très bruitée et de contraste atypique, rend l'information mutuelle peu discriminante ; la transformation d'elastix est utilisée pour lui seul, comme exception documentée (J-039).

**Effet de ce prétraitement sur la segmentation** : appliqué aux images prétraitées par les organisateurs, le même modèle nnU-Net obtient exactement le même Dice (0,803 contre 0,802, p = 0,91 ; J-057). Notre chaîne n'apporte pas de gain de performance ; elle apporte la maîtrise et la traçabilité depuis les données brutes.

Pour les cartes de groupe, chaque sujet est recalé sur le modèle **MNI152** par un recalage non linéaire **SyN**. La corrélation croisée (CC) remplace l'information mutuelle comme critère : elle est plus gourmande en calcul, mais indispensable pour suivre les **ventricules dilatés** des sujets atrophiés, que le réglage par défaut ne ramène pas dans le contour du modèle (sur un sujet atrophié : volume ventriculaire estimé 86 ml avec CC contre 26 ml avec le réglage par défaut). Le calcul sur le modèle à 2 mm divise le temps par 8 (6 min au lieu de 50 par sujet) pour une qualité presque identique (J-054).

### 2. Contrôle qualité

Indicateurs de qualité calculés en se basant sur les définitions de **MRIQC** (SNR, SNR de Dietrich, CNR, CJV, EFC, FBER, étendue du champ de biais), sur les images **brutes**. MRIQC ne gère pas la FLAIR ; ce qui a été adapté :
- **régions anatomiques issues de WMH-SynthSeg** (et non d'une classification d'intensité) : substance blanche érodée, **privée des HSB et d'une marge de 2 mm** autour d'elles (sinon le SNR dépend de la charge lésionnelle), substance grise réduite au cortex ;
- statistiques de tissu **dans le masque du cerveau HD-BET** ; air pris hors de la tête dilatée, **zéros exacts exclus** (le remplissage par des zéros fausse le FBER) ;
- **CJV et rapport SB/SG jugés sur la T1 seulement** : en FLAIR, substance blanche et grise ont presque la même intensité (rapport 0,87 à 0,99), et le CJV explose sans rapport avec la qualité ;
- T1 mesurée **sur sa grille d'origine** (le rééchantillonnage lisserait le bruit) ; pas de classifieur MRIQC (entraîné sur des T1 seulement).

Un z-score robuste est calculé **par scanner**, avec la médiane et la MAD plutôt que la moyenne et l'écart type, sensibles aux anomalies (un z global ne ferait que détecter les scanners). Ce z-score, ainsi qu'une Isolation Forest, donnent un classement **utilisable / à vérifier / à exclure** (z ≥ 3 : à vérifier ; z ≥ 5 : à exclure, recommandation de contrôle visuel, jamais automatique).

| Scanner | Utilisable | À vérifier | À exclure |
|---|---|---|---|
| Utrecht Philips 3T | 41 | 5 | 4 |
| Singapour Siemens 3T | 42 | 7 | 1 |
| Amsterdam GE 3T | 45 | 5 | 0 |
| Amsterdam Philips 3T | 7 | 2 | 1 |
| Amsterdam GE 1,5T | 8 | 1 | 1 |
| **Total** | **143** | **20** | **7** |

Principaux motifs : champ de biais de la FLAIR (9 sujets), SNR de la FLAIR (7), FBER (4). **Aucun lien avec la qualité de segmentation** : sur le test, Dice de M1 0,817 pour les 20 sujets signalés contre 0,799 (p = 0,54) ; les images du challenge ont été sélectionnées et nnU-Net absorbe les variations présentes.

**Validation sur des artefacts simulés** (TorchIO, 10 sujets utilisables, 4 niveaux croissants) : part des images détectées, et effet des mêmes artefacts sur la segmentation par nnU-Net (Dice sur l'image intacte : 0,787).

| Artefact | Détecté en FLAIR (niveaux 1 → 4) | Détecté en T1 (niveaux 1 → 4) | Dice de nnU-Net (niveau 1 → 4) |
|---|---|---|---|
| Champ de biais | 0 → 100 % | 10 → 100 % | 0,780 → **0,442** : détecté **avant** de nuire |
| Bruit | 30 → 70 % | 10 → 80 % | 0,781 → 0,699 |
| Images fantômes | 20 → 60 % | 80 → 100 % | 0,784 → 0,637 |
| **Mouvement** | **non concluant** | 50 → 70 % | **0,736** → 0,591 ; rappel lésionnel 0,72 → 0,53 dès le niveau 1 |

Aucune fausse alerte sur les images intactes. Le mouvement est le vrai risque : il efface les petites lésions dès le plus faible niveau et n'est pas détecté en FLAIR ; il l'est en partie sur la T1 de la même séance.

### 3. Segmentation

Trois méthodes sont comparées dans ce projet :

| | Méthode | Rôle |
|---|---|---|
| **M1** | **nnU-Net v2, ResEnc M**, 3d_fullres, FLAIR + T1 ; entraîné sur Kaggle (2 × T4, 250 epochs, ~10,5 h) sur le pli 0 de 5 plis stratifiés par scanner et par charge lésionnelle (48 cas d'entraînement, 12 de validation) | Méthode principale |
| M2 | **WMH-SynthSeg** (FreeSurfer), modèle publié, utilisé sans modification (CPU, ~6 min et 26 à 30 Go de RAM par sujet) | Modèle généraliste prêt à l'emploi |
| M3 | Seuillage de la FLAIR : z-score robuste dans le cerveau (seuil 2,2), profondeur minimale (14 mm du bord du cerveau) et taille minimale (40 mm³), réglés par recherche sur grille (1 674 combinaisons) sur l'entraînement | Référence classique |

**Variantes testées** :
- **M1 avec augmentation forte** (DA5, même budget) : aucun gain, y compris sur les scanners inconnus (Dice 0,798 contre 0,802, p Holm = 1) ;
- **M1 sur le prétraitement des organisateurs** : même Dice (0,803), voir plus haut ;
- **M3 par scanner** : le seuil optimal varie de 1,7 (Singapour) à 2,6 (Utrecht), signe de la dépendance d'une règle d'intensité au scanner ; il n'est donné qu'à titre d'information, un seul seuil global est utilisé ;
- **M3 avec 4 normalisations d'intensité** : voir Harmonisation.

### 4. Évaluation

Le script **officiel** du challenge (copié sans modification) produit les métriques suivantes : Dice, HD95, AVD, rappel lésionnel et F1 lésionnel. Les intervalles de confiance à 95 % sont produits par **bootstrap stratifié par scanner** (2 000 tirages), pour chaque métrique.

| Méthode (110 cas de test) | Dice ↑ | HD95 (mm) ↓ | AVD (%) ↓ | Rappel lésionnel ↑ | F1 lésionnel ↑ | Classement |
|---|---|---|---|---|---|---|
| **M1 nnU-Net** | **0,802** [0,784-0,820] | **6,1** [4,8-7,7] | **15,4** [13,1-18,0] | **0,728** [0,709-0,746] | 0,783 [0,768-0,798] | **9e / 58** |
| M1 + augmentation forte (DA5) | 0,798 [0,779-0,815] | 6,3 [4,7-8,0] | 16,3 [14,1-18,9] | 0,727 [0,707-0,744] | **0,786** [0,769-0,801] | 10e / 58 |
| M1 sur le prétraitement du challenge | 0,803 [0,783-0,821] | 6,4 [4,9-8,3] | 16,1 [13,3-19,0] | 0,724 [0,703-0,744] | 0,780 [0,762-0,796] | 10e / 58 |
| M3 seuillage | 0,595 [0,558-0,628] | 22,7 [20,9-24,3] | 72,7 [53,0-95,9] | 0,229 [0,210-0,249] | 0,284 [0,267-0,300] | 55e / 58 |
| M2 WMH-SynthSeg | 0,402 [0,361-0,442] | 14,0 [12,6-15,5] | 293 [210-389] | 0,428 [0,399-0,457] | 0,451 [0,432-0,471] | 55e / 58 |

Le classement est recalculé avec la formule officielle, chaque méthode étant insérée parmi les 57 équipes publiées. M1 est **2e au Dice, 1er à l'AVD**, mais **22e au rappel** : son point faible est la détection des petites lésions.

| Dice par scanner | Utrecht | Singapour | Ams. GE 3T | Ams. Philips 3T (inconnu) | Ams. GE 1,5T (inconnu) |
|---|---|---|---|---|---|
| M1 nnU-Net | 0,786 | 0,834 | 0,796 | 0,776 | 0,804 |
| M3 seuillage | 0,565 | 0,584 | 0,618 | 0,610 | 0,634 |
| M2 WMH-SynthSeg | 0,413 | 0,495 | 0,333 | 0,392 | 0,309 |

![Dice par méthode et par scanner](results/figures/dice_by_method_and_scanner.png)

Une **comparaison appariée** (différence sujet par sujet) est également réalisée pour chaque paire de méthodes et chaque métrique, avec le test de **Wilcoxon** et la correction de **Holm** sur les 140 tests (8 méthodes évaluées, 28 paires, 5 métriques).

| Comparaison (A − B) | Dice | HD95 (mm) | AVD (points) | Rappel | F1 |
|---|---|---|---|---|---|
| M1 − M3 | **+0,208** [0,186 ; 0,232] | **−16,6** | **−57** | **+0,50** | **+0,50** |
| M1 − M2 | **+0,400** [0,372 ; 0,429] | **−7,9** | **−278** | **+0,30** | **+0,33** |
| M3 − M2 | **+0,193** | +8,7 | **−221** | −0,20 | −0,17 |
| M1 − DA5 | +0,005 (n.s.) | −0,1 (n.s.) | −0,9 (n.s.) | +0,002 (n.s.) | −0,003 (n.s.) |
| M1 − M1 prétraitement challenge | −0,001 (n.s.) | −0,3 (n.s.) | −0,7 (n.s.) | +0,005 (n.s.) | +0,003 (n.s.) |

Les trois premières lignes : toutes les différences sont significatives (p Holm < 1e-9) ; n.s. : p Holm = 1. M3 et M2 ne se dominent pas : M3 a le meilleur recouvrement et le meilleur volume, M2 trouve plus de lésions et fait des erreurs moins éloignées.

**Scanners connus contre inconnus** (groupes de patients différents, donc test de **Mann-Whitney** et bootstrap de chaque groupe séparément) :

| M1 nnU-Net | Connus (90) | Inconnus (20) | Inconnus − connus [IC 95 %] | p |
|---|---|---|---|---|
| Dice | 0,805 | 0,790 | −0,015 [−0,066 ; +0,030] | 0,45 |
| HD95 (mm) | 5,6 | 8,5 | +2,9 [−2,2 ; +8,7] | 0,96 |
| AVD (%) | 15,6 | 14,4 | −1,2 [−6,3 ; +4,3] | 0,94 |
| Rappel lésionnel | 0,723 | 0,754 | +0,031 [−0,017 ; +0,077] | 0,17 |

Aucune dégradation démontrée, pour aucune méthode (avec Holm sur les 40 tests, plus petit p corrigé : 0,07). Avec 20 cas, les intervalles restent larges.

**Plafond inter-observateurs** : les experts O3 et O4 sont notés contre O1 avec exactement les mêmes métriques. Les 12 cas de validation du pli 0 sont les seuls où M1 (qui ne les a pas vus) et les experts sont jugés sur les mêmes images.

| Contre la référence O1 (12 cas) | Dice | HD95 (mm) | Rappel | F1 |
|---|---|---|---|---|
| **M1 nnU-Net** | **0,815** | **4,3** | 0,686 | 0,762 |
| Expert O3 | 0,757 | 5,7 | 0,641 | 0,721 |
| Expert O4 | 0,781 | 4,9 | 0,686 | 0,760 |
| *O4 contre O3 (deux experts indépendants)* | *0,740* | *8,0* | | *0,763* |

Sur les 60 cas d'entraînement, O3 et O4 obtiennent 0,770 et 0,785 contre O1, et 0,759 entre eux. M1 est au niveau des experts ; il a appris le style de O1, contre lequel il est jugé, ce n'est donc pas « mieux qu'un expert ».

### 5. Biomarqueurs

Volume de HSB (ml et % du volume intracrânien), découpe **périventriculaire / profonde** (règle des 10 mm autour des ventricules segmentés par WMH-SynthSeg), nombre et taille des lésions, **cartes de fréquence lésionnelle dans l'espace MNI** par site. Accord avec la référence : **Bland-Altman** sur l'échelle logarithmique (les erreurs croissent avec le volume), **ICC** d'accord absolu, biais par site.

| Volume de HSB contre O1 | Rapport moyen [limites d'accord 95 %] | ICC [IC 95 %] |
|---|---|---|
| **M1 nnU-Net** (test, 110) | **0,98** [0,62-1,55] | **0,982** [0,975-0,988] |
| Expert O3 (entraînement, 60) | 0,97 [0,60-1,57] | 0,981 |
| Expert O4 (entraînement, 60) | 0,91 [0,55-1,51] | 0,977 |
| M3 seuillage | 1,18 [0,33-4,14] | 0,814 |
| M2 WMH-SynthSeg | 2,44 [0,41-14,5] | 0,372 |

- Charge médiane (O1) de 0,44 % (Amsterdam GE 3T) à 1,18 % (Singapour) du volume intracrânien.
- Lésions **profondes** moins bien mesurées que périventriculaires (M1 : rapport 0,87 [0,37-2,04], ICC 0,964, contre 1,00 [0,64-1,56], ICC 0,982).
- Le biais dépend du site pour toutes les sources, **experts compris** (Kruskal-Wallis p < 0,001) : M1 sous-estime à Utrecht (0,87), mais l'expert O4 aussi à Singapour (0,80). Une partie du « biais de site » reflète le style de la référence.
- Cartes MNI : corrélation avec la carte de O1 de **0,995 pour M1**, 0,90 pour M3, 0,88 pour M2, dont l'excès se concentre dans un liseré le long des ventricules.

### 6. Harmonisation

**Normalisation d'intensité**, testée par son effet sur le seuillage, site par site :

| Normalisation de la FLAIR | Dice du seuillage | Effet du site sur le Dice (Kruskal-Wallis) |
|---|---|---|
| Aucune | 0,203 | fort (p < 0,001) ; Dice 0,00 sur 2 scanners |
| **z-score robuste** | **0,595** | aucun (p = 0,90) |
| WhiteStripe | 0,564 | aucun (p = 0,94) |
| Appariement d'histogrammes | 0,522 | aucun (p = 0,52) |

**Signature de site et ComBat** : 20 caractéristiques de la substance blanche saine (FLAIR et T1), **classifieur de site** en validation croisée (5 plis), **ComBat** (neuroHarmonize) appris dans les plis d'entraînement, avec la charge lésionnelle et l'ICV comme covariables protégées.

| Caractéristiques de la substance blanche saine | Scanner deviné (hasard 20 %) | Variance due au site | Lien avec la charge lésionnelle (\|ρ\|) |
|---|---|---|---|
| Sans normalisation | 90 % | 71 % | 0,24 |
| z-score robuste | 73 % | 21 % | 0,40 |
| z-score + **ComBat** | **28 %** | **2 %** | **0,44** |

La normalisation par image ne retire qu'une partie de l'empreinte du scanner ; ComBat l'efface presque entièrement, et le lien avec la biologie ressort même un peu mieux.

### 7. Reproductibilité

**Snakemake** (chaîne complète, étapes Kaggle décrites comme externes), environnement verrouillé (**uv**), **155 tests** pytest sur données synthétiques, **GitHub Actions** (lint, tests, construction de l'image Docker), **Docker** et **Apptainer**, script **SLURM** pour Jean Zay (5 plis en parallèle, non testé).

**Coût de calcul** : environ **5,5 kWh et 2,3 kg CO₂eq** au total, dont 98 % pour les calculs sur Kaggle (deux entraînements nnU-Net de ~10 h sur 2 × T4, WMH-SynthSeg sur CPU) ; mesuré avec **CodeCarbon** pour les étapes locales, estimé pour les autres ([`carbon_footprint.csv`](results/tables/carbon_footprint.csv)).

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
justification.md            journal des décisions (J-001 à J-065)
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
