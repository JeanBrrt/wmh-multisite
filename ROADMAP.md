# ROADMAP — wmh-multisite

> Document vivant : il dit **où on en est** et **où on va**. Les estimations sont grossières (temps d'implémentation à deux, hors temps de calcul) et seront corrigées au fil de l'eau.
> Les justifications des choix sont dans [justification.md](justification.md) (références `J-xxx`).

**Échéance** : candidature CNRS (CREATIS, BrainTwin) — **jeudi 8 octobre 2026, 23h59**.

**Statuts** : `[x]` fait · `[ ]` à faire (précisé « en cours » ou « en attente » si besoin) · ~~barré~~ abandonné
**Priorités** : **P0** = indispensable (version minimale présentable) · **P1** = attendu · **P2** = bonus si le temps le permet

---

## Où en est-on ?

| Date | État |
|---|---|
| 2026-10-03 | **Phases 0 et 1 validées** (hors 1.5 : dépôt GitHub). **Phase 2 : 2.1 et 2.2 faites** (jeu complet téléchargé et vérifié, BIDS + inventaire QC ; notebook `02_explore_orig_pre` : fichiers orig/pre, défacement, reproduction exacte du recalage elastix, champ de biais). 2.3 faite (70 tests). Décision reportée : nombre d'epochs nnU-Net (5.2). |

**Prochaine action** : Phase 3 (3.1 N4, 3.2 recalage rigide, 3.4 export nnU-Net des 60 cas d'entraînement), puis 5.1-5.2 pour lancer l'entraînement ce soir.

---

## Vue d'ensemble

```
orig/ (FLAIR + T1 3D brutes, 5 scanners)
  │ Phase 2  Données : téléchargement API + BIDS
  │ Phase 3  Prétraitement : N4, recalage T1→FLAIR, masque cerveau, SyN→MNI   (PC, CPU)
  │ Phase 4  Contrôle qualité : IQMs, outliers, validation par artefacts       (PC, CPU)
  ▼
  │ Phase 5  Segmentation : M1 nnU-Net (Kaggle GPU) · M2 WMH-SynthSeg (Kaggle CPU) · M3 seuillage (PC)
  │ Phase 6  Évaluation : script officiel, par scanner, connus vs inconnus
  │ Phase 7  Biomarqueurs : volume, péri-ventriculaire/profond, lésions, accord
  │ Phase 8  Harmonisation : intensités + ComBat
  ▼
  Phase 9  Reproductibilité : Snakemake, tests, CI, conteneurs, SLURM
  Phase 10 Restitution : README, figures, tag v1.0
```

| Jour | Date | Contenu principal |
|---|---|---|
| J0 | ven. 2 | Phase 0 (cadrage, faisabilité) |
| J1 | sam. 3 | Phases 1, 2, 3 (60 cas d'entraînement en priorité) → **lancement de l'entraînement nnU-Net le soir** (session 1/2) + HD-BET sur Kaggle |
| J2 | dim. 4 | Entraînement nnU-Net session 2/2, Phase 3 (fin, 110 cas test + SyN la nuit), Phase 4, seuillage (5.5), WMH-SynthSeg (session CPU) |
| J3 | lun. 5 | Inférence nnU-Net (5.3), Phase 6 |
| J4 | mar. 6 | Phases 7 et 8 |
| J5 | mer. 7 | Phase 9, Phase 10 (README) |
| J6 | jeu. 8 | Marge, lettre de motivation, envoi |

**Charge estimée** : ~40 h d'implémentation sur 5 jours → c'est **serré**. Si on prend du retard, couper dans l'ordre : P2, puis 8.4, 4.3, 7.5, 9.4.

---

## Phase 0 — Cadrage et faisabilité · J0 · P0 · validée le 2026-10-03

| # | Tâche | Statut | Notes |
|---|---|---|---|
| 0.1 | Analyse de l'offre et du CV, choix du sujet parmi 3 projets | [x] | J-001 |
| 0.2 | Choix et vérification du jeu de données (contenu, licence, accès API, en-têtes NIfTI) | [x] | J-002, J-004 |
| 0.3 | Test des outils sur la machine locale | [x] | 2 plantages → règles de sécurité, J-005 |
| 0.4 | Notebook de faisabilité Kaggle ([kaggle/00_feasibility_test.ipynb](kaggle/00_feasibility_test.ipynb)) | [x] | Rapport du 2026-10-02, ~1,1 h de GPU |
| 0.5 | Décisions issues du rapport Kaggle | [x] | voir tableau ci-dessous ; le nombre d'epochs nnU-Net est reporté à l'étape 5.2 |
| 0.6 | Nom du projet, arborescence, ROADMAP, journal de justification | [x] | J-003 |

**Décisions issues du rapport Kaggle (0.5)**

| Mesure | Résultat | Décision | Statut |
|---|---|---|---|
| Recalage rigide vs elastix | corr. 0,981-0,993 ; ~15 s/sujet | ANTs par défaut suffit | [x] (J-016) |
| SyN → MNI | 78,6 s (4 cœurs) | En local, la nuit | [x] (J-017) |
| HD-BET | 5,8 Go VRAM, 10,5 Go RAM | Kaggle GPU ; N4 de l'entrée nnU-Net avec masque d'Otsu | [x] (J-018) |
| WMH-SynthSeg | 300 s, 26-29 Go RAM ; Dice 0,48 / 0,73 / 0,22 | Sous-ensemble de 50 sujets | [x] (J-020) |
| nnU-Net 3d_fullres | 194 s/epoch, 6,8 Go VRAM | Configuration 3d_fullres retenue ; nombre d'epochs (250 en 2 sessions ≈13,5 h, ou 150 ≈8 h) tranché à l'étape 5.2 | [x] (J-019) |

---

## Phase 1 — Socle du dépôt · J1 matin · P0 · ~2 h 30 · validée le 2026-10-03 (sauf 1.5)

| # | Tâche | Estim. | Statut |
|---|---|---|---|
| 1.1 | `git init`, `.gitignore` (`data/`, `ressource/`, `*.nii*`, poids, sorties lourdes), `LICENSE` (MIT) ; filtre Git **nbstripout** (`.gitattributes`) : sorties des notebooks retirées au commit, conservées en local (J-023) | 20 min | [x] |
| 1.2 | `pyproject.toml` (dépendances de base + extras `qc`, `harmonize`, `seg`, `workflow`, `dev`, `local`) ; environnement **uv** (`uv sync --extra local`, `uv.lock` versionné, J-021) ; correctif neuroHarmonize (J-022) | 45 min | [x] |
| 1.3 | `config/config.yaml` : chemins, liste des scanners/sites, split train/test, paramètres par étape | 30 min | [x] (valeurs TBD tracées ; `flair_acquisition` = protocole du readme) |
| 1.4 | `utils/io.py` (load, load_data, load_labels, save_like, same_grid, describe) et `utils/guard.py` (watchdog RAM/VRAM, J-005) ; notebook d'exploration `notebooks/01_explore_nifti.ipynb` | 45 min | [x] (testés à la main sur Utrecht 0 ; tests pytest en 9.2) |
| 1.5 | Dépôt GitHub, premier commit et push | 10 min | [~] premier commit le 2026-10-06 (159 fichiers, aucune donnée ni document personnel ; `notes/` laissé à l'utilisateur) ; dépôt GitHub à créer par l'utilisateur |

**Fini quand** : `pip install -e .` fonctionne et `import wmh_multisite` passe.

---

## Phase 2 — Données · J1 · P0 · ~2 h 30 (+ ~30-60 min de téléchargement) · 2.1 et 2.2 faites le 2026-10-03

| # | Tâche | Estim. | Statut |
|---|---|---|---|
| 2.1 | `data/download.py` : listing via l'API DataverseNL, filtre des fichiers utiles (`orig/`, `wmh.nii.gz`, `pre/FLAIR` et `pre/T1`, observateurs O3/O4, readme ; **pas** `pre/3DT1`, J-015), téléchargement parallèle, vérification **SHA-1** (empreinte publiée), reprise, `data/raw/manifest.csv` (J-026) | 1 h | [x] validée (1 551 fichiers dont les 70 `FLAIR_mask` d'Amsterdam, J-028 ; 4,08 Go, 0 échec) |
| 2.2 | `data/bids.py` : BIDS 1.10 (`sub-000` à `sub-169`, J-025), sidecars JSON (protocole du config), `participants.tsv/json`, dérivés `manual` (O1, O3, O4, `dseg.tsv`) et `challenge` (elastix, SPM12, masques de défacement `space-T1w` et `space-FLAIR`, J-028), liens physiques ; inventaire QC `results/tables/bids_inventory.csv` | 1 h | [x] (170 sujets, aucune anomalie de géométrie ni de labels ; 1 T1 hors protocole : sub-017, J-027) |
| 2.3 | Tests pytest sur petits volumes synthétiques et faux serveur HTTP (`tests/` : io, download, bids, guard, config), J-029 | 30 min | [x] (70 tests, 5,6 s, couverture 91 % ; 1 bug corrigé dans `bids.py`) |

**Fini quand** : 170 sujets en BIDS, `participants.tsv` cohérent avec le readme du challenge (20/20/20 train, 30/30/30/10/10 test).

---

## Phase 3 — Prétraitement · J1 → J2 · P0 · ~5 h (+ calcul)

Entrée : `orig/` brut. Sortie : FLAIR + T1 dans l'espace FLAIR natif, masques, transformations (J-010).

| # | Tâche | Estim. | Prio | Statut |
|---|---|---|---|---|
| 3.1 | `preproc/bias.py` : correction N4 (ANTs) de la FLAIR et de la T1 brutes, masque de tête (Otsu + plus grande composante + remplissage des trous, J-018) ; sorties `desc-n4`, `desc-biasfield`, `desc-head_mask` ; audit `notebooks/03_n4_audit.ipynb` (J-030) | 45 min | P0 | [x] refait avec le masque HD-BET (J-036) : 120 images, 0 échec ; variante Otsu archivée ; 110 cas test à lancer (3.5) |
| 3.2 | `preproc/register.py` : recalage rigide T1 N4 -> FLAIR N4 (information mutuelle de Mattes), rééchantillonnage dans la grille FLAIR ; QC contre elastix (corrélation + écart géométrique), mono-thread pour le déterminisme (J-031) | 1 h 30 | P0 | [x] refait sur images brutes (J-036) : 60 cas, 0 alerte, corrélation médiane 0,998 ; 110 cas test à lancer (3.5) |
| 3.3 | `preproc/brainmask.py` + `preproc/hdbet_worker.py` : HD-BET en local sur GPU (environnement uv temporaire CUDA, lots sous watchdog, plafond mémoire GPU), masques vérifiés puis transportés sur la grille FLAIR, QC ; contrôle visuel `notebooks/04_brainmask_check.ipynb` (J-033) | 45 min | P0 | [x] 170 masques (0 alerte, 47 min) ; grille FLAIR : 60 cas d'entraînement, les 110 cas test après 3.5 |
| 3.4 | **Jalon J1 soir** : 60 cas d'entraînement prétraités (N4 + recalage) et exportés au format nnU-Net ; **dépôt du dataset Kaggle privé (utilisateur)** | 45 min | P0 | [x] dataset déposé sur Kaggle (N4 HD-BET) |
| 3.5 | Prétraitement des 110 cas test (nuit J1 → J2) | calcul | P0 | [x] 110 cas test : recalage (sub-146 : transformation elastix, J-039), masques du cerveau, N4 ; 0 échec |
| 3.6 | `preproc/atlas.py` : SyN T1 → MNI (~1 min/sujet, local la nuit, J-017), ventricules d'un atlas propagés en espace FLAIR puis ajustés sur la FLAIR (C) ; prototype C2 dans le notebook 06 | 1 h 30 | P1 | [x] arrêté au prototype (J-045) : code et notebook 06 conservés, SyN lancé sur 5 sujets seulement ; ventricules de 7.2 pris dans WMH-SynthSeg ; atlas + C seulement pour un sujet où WMH-SynthSeg échouerait |

**Fini quand** : 170 sujets prétraités, tableau QC du recalage (corrélation par sujet et par site), aucun échec silencieux.

---

## Phase 4 — Contrôle qualité des images · J2 · P1 · ~4 h

| # | Tâche | Estim. | Prio | Statut |
|---|---|---|---|---|
| 4.1 | `qc/iqm.py` : SNR, CNR, CJV, EFC, FBER, contraste SB/SG sur FLAIR et T1 (définitions MRIQC, J-011) **Rappeler J-028 avant d'implémenter.** | 1 h 30 | P1 | [x] 170 sujets, 15 min ; CJV et rapport SB/SG jugés pour la T1 seulement (J-051) |
| 4.2 | `qc/outliers.py` : z-scores robustes + IsolationForest → classes *utilisable / à vérifier / à exclure* **Rappeler J-028 avant d'implémenter.** | 1 h | P1 | [x] 136 usable / 30 check / 4 exclude ; aucun lien avec le Dice de M1 (p = 0,90) (J-051) |
| 4.3 | `qc/artifacts.py` : injection d'artefacts TorchIO (mouvement, bruit, biais, ghosting) à intensité croissante → courbe de détection | 1 h | P1 | [x] biais et bruit détectés, mouvement non concluant (artefact de la simulation), 0 fausse alerte (J-051) ; **extension faite** : nnU-Net sur FLAIR dégradées (10 sujets de test x 4 artefacts x 4 niveaux) : le mouvement nuit dès le niveau 1 (Dice 0,79 -> 0,74, rappel 0,72 -> 0,53) sans être détecté ; le biais est détecté avant de nuire (J-058) |
| 4.4 | `viz/report.py` : rapport QC HTML (tableau + vignettes des cas signalés) | 45 min | P1 | [x] `results/qc/qc_report.html`, 28 sujets avec vignettes, local (J-051) |

---

## Phase 5 — Segmentation · J1 soir → J3 · P0 · ~5 h (+ calcul)

| # | Tâche | Où | Estim. | Prio | Statut |
|---|---|---|---|---|---|
| 5.1 | `seg/nnunet.py` : export `Dataset001_WMH` (FLAIR N4 `_0000`, T1 N4 recalée `_0001`, labels uint8 avec **2 = ignore**), `dataset.json`, `case_mapping.csv`, `splits_final.json` stratifié par site et charge lésionnelle, archive pour Kaggle (J-032) | PC | 30 min | P0 | [x] 60 cas, vérificateur nnU-Net OK, archive 841 Mo |
| 5.2 | Mesure de la vitesse par epoch (`kaggle/01_epoch_benchmark.ipynb`), puis notebook Kaggle `03_nnunet_train` : plan + prétraitement + entraînement 3d_fullres (fold 0 ou `all`), sauvegarde des checkpoints, reprise `--c` (J-019). **Trancher ici : 250 ou 150 epochs** | Kaggle GPU | 1 h + ~13,5 h calcul (2 sessions) | P0 | [x] ResEnc M 250 epochs terminé (10,7 h, 149 s/epoch) ; Dice de validation fold 0 = 0,815 (J-038) ; 2e entraînement DA5 fait : 0,798 sur le test, pas de gain, y compris sur les scanners inconnus (J-052) |
| 5.3 | Notebook Kaggle `02_nnunet_infer` : inférence sur les 110 cas test (notre prétraitement) **et** sur `pre/` du challenge (expérience « effet du prétraitement », J-010) | Kaggle GPU | 45 min + ~1 h calcul | P0 | [x] modèle ResEnc M : inférence locale des 110 cas test (J-040), identique à Kaggle ; Dice rapide 0,803 (évaluation officielle en 6.1) ; DA5 fait (J-052) ; **effet du prétraitement fait** : sur les images `pre/` du challenge, Dice identique (0,803 contre 0,803, p = 0,91) ; N4 atténue le contraste des fortes charges (J-057) |
| 5.4 | Notebook Kaggle `05_wmhsynthseg` : mode CPU d'origine, **170 sujets** en 3 parties + contrôle `CHECK` (5 sujets), un sous-processus par sujet, segmentation complète conservée pour les ventricules de 7.2 (J-007, J-043, J-045) | Kaggle CPU | 45 min + ~21 h calcul (3 sessions de ~7 h en parallèle) | P1 | [x] 170/170 sujets (CHECK validé, J-045) ; test : Dice 0,402, sur-segmentation (volume x2) ; ICV et ventricules utilisés en 7.1 et 7.2 (J-053) |
| 5.5 | `seg/threshold.py` : FLAIR normalisée dans le cerveau, seuil réglé sur le train (recherche sur grille, Dice), ~~restriction à la substance blanche~~ remplacée par une profondeur minimale sous le bord du cerveau (J-044) **Rappeler J-028 avant d'implémenter.** | PC | 1 h 30 | P0 | [x] t = 2,2, profondeur 14 mm, taille 40 mm³ (J-044) ; test : Dice 0,595, rappel lésionnel 0,229 |

---

## Phase 6 — Évaluation · J3 · P0 · ~3 h 30

| # | Tâche | Estim. | Prio | Statut |
|---|---|---|---|---|
| 6.1 | `evaluate/metrics.py` : intégration du script officiel du challenge (licence MIT, J-012) — Dice, HD95, AVD, rappel et F1 lésionnels, label 2 ignoré | 45 min | P0 | [x] ResEnc M sur 110 cas : Dice 0,803, HD95 6,1 mm, AVD 15,4 %, rappel 0,728, F1 0,783 ; 9e/58 au classement (J-042) ; à refaire pour DA5, WMH-SynthSeg, seuillage |
| 6.2 | `evaluate/stats.py` : résultats par scanner, **connus vs inconnus**, IC bootstrap, tests appariés entre méthodes | 1 h | P0 | [x] 6 tests ; M1 0,803 [0,784-0,820], M3 0,595 [0,560-0,629] ; aucune dégradation démontrée hors domaine pour M1 (J-047) ; à relancer quand M2 et DA5 sont évalués |
| 6.3 | Accord inter-observateurs (O3, O4 vs référence) sur le train = plafond humain | 30 min | P1 | [x] `evaluate/interobserver.py`, 3 tests ; Dice O3 0,770, O4 0,785 contre O1 ; M1 0,815 sur les 12 cas de validation appariés (J-046) |
| 6.4 | `viz/figures.py` : mosaïque de cas par scanner, barres par scanner et par méthode, positionnement vs leaderboard | 1 h 15 | P0 | [x] 4 figures de synthèse ; classement officiel (readme.pdf) : M1 9e/58, DA5 10e, M3 et M2 55e (J-060) |

---

## Phase 7 — Biomarqueurs · J4 matin · P0/P1 · ~3 h 30

| # | Tâche | Estim. | Prio | Statut |
|---|---|---|---|---|
| 7.1 | `biomarkers/volumes.py` : volume HSB (ml), fraction du volume cérébral (masque HD-BET) ; **À NE PAS OUBLIER : ajouter la normalisation par le volume intracrânien (ICV) tiré de la segmentation complète de WMH-SynthSeg dès qu'elle est disponible** **Rappeler J-028 avant d'implémenter.** | 30 min | P0 | [x] volumes et fraction du masque HD-BET (J-048) ; ICV ajouté (WMH-SynthSeg, médiane 1 450 ml, J-053) |
| 7.2 | `biomarkers/location.py` : carte de distance aux ventricules, HSB péri-ventriculaires (≤ 10 mm) vs profondes ; ventricules de WMH-SynthSeg, labels 4 + 43 (J-045). **Avant** : contrôle des ventricules sur les 170 sujets (volume par scanner, rapport gauche/droite, sujets extrêmes vus sur la FLAIR), les 5 sujets de `CHECK` ne couvrant que 2 scanners | 1 h 30 | P1 | [x] règle des 10 mm, ventricules WMH-SynthSeg (J-050), 170 sujets ; M1 : périventriculaire bien mesuré (ICC 0,98), profond sous-estimé (0,87, ICC 0,96) (J-053) |
| 7.3 | `biomarkers/lesions.py` : composantes connexes, nombre et distribution des tailles | 30 min | P1 | [x] 26-connexité comme le script officiel, compte >= 10 mm³ comparable entre sites (J-048) |
| 7.4 | Accord prédit vs manuel : Bland-Altman, ICC, biais par site | 45 min | P0 | [x] Bland-Altman sur l'échelle log, ICC(A,1), biais par site + Kruskal-Wallis ; M1 au niveau des experts (ICC 0,98), biais de site chez M1 (Utrecht 0,87), M3 (Utrecht 1,72) et chez les humains (J-049) |
| 7.5 | Carte de fréquence lésionnelle dans l'espace MNI, par site ; suppose SyN sur les 170 sujets, non lancé (J-045) | 45 min + ~3 h calcul | P2 | [x] SyN CC 2 mm, 170 sujets ; cartes par site et par méthode ; corrélation avec O1 : M1 0,995, M3 0,90, M2 0,88 ; M2 sur-segmente le long des ventricules (J-054, J-056) |

---

## Phase 8 — Harmonisation · J4 après-midi · P1 · ~3 h

Pas de données démographiques → le « signal biologique à conserver » est la charge lésionnelle manuelle (J-004).

| # | Tâche | Estim. | Prio | Statut |
|---|---|---|---|---|
| 8.1 | `harmonize/intensity.py` : aucune / z-score / WhiteStripe / appariement d'histogrammes **Rappeler J-028 avant d'implémenter.** | 1 h | P1 | [x] none / zscore / whitestripe / histmatch, dans le cerveau (J-028), 6 tests (J-055) |
| 8.2 | Caractéristiques de la substance blanche d'apparence normale + classifieur de site (validation croisée) **Rappeler J-028 avant d'implémenter.** | 45 min | P1 | [x] 20 caractéristiques SB saine (FLAIR, T1) ; site deviné à 90 % (brut), 73 % (z robuste) (J-059) |
| 8.3 | `harmonize/combat.py` : ComBat (neuroHarmonize), ajusté sur le train ; précision du classifieur de site avant/après, conservation de l'association avec la charge lésionnelle | 45 min | P1 | [x] ComBat appris dans les plis : site deviné à 28 % (hasard 20 %), variance due au site 2 %, lien avec la charge lésionnelle conservé (J-059) |
| 8.4 | Effet de la normalisation d'intensité sur le seuillage (M3), par site | 30 min | P2 | [x] sans normalisation le seuillage s'effondre (0,20, 0,00 sur 2 sites) ; zscore meilleur (0,595), Dice indépendant du site pour les 3 normalisations par image (J-055) |

---

## Phase 9 — Reproductibilité et ingénierie · J5 · P1 · ~4 h 30

| # | Tâche | Estim. | Prio | Statut |
|---|---|---|---|---|
| 9.1 | `workflow/Snakefile` + règles : chaîne locale de bout en bout (étapes Kaggle documentées comme étapes externes) | 1 h 30 | P1 | [x] Snakefile par étapes, règles externes Kaggle, `ancient()` ; état existant enregistré, `-n` : rien à faire (J-061) |
| 9.2 | Tests pytest (fixtures synthétiques : sphères, bruit) | 1 h | P1 | [x] 155 tests sur données synthétiques, sans réseau (J-029) |
| 9.3 | GitHub Actions : ruff, pytest, construction + test rapide de l'image Docker (Docker absent en local, J-005) | 45 min | P1 | [x] GitHub Actions : uv verrouillé, ruff, pytest + couverture ; Docker à ajouter avec 9.4 (J-061) |
| 9.4 | `containers/Dockerfile`, `containers/apptainer.def`, `hpc/slurm_jeanzay.sh` (fourni **non testé**, indiqué comme tel) | 1 h | P1 | [ ] |
| 9.5 | CodeCarbon : empreinte des étapes principales | 20 min | P2 | [ ] **non réalisé, par choix (2026-10-06)** : priorité donnée aux analyses ; à faire si le temps le permet (mesure de l'empreinte des étapes principales : prétraitement, inférence, QC) |

---

## Phase 10 — Restitution · J5 soir → J6 · P0 · ~3 h

| # | Tâche | Estim. | Prio | Statut |
|---|---|---|---|---|
| 10.1 | `README.md` : question, données, chaîne, résultats chiffrés, figures, limites, reproduction en 3 commandes | 2 h | P0 | [ ] |
| 10.2 | Relecture de `justification.md`, tag `v1.0` (option : DOI Zenodo) | 30 min | P1 | [ ] |
| 10.3 | Lettre de motivation : relier le projet aux 4 axes de l'offre (hors dépôt) | 1 h 30 | P0 | [ ] |

---

### Points à étudier (en fin de projet)

Constats repérés en cours de route, à examiner une fois le pipeline principal terminé. Chaque point indique où il a été vu.

| Point | Constat | Piste d'étude | Source |
|---|---|---|---|
| sub-017 (Utrecht, entraînement) | T1 hors protocole : 288 × 288 × 192 voxels de 0,89 mm, au lieu de 256 × 256 × 192 à 1 mm pour les 49 autres sujets d'Utrecht | Passe tous les contrôles jusqu'ici (N4, recalage, HD-BET : 1 449 ml, un seul bloc). Vérifier son comportement au QC (Phase 4), dans la segmentation et dans les biomarqueurs ; documenter s'il apparaît comme atypique | J-027, J-033 |
| Champ N4 de la FLAIR de Singapour | Désaccord avec SPM12 sur 19 sujets sur 20 ; le biais réel y paraît plus faible | Effet sur la segmentation (expérience « effet du prétraitement », 5.3) | J-030 |
| Test intermittent du watchdog | **Résolu le 2026-10-05** : condition de concurrence dans `utils/guard.py` (processus terminé entre deux appels psutil), corrigée et testée | — | J-051 |
| sub-146 (Amsterdam GE 3T, test) | Recalage T1 -> FLAIR signalé (2 mm d'écart avec elastix) ; FLAIR très bruitée, contraste atypique | Transformation d'elastix proposée (J-039) ; vérifier sa FLAIR au QC et interpréter sa segmentation avec prudence | J-039 |
| Déterminisme de N4 | ITK multi-thread : non testé pour N4 (corrigé pour le recalage) | Relancer N4 deux fois sur un sujet et comparer | J-031 |

### À mentionner dans le README (notes accumulées au fil du projet)

Particularités des données publiées, constatées pendant le projet :
- Les `FLAIR_mask` d'Amsterdam ne sont pas binaires (~1 % de voxels interpolés sur la frontière) ; la zone défacée est `mask == 0`.
- `pre/FLAIR` est stockée en float32, mais `pre/T1` a été ré-arrondie en uint16 après la correction de biais (écart d'arrondi ≤ 0,5 sur des intensités de l'ordre de 1 000) : incohérence de format sans conséquence pratique.
- Les masques de vérité terrain (`wmh.nii.gz`) et de défacement sont stockés en float32 alors qu'ils ne contiennent que des entiers : convertis en uint8 à la lecture (`io.load_labels`).
- Les FLAIR d'Amsterdam (3D) ont été défacées par les organisateurs (`FLAIR_mask`), pas celles d'Utrecht et de Singapour (2D), qui contiennent pourtant une partie de la région du visage (tissu FLAIR dans la zone défacée de la T1, sur les 170 sujets : Singapour médiane 40 ml [0-100], Utrecht 7 ml [0,6-88] ; Amsterdam 0,5-8 ml de résidus de bord, zone à 98-99 % de zéros). Risque de réidentification réduit (coupes de 3 mm, couverture partielle) mais non nul : le projet ne publie aucune image ni rendu 3D de la tête.
- `orig/T1` et `pre/T1` sont sous-échantillonnées ponctuellement (interpolation linéaire sans filtrage, reproduite exactement à partir du fichier elastix).
- sub-017 (Utrecht) : T1 hors protocole (288 × 288 voxels de 0,89 mm).
- Le dossier publié `Philips_VU .PETMR_01.` (espace, point final) est renommé localement, Windows supprimant les points finaux.

## Extensions (P2, hors périmètre de base)

| Idée | Coût | Intérêt |
|---|---|---|
| 2e entraînement nnU-Net sur `pre/` du challenge (effet du prétraitement à l'entraînement) | 6-9 h GPU | Axe 1 |
| WMH-SynthSeg en fp16 sur GPU, validé contre la sortie CPU (option B de J-007) | ~2 h | Coût de calcul |
| Fine-tuning de WMH-SynthSeg ou d'un modèle 3D pré-entraîné | ~1 j | Axe 2 |
| Extension pronostique sur une base avec données cliniques (ex. ISLES'24) | > 1 j | Axe 4 |

---

## Historique des modifications

| Date | Modification |
|---|---|
| 2026-10-03 | Création |
| 2026-10-03 | Intégration du rapport Kaggle : HD-BET sur Kaggle, SyN en local, N4 avec masque d'Otsu, nnU-Net ≈13,5 h, WMH-SynthSeg sur 50 sujets |
| 2026-10-04 | Ventricules de 7.2 : WMH-SynthSeg confirmé sur 5 sujets (J-045) ; atlas 3.6 arrêté au prototype ; contrôle des ventricules sur 170 sujets ajouté à 7.2 ; WMH-SynthSeg ≈7 h par partie |
