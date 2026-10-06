# Estimation de la densité mammaire en régression ordinale

Proof of concept : classifieur BI-RADS de densité mammaire (A/B/C/D) entraîné avec
une loss ordinal-aware (CORN, Shi et al. 2022) et comparé à une baseline
cross-entropy classique. Évaluation avec Quadratic Weighted Kappa (QWK).

Résultats du prototype et vérifications : voir `RAPPORT_PROTOTYPE.md`.

## 1. Le dataset : VinDr-Mammo (PhysioNet)

VinDr-Mammo (Nguyen et al., 2023) contient ~5000 études de mammographie
(~20 000 vues DICOM, 4 vues/étude : CC/MLO x gauche/droite), avec pour chaque
sein :
- `breast_density` : catégorie BI-RADS A/B/C/D (variable **ordinale**, cible de ce projet)
- `breast_birads` : score de suspicion BI-RADS 1-5
- Annotations de lésions (boîtes, findings) — non utilisées ici

### Accès et taille

Accès restreint : compte PhysioNet *credentialed* + signature de la Data Use
Agreement sur https://physionet.org/content/vindr-mammo/1.0.0/

**PhysioNet n'héberge aucun environnement de calcul** (pas de notebook, pas de
VM) et VinDr-Mammo n'a pas de miroir cloud GCP/AWS — c'est uniquement du
téléchargement. Le dataset complet fait ~50 Go, mais il n'est pas nécessaire :
un sous-ensemble stratifié de 400-500 études (~4-5 Go) suffit pour ce proof of
concept. Voir `scripts/download_vindr_subset.py` ou la section 2 du notebook.

Structure publiée par PhysioNet :

```
vindr-mammo/1.0.0/
  breast-level_annotations.csv   # study_id, series_id, image_id, laterality,
                                  # view_position, height, width, breast_birads,
                                  # breast_density, split
  finding_annotations.csv
  metadata.csv                    # âge patiente, modèle d'appareil
  images/<study_id>/<image_id>.dicom
```

`breast_density` vaut `DENSITY A` ... `DENSITY D`. `split` vaut `training`
(4000 études) ou `test` (1000 études) — split officiel stratifié, à conserver.

Deux pièges techniques traités dans le code : les DICOM sont en 16 bits avec
souvent `PhotometricInterpretation = MONOCHROME1` (échelle inversée — il faut
inverser, sinon le modèle apprend sur des négatifs) et le dataset est
**déséquilibré** (B et C dominent, A et D sont rares), géré par
`WeightedRandomSampler` plutôt qu'en supprimant des données.

## 2. En attendant l'accès : données synthétiques

`src/data.py::make_synthetic_breast_density` génère des images en niveaux de
gris dont la texture et la luminosité moyenne sont corrélées à une classe de
densité simulée A/B/C/D, avec du bruit de label sur les classes adjacentes
(imite l'accord inter-lecteur imparfait du BI-RADS). Cela permet de valider
toute la logique (loss ordinale → décodage du rang → métriques) sans images
réelles et sans PyTorch, via `src/prototype_numpy.py`.

**Important** : les résultats obtenus sur données synthétiques ne sont pas des
résultats scientifiques sur la densité mammaire — c'est un test de correction
de l'implémentation.

## 3. Structure du projet

```
breast_density_ordinal/
  notebooks/
    vindr_mammo_ordinal_density.ipynb   # LIVRABLE PRINCIPAL : Colab de bout en bout
  src/
    data.py              # loader VinDr-Mammo (DICOM) + générateur synthétique
    models.py            # ResNet18 + tête CE (baseline) + tête CORN
    losses.py            # cross-entropy, CORN, CORAL
    metrics.py           # QWK, MAE ordinal, histogramme des distances d'erreur
    train.py             # CLI d'entraînement sur les vraies données
    prototype_numpy.py   # prototype sans torch (validation de la logique)
  scripts/
    download_vindr_subset.py   # téléchargement stratifié du sous-ensemble PhysioNet
  results/
    prototype_metrics.json     # résultats des 5 seeds du prototype
  RAPPORT_PROTOTYPE.md         # résultats, vérifications, limites, étapes suivantes
```

## 4. Chemin recommandé : le notebook Colab

Ouvre `notebooks/vindr_mammo_ordinal_density.ipynb` dans Google Colab, active
un GPU (`Exécution > Modifier le type d'exécution > GPU T4`) et exécute les
cellules dans l'ordre. Le notebook gère tout : identifiants PhysioNet,
téléchargement du sous-ensemble, décodage DICOM, cache NPY, entraînement des
deux modèles, comparaison et figures.

Budget temps indicatif avec 400 études d'entraînement et 15 epochs : ~25 min de
téléchargement, ~10 min de mise en cache, ~30-45 min pour les deux
entraînements.

Attention : `/content` est effacé à la fin de la session Colab. La dernière
cellule sauvegarde les résultats et propose une copie vers Google Drive.

## 5. Hors Colab : ligne de commande

```bash
pip install -r requirements.txt

# 1. sous-ensemble stratifié depuis PhysioNet
python scripts/download_vindr_subset.py --user TON_LOGIN --out ./data/vindr_subset \
    --n-train-studies 400 --n-test-studies 120

# 2. les trois approches, stricte parité d'hyperparamètres
python src/train.py --model baseline --data-root ./data/vindr_subset --epochs 20
python src/train.py --model corn     --data-root ./data/vindr_subset --epochs 20
python src/train.py --model coral    --data-root ./data/vindr_subset --epochs 20
```

`train.py` lit par défaut `breast-level_annotations_subset.csv`, le CSV filtré
produit par le script de téléchargement. Pour le dataset complet, ajouter
`--csv-name breast-level_annotations.csv`.

GPU fortement recommandé. Sur CPU, réduire `--img-size` (ex. 224) et le nombre
d'études.

## 5bis. Valider la logique sans torch ni données

```bash
python tests/test_ordinal.py         # 24 assertions : losses, décodage, QWK
python src/prototype_numpy.py --n-seeds 5
```

Le prototype entraîne les deux approches sur des données synthétiques ordinales
en numpy pur et affiche QWK, MAE ordinal, matrices de confusion, distribution
des distances d'erreur et deltas appariés par seed. Résultats obtenus : voir
`RAPPORT_PROTOTYPE.md`.

## 6. Ce que CORN apporte par rapport à la baseline

- **Baseline (cross-entropy)** : traite A/B/C/D comme 4 classes indépendantes.
  Une confusion A↔D coûte la même chose qu'une confusion B↔C.
- **CORN** (Shi, Cao & Raschka, 2022) : reformule le problème en K−1 tâches
  binaires **conditionnelles** f_k = P(Y>k | Y≥k), chacune entraînée seulement
  sur le sous-ensemble {y ≥ k}. La probabilité non conditionnelle s'obtient par
  chaînage P(Y>k) = ∏_{j≤k} f_j, monotone décroissante par construction : la
  cohérence de rang est acquise sans imposer de biais ordonnés partagés entre
  tous les exemples, contrairement à CORAL.
- **CORAL** (Cao, Mirjalili & Raschka, 2020) : probabilités cumulées P(Y>k) avec
  un vecteur de poids partagé et K−1 biais. Monotonicité garantie par
  l'architecture, mais contrainte plus forte. Utile comme troisième point de
  comparaison — ne pas confondre les deux papiers, l'un est de 2020 (CORAL),
  l'autre de 2022 (CORN).
- **Métrique** : le QWK pénalise les erreurs proportionnellement au carré de la
  distance de rang, donc une confusion A↔D neuf fois plus qu'une confusion B↔C —
  ce qui correspond au coût clinique réel d'une erreur de densité.

## 7. Pièges techniques traités (à ne pas réintroduire)

**Décodage CORN** : il faut chaîner les probabilités conditionnelles *puis*
seuiller, et non l'inverse. Seuiller d'abord donne un décodeur non conforme qui
surestime le rang (≈15 % de prédictions différentes). Testé dans
`tests/test_ordinal.py` et assert dans le notebook.

**VOI LUT DICOM** : appliquer `apply_voi_lut` avant toute normalisation. Sans
cela les images sont plates et peu contrastées — or la densité BI-RADS *est* une
information de contraste.

**Rapport d'aspect** : une mammographie fait ~2800×3500 ; la redimensionner en
carré déforme la surface de tissu dense. Le loader fait du letterbox par défaut.

**Fuite de données** : le split validation se fait au niveau `study_id`, jamais
au niveau image (les 4 vues d'une patiente doivent rester du même côté), et la
standardisation est ajustée sur le train uniquement.
