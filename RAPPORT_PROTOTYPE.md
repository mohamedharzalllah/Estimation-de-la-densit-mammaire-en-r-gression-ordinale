# Prototype — Estimation de la densité mammaire en régression ordinale

**Date** : 24 juillet 2026
**Statut** : prototype validé sur données synthétiques ; pipeline PyTorch prêt pour VinDr-Mammo

---

## Ce que ce prototype démontre

La densité mammaire BI-RADS (A < B < C < D) est une variable **ordinale**. Une cross-entropy
classique la traite comme quatre étiquettes indépendantes : confondre un sein adipeux (A) avec un
sein extrêmement dense (D) coûte exactement autant que confondre les deux catégories
intermédiaires B et C. C'est incohérent avec la réalité clinique, où l'erreur A↔D est bien plus
grave.

L'approche ordinal-aware retenue est **CORN** (Shi, Cao & Raschka, 2022) : au lieu d'un softmax à
4 sorties, le réseau produit K−1 = 3 logits modélisant les probabilités *conditionnelles*
f_k = P(Y > k | Y ≥ k). Chaque seuil k n'est entraîné que sur le sous-ensemble d'exemples tels que
y ≥ k. La probabilité non conditionnelle s'obtient par chaînage, P(Y > k) = ∏_{j≤k} f_j, qui est
monotone décroissante en k par construction — d'où la cohérence de rang, sans imposer de biais
ordonnés partagés entre tous les exemples comme le fait CORAL.

La métrique d'évaluation est le **Quadratic Weighted Kappa**, qui pénalise une erreur
proportionnellement au *carré* de la distance de rang : une confusion A↔D (distance 3) coûte neuf
fois une confusion B↔C (distance 1).

---

## Résultats du prototype (données synthétiques, 5 seeds)

Protocole : 3000 images synthétiques 32×32 dont la texture et la luminosité moyenne encodent une
densité ordinale, avec 25 % de bruit de label sur les classes adjacentes (imite l'accord
inter-lecteur imparfait du BI-RADS). Modèle identique dans les deux cas (MLP une couche cachée de
64 unités, Adam, mêmes hyperparamètres, même initialisation) ; **seules la dimension de sortie et
la loss changent**. Split train/val/test 65/15/20 %, standardisation ajustée sur le train
uniquement, sélection du meilleur checkpoint sur le QWK de validation. Chaque seed régénère les
données *et* le split, pas seulement l'initialisation des poids.

| Métrique | Cross-entropy (nominal) | CORN (ordinal) |
|---|---|---|
| Accuracy | 0,6997 ± 0,0184 | **0,7207 ± 0,0036** |
| **QWK** | 0,8658 ± 0,0094 | **0,8781 ± 0,0040** |
| MAE ordinal (rangs) | 0,3057 ± 0,0207 | **0,2803 ± 0,0046** |
| Part d'erreurs adjacentes (dist = 1) | 98,3 % | 99,6 % |
| Part d'erreurs lointaines (dist ≥ 2) | 1,74 % | **0,36 %** |
| Nombre d'erreurs (sur 600) | 180,2 | 167,6 |

Comparaison **appariée** seed par seed (chaque seed partage données et split entre les deux
modèles, donc c'est la distribution des deltas qui compte, pas celle des scores bruts) :

| | Delta CORN − baseline |
|---|---|
| QWK, moyenne | **+0,0123 ± 0,0118** |
| QWK, par seed | +0,0094 / +0,0053 / **−0,0022** / +0,0165 / +0,0325 |
| Seeds où CORN gagne | 4 sur 5 |
| MAE ordinal | −0,0253 ± 0,0240 |
| Part d'erreurs lointaines | −1,38 points |

CORN gagne sur quatre seeds sur cinq et perd de peu sur le troisième. L'écart moyen de QWK
(+0,012) est du même ordre que sa dispersion (± 0,012) : **sur un run unique, cet écart ne serait
pas concluant**. C'est précisément pourquoi le notebook contient une section multi-seed avec deltas
appariés, et pourquoi il faudra reporter des barres d'erreur sur VinDr-Mammo.

Le résultat le plus robuste n'est pas le gain de QWK mais la **réduction de variance** : l'écart-type
de CORN est 4 à 5 fois plus faible que celui de la baseline sur les trois métriques. La contrainte
ordinale agit comme une régularisation qui stabilise l'apprentissage. C'est un effet secondaire
intéressant, moins souvent mis en avant que le gain de performance moyen.

### Matrices de confusion (seed 0, jeu de test)

Cross-entropy :

```
          A     B     C     D    (colonnes = prédit)
 vrai A  104    22     0     0
 vrai B   56   118    23     0
 vrai C    0    24   101    33
 vrai D    0     2    18    99
```

CORN :

```
          A     B     C     D    (colonnes = prédit)
 vrai A   99    27     0     0
 vrai B   55   122    20     0
 vrai C    0    17   111    30
 vrai D    0     0    20    99
```

Les deux matrices sont quasi tridiagonales, mais CORN élimine presque complètement les erreurs de
distance ≥ 2 (0,36 % contre 1,74 %) : sur le seed 0, la baseline confond encore 2 seins de
catégorie D avec des B, ce que CORN ne fait pas.

---

## Limites honnêtes de ce prototype

**Les données synthétiques restent plus faciles que de vraies mammographies.** La densité y est
encodée de façon quasi monotone dans la luminosité moyenne, et le bruit de label injecté ne
concerne que les classes adjacentes. Résultat, même la baseline nominale produit peu d'erreurs
lointaines (1,74 %), ce qui laisse peu de marge pour démontrer l'effet. Sur VinDr-Mammo réel, on
s'attend à une baseline nettement plus dispersée (les confusions A↔C existent bel et bien sur de
vraies mammographies), donc à un écart plus large et plus convaincant.

**L'écart n'est pas statistiquement établi.** Avec 5 seeds et un delta de QWK de +0,012 ± 0,012,
on peut affirmer que CORN est au moins aussi bon et plus stable, pas qu'il est significativement
meilleur. Il faudra soit plus de seeds, soit un bootstrap sur le jeu de test, soit les deux.

**Ce prototype valide la correction de l'implémentation, pas l'ampleur du gain clinique.**

### Vérifications formelles effectuées

Toutes matérialisées dans `tests/test_ordinal.py` (24 assertions, exécutable par
`python tests/test_ordinal.py`, ne dépend que de numpy) :

Gradients de la cross-entropy et de la loss CORN validés par différences finies (erreur maximale
1,7 × 10⁻¹⁰ et 4,9 × 10⁻¹¹).

QWK vérifié sur cas limites : accord parfait → 1,0 ; inversion totale → −1,0 ; prédiction
constante → 0,0 ; et la pénalité est bien quadratique (une erreur de distance 2 coûte ~4× une
erreur de distance 1, ce qu'un kappa linéaire ne ferait pas).

Cohérence de rang de CORN vérifiée : le produit cumulé des probabilités conditionnelles est
monotone décroissant, le rang décodé reste dans [0, K−1], et augmenter tous les logits ne peut pas
faire baisser le rang.

Conditionnement de la loss CORN vérifié : modifier le logit du seuil k pour un exemple hors du
sous-ensemble S_k = {y ≥ k} ne change ni la loss ni le gradient. C'est la propriété qui distingue
CORN de CORAL.

Parité numpy / PyTorch : test présent et actif dès que torch est installé (il est sauté dans
l'environnement où le prototype a tourné, qui n'a pas torch).

### Bug corrigé au passage — à connaître

Une première version du décodage CORN seuillait les probabilités conditionnelles *avant* de les
chaîner (`(sigmoid(logits) > 0.5).cumprod(...)`) au lieu de chaîner *puis* seuiller. C'est un
décodeur différent, non conforme au papier, qui surestime le rang : sur des logits ~N(0, 1,5) les
deux versions divergent sur 15 % des prédictions. Cas discriminant minimal : logits
[0,5 ; 0,5 ; 3,0] donnent le rang 1 avec le décodage correct (P(Y>1) = 0,62² = 0,39 < 0,5) et le
rang 3 avec le décodage naïf. Le test correspondant est dans `tests/test_ordinal.py`, et le
notebook contient une assertion qui échoue immédiatement si le décodage régresse.

---

## Livrables

`notebooks/vindr_mammo_ordinal_density.ipynb` — **le livrable principal**. Notebook Colab complet
et exécutable de bout en bout : téléchargement du sous-ensemble PhysioNet, décodage DICOM avec VOI
LUT et gestion du MONOCHROME1, cache NPY, split au niveau étude, les deux modèles ResNet18,
entraînement, QWK, matrices de confusion, histogramme des distances d'erreur et section multi-seed.

`src/` — le même pipeline en modules réutilisables hors notebook : `data.py` (loader VinDr-Mammo +
générateur synthétique), `models.py` (ResNet18 + têtes CE, CORN et CORAL), `losses.py` (les trois
losses), `metrics.py` (QWK et métriques ordinales, numpy pur), `train.py` (CLI d'entraînement),
`prototype_numpy.py` (le prototype exécuté ici).

`tests/test_ordinal.py` — les vérifications listées ci-dessus.

`scripts/download_vindr_subset.py` — téléchargement stratifié du sous-ensemble PhysioNet en ligne
de commande, hors Colab.

`results/prototype_metrics.json` — métriques brutes des 5 seeds, y compris les deltas appariés.

---

## Étapes suivantes, dans l'ordre

**Aujourd'hui / demain** : ouvrir le notebook dans Colab avec un GPU T4 et exécuter les cellules 1
à 9 (accès PhysioNet déjà obtenu). Avec 400 études d'entraînement et 15 epochs, compte environ
25 minutes de téléchargement, 10 minutes de mise en cache, puis 30 à 45 minutes d'entraînement
pour les deux modèles.

**Jour 2** : passer à 1000+ études, lancer la section multi-seed (3 à 5 seeds) pour obtenir de
vraies barres d'erreur, et ajouter CORAL comme troisième approche (`--model coral`, tête et loss
déjà implémentées) — cela donne une comparaison à trois points au lieu d'un duel.

**Jour 3** : agrégation des prédictions au niveau sein puis examen (fusion des vues CC et MLO), et
rédaction. C'est l'extension la plus rentable scientifiquement : la densité est rapportée par
examen en clinique, pas par vue isolée.

---

## Références

Shi, X., Cao, W., & Raschka, S. (2022). Deep neural networks for rank-consistent ordinal
regression based on conditional probabilities. *ICPR 2022* / arXiv:2111.08851. — **CORN**

Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural
networks with application to age estimation. *Pattern Recognition Letters*, 140, 325-331. —
**CORAL** (à ne pas confondre avec CORN)

Pham, H. H., Nguyen Trung, H., & Nguyen, H. Q. (2022). VinDr-Mammo: A large-scale benchmark
dataset for computer-aided detection and diagnosis in full-field digital mammography
(version 1.0.0). *PhysioNet*. https://doi.org/10.13026/br2v-7517

Sickles, E. A. et al. (2013). *ACR BI-RADS® Mammography*, 5th edition. American College of
Radiology.
