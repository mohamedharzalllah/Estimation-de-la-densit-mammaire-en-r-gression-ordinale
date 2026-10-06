
# Régression ordinale pour la densité mammaire — résultats sur MIAS

**Date** : 26 juillet 2026
**Dataset** : MIAS v1.21, 322 images, 161 patientes, densité `BG` à 3 niveaux (F < G < D)
**Modèles** : ResNet18 pré-entraîné, tête cross-entropy (nominale) vs tête CORN (ordinale)
**Métrique** : Quadratic Weighted Kappa
**Matériel** : Kaggle, Tesla T4, ~3 s par epoch

---

## 1. Résultat principal

Validation croisée en 5 plis au niveau patiente, répétée sur 3 seeds. Chaque image
sert une fois au test, donc le QWK est estimé sur l'ensemble des 322 images.

| | QWK moyen | écart-type inter-pli |
|---|---|---|
| Cross-entropy (nominal) | 0,7875 | 0,061 |
| CORN (ordinal) | **0,7923** | 0,060 |

Comparaison appariée, le pli étant l'unité d'analyse indépendante (les 3 seeds
partagent les mêmes plis, les 15 mesures sont donc corrélées) :

| | valeur |
|---|---|
| Delta moyen CORN − baseline | **+0,0048** |
| IC 95 % (Student, n = 5, df = 4) | **[−0,0243 ; +0,0339]** |
| t observé | 0,46 (seuil 2,776) |
| Deltas par pli | −0,020 / −0,005 / +0,026 / +0,033 / −0,011 |
| CORN gagne | 8 comparaisons sur 15, 2 plis sur 5 |
| Test des signes | p = 1,00 |

**Aucune différence significative entre les deux approches.** L'intervalle de
confiance contient zéro et le delta moyen (+0,005) est six fois plus petit que la
demi-largeur de l'intervalle (0,029). Le résultat n'est ni une confirmation ni une
réfutation de l'hypothèse ordinale : il est **non concluant**.

Les deux modèles atteignent un QWK d'environ 0,79, ce qui est correct pour une
densité à 3 niveaux sur des mammographies numérisées à partir de films.

---

## 2. Le premier protocole donnait un résultat faux

Le protocole initial — un découpage unique train/val/test, 5 seeds faisant varier
seulement l'initialisation des poids — concluait à un delta de **−0,049 ± 0,055**,
soit CORN nettement moins bon, avec seulement 2 seeds gagnants sur 5.

| Protocole | Delta | IC 95 % | Test |
|---|---|---|---|
| Découpage unique, 5 seeds | −0,0488 | [−0,1022 ; +0,0045] | 80 images |
| Validation croisée 5×3 | +0,0048 | [−0,0243 ; +0,0339] | 322 images cumulées |

L'écart entre les deux conclusions vient de trois défauts du premier protocole.

**Le jeu de test était trop petit.** 80 images donnent un intervalle de confiance de
l'ordre de ±0,05, du même ordre que l'effet recherché. La validation croisée réduit
la largeur de l'intervalle d'un facteur 1,8.

**La variance mesurée n'était pas la bonne.** Faire varier le seed à découpage fixe
ne capture que la variance d'optimisation. Or la validation croisée montre que la
dispersion inter-pli (0,061) est **trois fois** la dispersion inter-seed (0,021) : ce
qui compte le plus, c'est quelles patientes tombent dans le test. Le pli 2 est
nettement plus difficile que les autres (QWK ≈ 0,69 contre 0,84 pour le pli 3), et un
découpage unique peut tomber n'importe où dans cette plage.

**La sélection du checkpoint amplifiait le bruit.** Prendre le maximum du QWK de
validation sur 25 epochs, avec 48 images de validation, revient à sélectionner du
bruit : les valeurs oscillaient entre 0,64 et 0,90 d'une epoch à l'autre. Le second
protocole sélectionne sur une moyenne glissante de 3 epochs.

Un quatrième défaut, corrigé aussi : le `WeightedRandomSampler` était actif alors que
les classes sont déjà équilibrées (66/64/64). Le tirage avec remise ne corrigeait rien
et ne présentait qu'environ 63 % d'images uniques par epoch, gaspillant un tiers des
données d'entraînement.

---

## 3. Deux erreurs dans le code d'analyse, corrigées

**Le verdict automatique ignorait le signe du delta.** La condition
`if delta.mean() <= delta.std()` affichait « on peut conclure que CORN est au moins
aussi bon et plus stable » même lorsque le delta était négatif. Avec un delta de
−0,049 le notebook affirmait donc l'inverse de ce que montraient ses propres chiffres.
Remplacé par un test qui compare les bornes de l'intervalle à zéro et nomme
explicitement le sens de l'écart.

**Le contrôle de luminosité était mal construit.** Il mesurait la moyenne sur toute
l'image, fond noir du letterbox inclus, et signalait un échec (`[59,9 ; 54,9 ; 44,9]
croissante : False`). Ce n'est pas une inversion des images : les seins denses
appartiennent à des femmes plus jeunes, donc sont plus petits et laissent plus de fond
noir, ce qui abaisse la moyenne globale. La mesure corrigée porte sur les pixels du
tissu uniquement. À noter qu'un CNN n'a de toute façon pas besoin qu'une statistique
globale simple soit monotone pour apprendre : ce contrôle sert seulement à détecter une
inversion grossière.

---

## 4. Pourquoi CORN n'apporte rien ici

L'explication la plus plausible tient au nombre de niveaux. Avec K = 3, CORN ne dispose
que de deux tâches binaires conditionnelles, et le conditionnement n'entraîne le seuil
k = 1 que sur le sous-ensemble {y ≥ 1}, soit environ deux tiers des données — donc
environ 145 images sur 220. La cross-entropy, elle, exploite tout l'échantillon pour
chacun de ses paramètres.

L'avantage structurel de CORN croît avec le nombre de niveaux ordinaux, puisque c'est
là que la contrainte de cohérence de rang devient informative. À 3 niveaux et 220
images d'entraînement, le coût en efficacité statistique compense le gain structurel.

C'est cohérent avec le prototype sur données synthétiques, où CORN gagnait +0,012 de
QWK sur 4 classes avec 1950 images d'entraînement — un effet déjà modeste, du même ordre
que sa dispersion.

---

## 5. Limites de MIAS

**Puissance statistique insuffisante.** Même en validation croisée, seul un effet
supérieur à environ 0,029 de QWK serait détectable. Un gain ordinal plus fin est hors
de portée de ce dataset, quelle que soit la qualité du protocole.

**Technologie obsolète.** Films numérisés, alors que la pratique clinique et
VinDr-Mammo utilisent le numérique plein champ. Bruit et contraste diffèrent, les
performances ne se transposent pas.

**Trois niveaux, pas quatre.** L'échelle MIAS n'est pas l'échelle BI-RADS.

**Ne pas présenter MIAS comme du BI-RADS.** Le miroir Kaggle fournit une colonne
`DENSITY` en A/B/C/D, mais elle est dérivée de `BG` par correspondance un-pour-un
(F→B, G→C, D→D) et la catégorie A est entièrement absente. Annoncer quatre classes
BI-RADS sur ces données serait inexact. C'est `BG` qui a été utilisé.

**Surapprentissage.** 220 images d'entraînement pour 11 millions de paramètres. La
loss d'entraînement descend sous 0,05 pendant que le QWK de validation stagne.

---

## 6. Ce que l'expérience établit

Le pipeline fonctionne de bout en bout sur de vraies mammographies. La loss CORN
s'entraîne correctement, le décodage du rang respecte la cohérence par construction
(chaînage des probabilités conditionnelles puis seuillage, conforme à Shi et al. 2022),
le QWK est vérifié sur ses cas limites, et le protocole de comparaison garantit une
parité stricte : même backbone, mêmes hyperparamètres, mêmes seeds, seules la tête de
sortie et la loss changent.

Le protocole d'évaluation est désormais correct : validation croisée au niveau
patiente, comparaisons appariées, intervalles de confiance tenant compte de la
structure de corrélation.

La question scientifique reste ouverte, et c'est une conclusion défendable plutôt
qu'un échec. Elle justifie de porter l'expérience sur VinDr-Mammo, où 4 niveaux
BI-RADS et 20 000 images annotées en double lecture avec arbitrage permettront de
trancher.

---

## 7. Leçon transférable à VinDr-Mammo

La variance dominante vient du choix des patientes de test, pas de l'initialisation
des poids — facteur trois entre les deux. Un protocole qui ne fait varier que le seed
sur un découpage fixe conclut à tort, comme démontré ici avec un delta apparent de
−0,049 qui s'est révélé être du bruit.

Sur VinDr-Mammo, il faudra donc une validation croisée au niveau étude, ou un jeu de
test assez grand pour que l'intervalle de confiance soit inférieur à l'effet attendu.
Le split officiel du dataset (4000 études d'entraînement, 1000 de test) offre environ
4000 images de test, ce qui devrait suffire — mais il faudra le vérifier plutôt que le
supposer.

---

## 8. Prochaines étapes, par ordre de rendement

**Passer à VinDr-Mammo.** Seul levier qui change l'ordre de grandeur. Le script
`scripts/fetch_and_cache_local.py` télécharge et convertit en cache NPY depuis une
machine locale, où le débit PhysioNet est généralement bien supérieur à celui de Colab
ou Kaggle. Un sous-ensemble de 400 études suffit pour démarrer.

**Ajouter CORAL en troisième point de comparaison.** Déjà implémenté dans
`src/models.py` et `src/losses.py`. Si les deux approches ordinales se comportent comme
la cross-entropy, cela confirme que la cause est le manque de niveaux et de données
plutôt qu'un défaut propre à CORN.

**Réduire le surapprentissage.** Geler les premiers blocs du ResNet18 et n'entraîner
que le dernier bloc plus la tête, ou choisir un backbone plus léger.

**Augmentation au moment du test.** Moyenner les prédictions de l'image et de son
miroir horizontal, pour un coût négligeable.

**Analyse secondaire en deux classes.** Non-dense (F) contre dense (G + D), la
dichotomie réellement utilisée en clinique pour décider d'un dépistage complémentaire.
Statistiquement plus stable et cliniquement interprétable.

---

## Références

Shi, X., Cao, W., & Raschka, S. (2022). Deep neural networks for rank-consistent
ordinal regression based on conditional probabilities. *ICPR 2022* / arXiv:2111.08851.
— **CORN**

Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for
neural networks with application to age estimation. *Pattern Recognition Letters*, 140,
325-331. — **CORAL**, à ne pas confondre avec CORN

Suckling, J., Parker, J., Dance, D., Astley, S., Hutt, I. et al. (2015). Mammographic
Image Analysis Society (MIAS) database v1.21. Apollo, University of Cambridge.
https://doi.org/10.17863/CAM.105113 — licence CC BY 2.0 UK

Pham, H. H., Nguyen Trung, H., & Nguyen, H. Q. (2022). VinDr-Mammo: A large-scale
benchmark dataset for computer-aided detection and diagnosis in full-field digital
mammography (version 1.0.0). *PhysioNet*. https://doi.org/10.13026/br2v-7517
