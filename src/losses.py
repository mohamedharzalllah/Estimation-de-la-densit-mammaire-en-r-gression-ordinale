"""Loss functions : cross-entropy (baseline nominal) vs CORN / CORAL (ordinal).

CORN : Shi, X., Cao, W., & Raschka, S. (2022). "Deep Neural Networks for
Rank-Consistent Ordinal Regression Based on Conditional Probabilities."
ICPR 2022 / arXiv:2111.08851.

CORAL : Cao, W., Mirjalili, V., & Raschka, S. (2020). "Rank consistent ordinal
regression for neural networks with application to age estimation."
Pattern Recognition Letters, 140, 325-331.

Attention à ne pas confondre les deux références : CORAL est le papier de 2020
(probabilités cumulées, poids partagés + biais ordonnés), CORN celui de 2022
(probabilités conditionnelles, pas de contrainte de poids partagés).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Baseline : cross-entropy sur K classes indépendantes.

    Aucune information d'ordre : confondre A avec D est puni exactement comme
    confondre B avec C.
    """
    return F.cross_entropy(logits, targets)


def corn_loss(logits: torch.Tensor, targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Loss CORN sur K-1 tâches binaires conditionnelles.

    logits : [B, K-1], logit k modélise P(Y > k | Y >= k)
    targets: [B] entiers dans [0, K-1]

    Pour chaque seuil k = 0..K-2 :
      - on ne garde que les exemples du sous-ensemble conditionnel S_k = {i : y_i >= k}
      - la cible binaire est 1[y_i > k]
      - on somme les log-vraisemblances binaires sur ce sous-ensemble

    C'est ce conditionnement qui garantit la cohérence de rang :
    P(Y>0) >= P(Y>1) >= ... par construction (produit de probabilités
    conditionnelles), sans avoir besoin d'imposer des biais ordonnés
    partagés entre tous les exemples comme le fait CORAL.
    """
    losses = []
    for k in range(n_classes - 1):
        subset_mask = targets >= k
        if subset_mask.sum() == 0:
            continue
        logit_k = logits[subset_mask, k]
        label_k = (targets[subset_mask] > k).float()
        losses.append(F.binary_cross_entropy_with_logits(
            logit_k, label_k, reduction="sum"))
    if not losses:
        return logits.sum() * 0.0
    # Normalisation par le nombre total de termes conditionnels (moyenne globale)
    total_terms = sum((targets >= k).sum().item() for k in range(n_classes - 1))
    return torch.stack(losses).sum() / max(total_terms, 1)


def coral_loss(logits: torch.Tensor, targets: torch.Tensor, n_classes: int,
               importance_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Loss CORAL (Cao et al. 2020), pour comparer une troisième approche.

    CORAL entraîne K-1 classifieurs binaires cumulés P(Y > k) sur TOUS les
    exemples (pas de conditionnement), la monotonicité étant assurée par
    l'architecture : poids partagés + biais distincts.

    Important : cette loss n'a de sens qu'avec `models.CoralClassifier`
    (poids partagé). L'utiliser avec une tête nn.Linear ordinaire à K-1
    sorties indépendantes ne garantit PAS la cohérence de rang et ce ne serait
    pas CORAL.
    logits : [B, K-1]
    """
    # Cible cumulée : levels[i, k] = 1 si y_i > k
    levels = torch.zeros_like(logits)
    for k in range(n_classes - 1):
        levels[:, k] = (targets > k).float()

    logprobs = F.logsigmoid(logits)
    log1m = F.logsigmoid(logits) - logits  # log(1 - sigmoid(x)) stable
    term = logprobs * levels + log1m * (1 - levels)
    if importance_weights is not None:
        term = term * importance_weights
    return -term.sum(dim=1).mean()


class LossWrapper(nn.Module):
    """Interface unifiée pour la boucle d'entraînement."""

    def __init__(self, kind: str, n_classes: int = 4):
        super().__init__()
        assert kind in {"ce", "corn", "coral"}
        self.kind = kind
        self.n_classes = n_classes

    def forward(self, logits, targets):
        if self.kind == "ce":
            return cross_entropy_loss(logits, targets)
        if self.kind == "corn":
            return corn_loss(logits, targets, self.n_classes)
        return coral_loss(logits, targets, self.n_classes)
