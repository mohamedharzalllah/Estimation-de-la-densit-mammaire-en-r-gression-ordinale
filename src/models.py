"""Modèles PyTorch pour la classification de densité mammaire (à utiliser sur
la vraie chaîne VinDr-Mammo, GPU recommandé). Import de torch en tête de
fichier : ce module n'est à importer que dans un environnement avec
torch/torchvision installés (voir requirements.txt).

Pour valider la logique sans torch ni GPU, voir src/prototype_numpy.py, qui
réimplémente les mêmes losses et le même décodage en numpy pur.

Trois têtes disponibles :
  baseline : softmax K classes + cross-entropy         (nominal)
  corn     : K-1 probabilités conditionnelles + CORN   (ordinal, Shi et al. 2022)
  coral    : K-1 logits cumulés à poids partagés       (ordinal, Cao et al. 2020)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _make_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        net = tv_models.resnet18(weights=weights)
        n_features = net.fc.in_features
        net.fc = nn.Identity()
        return net, n_features
    raise ValueError(f"Backbone inconnu: {name}")


class BaselineClassifier(nn.Module):
    """Baseline nominal : backbone + tête softmax K classes, entraîné en CE.

    Traite A/B/C/D comme 4 classes indépendantes (aucune notion d'ordre).
    """

    def __init__(self, n_classes: int = 4, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone, n_features = _make_backbone(backbone, pretrained)
        self.head = nn.Linear(n_features, n_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)  # logits [B, K]

    @staticmethod
    def predict_labels(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=1)


class CornClassifier(nn.Module):
    """Tête CORN (Shi, Cao & Raschka, 2022 — "Deep Neural Networks for
    Rank-Consistent Ordinal Regression Based on Conditional Probabilities",
    ICPR 2022 / arXiv:2111.08851).

    Au lieu d'un softmax K-classes, on prédit K-1 logits pour les tâches
    binaires conditionnelles f_k = P(Y > k | Y >= k), k = 0..K-2. La loss CORN
    (voir losses.py) entraîne chaque tâche uniquement sur le sous-ensemble
    d'exemples pour lesquels Y >= k. La probabilité non conditionnelle
    s'obtient par la règle de chaînage P(Y > k) = prod_{j<=k} f_j, qui est
    monotone décroissante en k par construction — d'où la cohérence de rang,
    obtenue sans imposer de biais ordonnés partagés entre tous les exemples
    comme le fait CORAL.
    """

    def __init__(self, n_classes: int = 4, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.backbone, n_features = _make_backbone(backbone, pretrained)
        self.head = nn.Linear(n_features, n_classes - 1)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)  # logits conditionnels [B, K-1]

    @staticmethod
    def predict_labels(logits: torch.Tensor) -> torch.Tensor:
        """Convertit les logits conditionnels en label de rang final.

        ORDRE DES OPÉRATIONS CRITIQUE : on chaîne d'abord les probabilités
        conditionnelles (cumprod des sigmoïdes) pour obtenir les probabilités
        non conditionnelles P(Y > k), PUIS on seuille à 0,5 et on somme.

            rank = sum_k 1[ prod_{j<=k} sigmoid(logit_j) > 0.5 ]

        Seuiller avant de chaîner (`(probs > 0.5).cumprod(...)`) donne un
        décodeur DIFFÉRENT et non conforme au papier : il surestime le rang
        (par ex. logits [0.5, 0.5, 3.0] -> 3 au lieu de 1).
        """
        probs = torch.sigmoid(logits).cumprod(dim=1)  # P(Y>0), P(Y>1), ...
        return (probs > 0.5).sum(dim=1).long()


class CoralClassifier(nn.Module):
    """Tête CORAL (Cao, Mirjalili & Raschka, 2020 — "Rank consistent ordinal
    regression for neural networks with application to age estimation",
    Pattern Recognition Letters 140:325-331).

    CORAL modélise directement les probabilités cumulées P(Y > k) avec un
    vecteur de poids PARTAGÉ et K-1 biais distincts. Le partage des poids est
    ce qui garantit la monotonicité — c'est une contrainte plus forte que
    celle de CORN, d'où l'intérêt de comparer les deux.
    """

    def __init__(self, n_classes: int = 4, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.backbone, n_features = _make_backbone(backbone, pretrained)
        # poids partagé (une seule direction dans l'espace de features)
        self.shared = nn.Linear(n_features, 1, bias=False)
        # K-1 biais indépendants = les seuils
        self.thresholds = nn.Parameter(torch.zeros(n_classes - 1))

    def forward(self, x):
        feats = self.backbone(x)
        score = self.shared(feats)                 # [B, 1]
        return score + self.thresholds             # [B, K-1] par broadcast

    @staticmethod
    def predict_labels(logits: torch.Tensor) -> torch.Tensor:
        # Les logits sont déjà cumulés et monotones par construction
        return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()


def build_model(model_name: str, n_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True) -> nn.Module:
    if model_name == "baseline":
        return BaselineClassifier(n_classes, backbone, pretrained)
    if model_name == "corn":
        return CornClassifier(n_classes, backbone, pretrained)
    if model_name == "coral":
        return CoralClassifier(n_classes, backbone, pretrained)
    raise ValueError(f"Modèle inconnu: {model_name}")
