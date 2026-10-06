"""Métriques pour la classification ordinale de densité mammaire (A/B/C/D).

Implémentation pure numpy (pas de dépendance à scikit-learn) pour que ce
module tourne aussi dans des environnements minimalistes. Utilisable aussi
bien par le pipeline numpy de prototype que par le pipeline PyTorch réel.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Matrice de confusion K x K, lignes = vérité, colonnes = prédiction."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Quadratic Weighted Kappa (Cohen's kappa avec pondération quadratique).

    QWK = 1 - sum(W * O) / sum(W * E)
    où O = matrice de confusion observée, E = matrice attendue sous
    indépendance (produit des marginales), W_ij = (i-j)^2 / (K-1)^2.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return 0.0

    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n

    numerator = np.sum(weights * cm)
    denominator = np.sum(weights * expected)
    if denominator == 0:
        return 1.0
    return 1.0 - numerator / denominator


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def ordinal_mean_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Distance moyenne en rangs entre vérité et prédiction (0 = accord parfait)."""
    return float(np.mean(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64))))


def error_distance_histogram(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Distribution des erreurs par distance de rang |y_true - y_pred|.

    Sert à vérifier l'hypothèse : un modèle ordinal-aware doit concentrer ses
    erreurs sur les distances courtes (1 = classe voisine, ex B/C) plutôt que
    longues (n_classes-1 = classes opposées, ex A/D).
    """
    dist = np.abs(y_true.astype(np.int64) - y_pred.astype(np.int64))
    # Clés en str : json.dump convertit de toute façon les clés entières en
    # chaînes, autant être cohérent pour que relire le JSON ne casse pas.
    hist = {str(d): int(np.sum(dist == d)) for d in range(n_classes)}
    hist["n_errors"] = int(np.sum(dist > 0))
    hist["n_total"] = int(len(y_true))
    return hist


def summarize(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, class_names=None) -> dict:
    if class_names is None:
        class_names = [chr(ord("A") + i) for i in range(n_classes)]
    return {
        "accuracy": accuracy(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, n_classes),
        "ordinal_mae": ordinal_mean_abs_error(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes).tolist(),
        "error_distance_histogram": error_distance_histogram(y_true, y_pred, n_classes),
        "class_names": class_names,
    }
