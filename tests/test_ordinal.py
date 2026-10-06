#!/usr/bin/env python3
"""Tests de correction du cœur ordinal : losses, décodage du rang, QWK.

Ne dépend que de numpy — tourne partout, y compris sans PyTorch.
Les tests de parité torch sont sautés si torch n'est pas installé ; ils
vérifient que l'implémentation PyTorch coïncide numériquement avec celle numpy.

Usage : python tests/test_ordinal.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from data import make_synthetic_breast_density  # noqa: E402
from metrics import (error_distance_histogram, ordinal_mean_abs_error,  # noqa: E402
                      quadratic_weighted_kappa)
from prototype_numpy import (ce_loss_and_grad, corn_loss_and_grad,  # noqa: E402
                              corn_predict, sigmoid)

K = 4
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        _failures.append(name)


# ---------------------------------------------------------------- QWK
def test_qwk():
    print("\n[QWK]")
    y = np.array([0, 1, 2, 3])
    check("accord parfait -> 1.0",
          abs(quadratic_weighted_kappa(y, y, K) - 1.0) < 1e-12)
    check("inversion totale -> -1.0",
          abs(quadratic_weighted_kappa(y, y[::-1], K) + 1.0) < 1e-12)

    # Une prédiction constante ne porte aucune information -> kappa nul
    yt = np.array([0, 1, 2, 3] * 10)
    check("prédiction constante -> 0.0",
          abs(quadratic_weighted_kappa(yt, np.zeros_like(yt), K)) < 1e-12)

    # Le QWK doit préférer les erreurs proches aux erreurs lointaines
    yt = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    near = np.array([0, 1, 1, 2, 2, 3, 3, 3])   # toutes les erreurs à distance 1
    far = np.array([3, 3, 2, 2, 1, 1, 0, 0])    # erreurs lointaines
    check("erreurs proches mieux notées que lointaines",
          quadratic_weighted_kappa(yt, near, K) > quadratic_weighted_kappa(yt, far, K))

    # Pondération quadratique : une erreur de distance 2 doit coûter ~4x
    # une erreur de distance 1 (et non 2x, ce que ferait un kappa linéaire)
    base = np.arange(K).repeat(25)
    p1 = base.copy(); p1[0] = base[0] + 1
    p2 = base.copy(); p2[0] = base[0] + 2
    d1 = 1.0 - quadratic_weighted_kappa(base, p1, K)
    d2 = 1.0 - quadratic_weighted_kappa(base, p2, K)
    check("pénalité quadratique (dist 2 coûte ~4x dist 1)",
          abs(d2 / d1 - 4.0) < 0.2, f"ratio={d2 / d1:.3f}")


# ------------------------------------------------- décodage du rang CORN
def test_corn_decoding():
    print("\n[Décodage CORN]")
    # Logits extrêmes : chaque ligne doit donner un rang déterminé
    lg = np.array([[9., 9., 9.], [9., 9., -9.], [9., -9., -9.], [-9., -9., -9.]])
    check("logits extrêmes -> rangs 3,2,1,0",
          np.array_equal(corn_predict(lg), np.array([3, 2, 1, 0])),
          f"obtenu {corn_predict(lg)}")

    # Conformité au papier : chaînage AVANT seuillage.
    # Cas discriminant [0.5, 0.5, 3.0] :
    #   P(Y>0)=0.622 > 0.5 ; P(Y>1)=0.622*0.622=0.387 < 0.5  -> rang 1
    # Le décodeur naïf (seuiller puis cumprod) donnerait 3.
    lg = np.array([[0.5, 0.5, 3.0]])
    check("chaînage avant seuillage (cas discriminant)",
          corn_predict(lg)[0] == 1, f"obtenu {corn_predict(lg)[0]}, attendu 1")

    rng = np.random.default_rng(0)
    lg = rng.normal(0, 3, size=(5000, K - 1))
    preds = corn_predict(lg)
    check("rang toujours dans [0, K-1]", bool(preds.min() >= 0 and preds.max() <= K - 1))

    # Cohérence de rang : les probabilités cumulées sont décroissantes
    cum = np.cumprod(sigmoid(lg), axis=1)
    check("probabilités cumulées monotones décroissantes",
          bool(np.all(np.diff(cum, axis=1) <= 1e-12)))

    # Monotonie : augmenter tous les logits ne peut pas faire baisser le rang
    check("monotonie en les logits",
          bool(np.all(corn_predict(lg + 1.0) >= corn_predict(lg))))


# ------------------------------------------------------ losses & gradients
def numerical_grad(fn, logits, y, eps=1e-6):
    num = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            lp = logits.copy(); lp[i, j] += eps
            lm = logits.copy(); lm[i, j] -= eps
            num[i, j] = (fn(lp, y)[0] - fn(lm, y)[0]) / (2 * eps)
    return num


def test_gradients():
    print("\n[Gradients (différences finies)]")
    rng = np.random.default_rng(0)
    for name, fn, n_out in [("cross-entropy", ce_loss_and_grad, K),
                             ("CORN", corn_loss_and_grad, K - 1)]:
        logits = rng.normal(size=(9, n_out))
        y = rng.integers(0, K, size=9)
        _, analytic = fn(logits, y)
        num = numerical_grad(fn, logits, y)
        err = float(np.abs(num - analytic).max())
        check(f"gradient {name} (erreur max {err:.2e})", err < 1e-7)


def test_corn_loss_semantics():
    print("\n[Sémantique de la loss CORN]")
    y = np.array([0, 1, 2, 3])
    perfect = np.array([[-20., -20., -20.], [20., -20., -20.],
                         [20., 20., -20.], [20., 20., 20.]])
    loss, _ = corn_loss_and_grad(perfect, y)
    check("prédiction parfaite -> loss ~0", loss < 1e-6, f"loss={loss:.2e}")

    loss_wrong, _ = corn_loss_and_grad(-perfect, y)
    check("prédiction inversée -> loss élevée", loss_wrong > 10, f"loss={loss_wrong:.1f}")

    # Le conditionnement : le seuil k ne voit que les exemples y >= k. Modifier
    # le logit du seuil 2 pour un exemple de classe 0 (hors S_2) ne doit RIEN
    # changer à la loss. C'est la propriété qui distingue CORN de CORAL.
    y0 = np.array([0, 3])
    lg = np.zeros((2, K - 1))
    base, _ = corn_loss_and_grad(lg, y0)
    lg2 = lg.copy(); lg2[0, 2] = 5.0
    modified, _ = corn_loss_and_grad(lg2, y0)
    check("conditionnement sur S_k respecté", abs(base - modified) < 1e-12,
          f"{base:.6f} vs {modified:.6f}")

    _, g = corn_loss_and_grad(lg, y0)
    check("gradient nul hors du sous-ensemble conditionnel",
          abs(g[0, 1]) < 1e-15 and abs(g[0, 2]) < 1e-15)


# ---------------------------------------------------- métriques ordinales
def test_ordinal_metrics():
    print("\n[Métriques ordinales]")
    yt = np.array([0, 1, 2, 3])
    check("MAE ordinal, accord parfait -> 0", ordinal_mean_abs_error(yt, yt) == 0.0)
    check("MAE ordinal, décalage borné -> 0.75",
          abs(ordinal_mean_abs_error(yt, np.clip(yt + 1, 0, 3)) - 0.75) < 1e-12)

    h = error_distance_histogram(np.array([0, 0, 0]), np.array([0, 1, 3]), K)
    check("histogramme des distances (clés str, JSON-safe)",
          h["0"] == 1 and h["1"] == 1 and h["3"] == 1 and h["n_errors"] == 2, str(h))


# ------------------------------------------------ données synthétiques
def test_synthetic_data():
    print("\n[Données synthétiques]")
    imgs, labels = make_synthetic_breast_density(n_samples=2000, img_size=16,
                                                  label_noise=0.25, seed=7)
    check("images dans [0,1]", bool(imgs.min() >= 0 and imgs.max() <= 1))
    check("labels dans [0,K-1]", bool(labels.min() >= 0 and labels.max() <= K - 1))
    check("les 4 classes sont présentes", len(np.unique(labels)) == K)

    # Le bruit demandé doit être effectivement appliqué : un clip naïf sur les
    # classes extrêmes en annulerait une partie (taux effectif ~18% au lieu de 25%).
    _, clean = make_synthetic_breast_density(n_samples=2000, img_size=16,
                                              label_noise=0.0, seed=7)
    rate = float((clean != labels).mean())
    check(f"taux de bruit effectif ~25% (mesuré {rate:.1%})", abs(rate - 0.25) < 0.02)

    # Le signal doit être monotone : luminosité moyenne croissante avec la densité
    means = [float(imgs[clean == k].mean()) for k in range(K)]
    check("luminosité moyenne croissante avec la densité",
          all(means[i] < means[i + 1] for i in range(K - 1)),
          f"{[round(m, 3) for m in means]}")


# -------------------------------------------------- parité numpy / torch
def test_torch_parity():
    print("\n[Parité numpy / PyTorch]")
    try:
        import torch
    except ImportError:
        print("  SKIP (torch non installé dans cet environnement)")
        return

    from losses import corn_loss as corn_loss_torch, cross_entropy_loss
    from models import CornClassifier

    rng = np.random.default_rng(3)
    lg = rng.normal(size=(32, K - 1))
    y = rng.integers(0, K, size=32)
    t_lg = torch.tensor(lg, dtype=torch.float64)
    t_y = torch.tensor(y, dtype=torch.long)

    np_loss, _ = corn_loss_and_grad(lg, y)
    th_loss = float(corn_loss_torch(t_lg, t_y, K))
    check(f"loss CORN identique (écart {abs(np_loss - th_loss):.2e})",
          abs(np_loss - th_loss) < 1e-10)

    np_pred = corn_predict(lg)
    th_pred = CornClassifier.predict_labels(t_lg).numpy()
    check("décodage CORN identique", np.array_equal(np_pred, th_pred))

    lg4 = rng.normal(size=(32, K))
    np_ce, _ = ce_loss_and_grad(lg4, y)
    th_ce = float(cross_entropy_loss(torch.tensor(lg4, dtype=torch.float64), t_y))
    check(f"loss cross-entropy identique (écart {abs(np_ce - th_ce):.2e})",
          abs(np_ce - th_ce) < 1e-10)


def main():
    print("=" * 66)
    print("TESTS — densité mammaire ordinale (losses, décodage, métriques)")
    print("=" * 66)
    test_qwk()
    test_corn_decoding()
    test_gradients()
    test_corn_loss_semantics()
    test_ordinal_metrics()
    test_synthetic_data()
    test_torch_parity()
    print("\n" + "=" * 66)
    if _failures:
        print(f"{len(_failures)} TEST(S) EN ÉCHEC : {', '.join(_failures)}")
        sys.exit(1)
    print("Tous les tests passent.")
    print("=" * 66)


if __name__ == "__main__":
    main()
