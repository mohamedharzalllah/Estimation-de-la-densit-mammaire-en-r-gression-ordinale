#!/usr/bin/env python3
"""Prototype numpy pur : baseline cross-entropy vs CORN ordinal.

But : valider toute la logique (loss ordinale, décodage du rang, QWK,
analyse des distances d'erreur) SANS dépendre de PyTorch, sur des données
synthétiques dont la structure imite la densité mammaire ordinale.

Le modèle est volontairement simple et STRICTEMENT identique entre les deux
approches : MLP à une couche cachée sur images 32x32 aplaties, même
initialisation (même seed), même optimiseur, mêmes hyperparamètres.
Seules diffèrent la dimension de sortie et la fonction de perte :

    baseline : 4 sorties, softmax + cross-entropy      (classes nominales)
    corn     : 3 sorties, K-1 BCE conditionnelles      (structure ordinale)

Référence CORN : Shi, Cao & Raschka (2022), arXiv:2111.08851 (à ne pas
confondre avec CORAL, Cao, Mirjalili & Raschka 2020).

C'est exactement le protocole de comparaison qui sera appliqué au ResNet18
sur VinDr-Mammo (voir src/train.py) : on ne change QUE la tête et la loss.

Usage : python src/prototype_numpy.py [--epochs 60] [--out-dir results]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import (CLASS_NAMES, N_CLASSES, make_synthetic_breast_density,  # noqa: E402
                   train_val_test_split)
from metrics import summarize  # noqa: E402


# --------------------------------------------------------------------------
# Modèle : MLP 1 couche cachée, forward/backward en numpy
# --------------------------------------------------------------------------
class MLP:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He init pour la couche ReLU
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, x):
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._h = np.maximum(self._z1, 0.0)
        return self._h @ self.W2 + self.b2  # logits

    def backward(self, dlogits):
        gW2 = self._h.T @ dlogits
        gb2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dz1 = dh * (self._z1 > 0)
        gW1 = self._x.T @ dz1
        gb1 = dz1.sum(axis=0)
        return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}


class AdamOptimizer:
    def __init__(self, params: dict, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            if self.wd and k.startswith("W"):
                g = g + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ce_loss_and_grad(logits, y, n_classes=N_CLASSES):
    """Cross-entropy classique : aucune notion d'ordre entre A/B/C/D."""
    p = softmax(logits)
    n = len(y)
    loss = -np.mean(np.log(np.maximum(p[np.arange(n), y], 1e-12)))
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    dlogits = (p - onehot) / n
    return loss, dlogits


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def corn_loss_and_grad(logits, y, n_classes=N_CLASSES):
    """CORN (Shi, Cao & Raschka, 2022) : K-1 tâches binaires CONDITIONNELLES.

    Pour chaque seuil k, on n'entraîne que sur le sous-ensemble
    S_k = {i : y_i >= k}, avec cible binaire 1[y_i > k]. Le conditionnement
    est ce qui rend les probabilités cohérentes en rang.
    """
    p = sigmoid(logits)                       # [N, K-1]
    n_thresh = n_classes - 1
    mask = np.zeros_like(logits)              # 1 si l'exemple compte pour ce seuil
    target = np.zeros_like(logits)
    for k in range(n_thresh):
        in_subset = (y >= k)
        mask[:, k] = in_subset.astype(float)
        target[:, k] = (y > k).astype(float)

    total_terms = max(mask.sum(), 1.0)
    bce = -(target * np.log(np.maximum(p, 1e-12))
            + (1 - target) * np.log(np.maximum(1 - p, 1e-12)))
    loss = float((bce * mask).sum() / total_terms)
    dlogits = mask * (p - target) / total_terms
    return loss, dlogits


def ce_predict(logits):
    return logits.argmax(axis=1)


def corn_predict(logits):
    """Décodage du rang CORN, conforme à Shi, Cao & Raschka (2022).

    ORDRE CRITIQUE : chaîner d'abord les probabilités conditionnelles pour
    obtenir P(Y > k) = prod_{j<=k} sigmoid(logit_j), PUIS seuiller à 0,5.
    Seuiller avant de chaîner donne un décodeur différent, non conforme au
    papier, qui surestime le rang.
    """
    cum_probs = np.cumprod(sigmoid(logits), axis=1)   # P(Y>0), P(Y>1), ...
    return (cum_probs > 0.5).sum(axis=1)


# --------------------------------------------------------------------------
# Boucle d'entraînement commune
# --------------------------------------------------------------------------
def train_one(kind: str, X_tr, y_tr, X_val, y_val, X_te, y_te,
              epochs=60, batch_size=32, lr=1e-3, hidden=64, seed=0, verbose=True):
    assert kind in {"baseline", "corn"}
    n_out = N_CLASSES if kind == "baseline" else N_CLASSES - 1
    loss_fn = ce_loss_and_grad if kind == "baseline" else corn_loss_and_grad
    predict = ce_predict if kind == "baseline" else corn_predict

    model = MLP(X_tr.shape[1], hidden, n_out, seed=seed)
    opt = AdamOptimizer(model.params(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    best_qwk, best_state, history = -np.inf, None, []
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(len(X_tr))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            logits = model.forward(X_tr[idx])
            loss, dlogits = loss_fn(logits, y_tr[idx])
            grads = model.backward(dlogits)
            opt.step(model.params(), grads)
            epoch_loss += loss
            n_batches += 1

        val_pred = predict(model.forward(X_val))
        m = summarize(y_val, val_pred, N_CLASSES, CLASS_NAMES)
        history.append({"epoch": epoch, "train_loss": epoch_loss / n_batches,
                         "val_accuracy": m["accuracy"], "val_qwk": m["qwk"],
                         "val_ordinal_mae": m["ordinal_mae"]})
        # Sélection sur le QWK (métrique ordinale), comme dans le pipeline torch
        if m["qwk"] > best_qwk:
            best_qwk = m["qwk"]
            best_state = {k: v.copy() for k, v in model.params().items()}
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  [{kind} ep {epoch:3d}] loss={epoch_loss / n_batches:.4f} "
                  f"val_acc={m['accuracy']:.3f} val_QWK={m['qwk']:.3f}")

    for k, v in best_state.items():
        model.params()[k][...] = v
    test_pred = predict(model.forward(X_te))
    test_metrics = summarize(y_te, test_pred, N_CLASSES, CLASS_NAMES)
    return {"kind": kind, "history": history, "best_val_qwk": best_qwk,
            "test_metrics": test_metrics}


def print_confusion(cm, class_names):
    header = "        " + "  ".join(f"{c:>4}" for c in class_names)
    print(header + "     (colonnes = prédit)")
    for name, row in zip(class_names, cm):
        print(f"  vrai {name}  " + "  ".join(f"{v:>4}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--img-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--label-noise", type=float, default=0.25)
    ap.add_argument("--n-seeds", type=int, default=5,
                     help="nombre de répétitions pour estimer la variabilité")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 74)
    print("PROTOTYPE — densité mammaire ordinale : cross-entropy vs CORN")
    print("=" * 74)
    print(f"Données synthétiques : {args.n_samples} images {args.img_size}x{args.img_size}, "
          f"bruit de label={args.label_noise:.0%}")

    all_runs = {"baseline": [], "corn": []}
    for seed in range(args.n_seeds):
        print(f"--- seed {seed} " + "-" * 55)
        # Chaque seed régénère LES DONNÉES et LE SPLIT, pas seulement
        # l'initialisation des poids. Sinon le "± écart-type" ne mesure que la
        # variance d'optimisation sur un unique jeu de test figé, et non
        # l'incertitude de généralisation.
        images, labels = make_synthetic_breast_density(
            n_samples=args.n_samples, img_size=args.img_size,
            label_noise=args.label_noise, seed=1000 + seed)
        X_raw = images.reshape(len(images), -1)
        tr, va, te = train_val_test_split(len(X_raw), val_frac=0.15,
                                           test_frac=0.20, seed=1000 + seed)

        # Standardisation ajustée SUR LE TRAIN UNIQUEMENT (sinon fuite des
        # statistiques de val/test dans le prétraitement)
        mu = X_raw[tr].mean(axis=0, keepdims=True)
        sd = X_raw[tr].std(axis=0, keepdims=True) + 1e-6
        X = (X_raw - mu) / sd

        X_tr, y_tr = X[tr], labels[tr]
        X_val, y_val = X[va], labels[va]
        X_te, y_te = X[te], labels[te]
        if seed == 0:
            print(f"Split : train={len(tr)} val={len(va)} test={len(te)}")
            print(f"Distribution train A/B/C/D : {np.bincount(y_tr, minlength=4).tolist()}")

        for kind in ("baseline", "corn"):
            res = train_one(kind, X_tr, y_tr, X_val, y_val, X_te, y_te,
                             epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                             seed=seed, verbose=(seed == 0))
            all_runs[kind].append(res)
            tm = res["test_metrics"]
            print(f"  -> {kind:8s} TEST acc={tm['accuracy']:.4f} "
                  f"QWK={tm['qwk']:.4f} ordinal_MAE={tm['ordinal_mae']:.4f}")

    print("\n" + "=" * 74)
    print(f"RÉSULTATS AGRÉGÉS SUR {args.n_seeds} SEEDS (moyenne ± écart-type, jeu de test)")
    print("=" * 74)
    summary = {}
    for kind in ("baseline", "corn"):
        accs = np.array([r["test_metrics"]["accuracy"] for r in all_runs[kind]])
        qwks = np.array([r["test_metrics"]["qwk"] for r in all_runs[kind]])
        maes = np.array([r["test_metrics"]["ordinal_mae"] for r in all_runs[kind]])
        # Erreurs "lointaines" = |rang vrai - rang prédit| >= 2 (ex. A confondu avec C/D)
        far, near, tot_err = [], [], []
        for r in all_runs[kind]:
            h = r["test_metrics"]["error_distance_histogram"]
            n_err = h["n_errors"]
            n_far = sum(h[str(d)] for d in range(2, N_CLASSES))
            far.append(n_far / max(n_err, 1))
            near.append(h["1"] / max(n_err, 1))
            tot_err.append(n_err)
        summary[kind] = {
            "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
            "qwk_mean": float(qwks.mean()), "qwk_std": float(qwks.std()),
            "ordinal_mae_mean": float(maes.mean()), "ordinal_mae_std": float(maes.std()),
            "frac_errors_adjacent_mean": float(np.mean(near)),
            "frac_errors_distant_mean": float(np.mean(far)),
            "n_errors_mean": float(np.mean(tot_err)),
        }
        s = summary[kind]
        print(f"\n{kind.upper()}")
        print(f"  Accuracy      : {s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f}")
        print(f"  QWK           : {s['qwk_mean']:.4f} ± {s['qwk_std']:.4f}   <-- métrique ordinale")
        print(f"  Ordinal MAE   : {s['ordinal_mae_mean']:.4f} ± {s['ordinal_mae_std']:.4f}")
        print(f"  Erreurs adjacentes (dist=1, ex. B<->C) : {s['frac_errors_adjacent_mean']:.1%} des erreurs")
        print(f"  Erreurs lointaines (dist>=2, ex. A<->D): {s['frac_errors_distant_mean']:.1%} des erreurs")

    print("\n" + "-" * 74)
    print("Matrices de confusion (seed 0, jeu de test)")
    for kind in ("baseline", "corn"):
        print(f"\n{kind.upper()} :")
        print_confusion(all_runs[kind][0]["test_metrics"]["confusion_matrix"], CLASS_NAMES)

    # Comparaison APPARIÉE seed par seed : chaque seed partage les mêmes
    # données et le même split entre les deux modèles, donc les deltas sont
    # appariés et c'est leur distribution qui compte, pas celle des scores bruts.
    qwk_b = np.array([r["test_metrics"]["qwk"] for r in all_runs["baseline"]])
    qwk_c = np.array([r["test_metrics"]["qwk"] for r in all_runs["corn"]])
    mae_b = np.array([r["test_metrics"]["ordinal_mae"] for r in all_runs["baseline"]])
    mae_c = np.array([r["test_metrics"]["ordinal_mae"] for r in all_runs["corn"]])
    d_qwk_per_seed = qwk_c - qwk_b
    d_mae_per_seed = mae_c - mae_b
    n_wins = int((d_qwk_per_seed > 0).sum())
    d_far = (summary["corn"]["frac_errors_distant_mean"]
             - summary["baseline"]["frac_errors_distant_mean"])

    print("\n" + "=" * 74)
    print("DELTA CORN - BASELINE (comparaison appariée, seed par seed)")
    print(f"  QWK        : {d_qwk_per_seed.mean():+.4f} ± {d_qwk_per_seed.std():.4f}  "
          f"(positif = CORN meilleur)")
    print(f"               deltas par seed : "
          f"{', '.join(f'{d:+.4f}' for d in d_qwk_per_seed)}")
    print(f"               CORN gagne sur {n_wins}/{args.n_seeds} seeds")
    print(f"  Ordinal MAE: {d_mae_per_seed.mean():+.4f} ± {d_mae_per_seed.std():.4f}  "
          f"(négatif = CORN meilleur)")
    print(f"  Part d'erreurs lointaines : {d_far:+.2%}  "
          f"(négatif = CORN se trompe 'moins loin')")
    if args.n_seeds < 5:
        print("  ATTENTION : moins de 5 seeds, ces écarts ne sont pas interprétables.")
    print("=" * 74)
    summary["paired_delta"] = {
        "qwk_delta_per_seed": d_qwk_per_seed.tolist(),
        "qwk_delta_mean": float(d_qwk_per_seed.mean()),
        "qwk_delta_std": float(d_qwk_per_seed.std()),
        "corn_wins": n_wins, "n_seeds": args.n_seeds,
        "ordinal_mae_delta_mean": float(d_mae_per_seed.mean()),
        "frac_distant_errors_delta": float(d_far),
    }

    out = {"config": vars(args), "summary": summary,
           "runs": {k: [{"test_metrics": r["test_metrics"], "best_val_qwk": r["best_val_qwk"]}
                         for r in v] for k, v in all_runs.items()},
           "history_seed0": {k: v[0]["history"] for k, v in all_runs.items()}}
    out_path = os.path.join(args.out_dir, "prototype_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] métriques -> {out_path}")


if __name__ == "__main__":
    main()
