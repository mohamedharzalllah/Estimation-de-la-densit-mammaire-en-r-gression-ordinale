#!/usr/bin/env python3
"""Entraînement d'un classifieur de densité mammaire BI-RADS A/B/C/D.

Trois modes strictement comparables (même backbone, même optimiseur, mêmes
hyperparamètres, même seed — SEULES la tête de sortie et la loss changent) :

    --model baseline   ResNet18 + softmax 4 classes + cross-entropy    (nominal)
    --model corn       ResNet18 + 3 probas conditionnelles + CORN      (ordinal)
    --model coral      ResNet18 + poids partagé + 3 seuils + CORAL     (ordinal)

Exemples (GPU recommandé) :
    python src/train.py --model baseline --data-root ./data/vindr_subset --epochs 20
    python src/train.py --model corn     --data-root ./data/vindr_subset --epochs 20

Le CSV lu par défaut est `breast-level_annotations_subset.csv`, celui produit
par scripts/download_vindr_subset.py. Pour le dataset complet, passer
`--csv-name breast-level_annotations.csv`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import N_CLASSES, CLASS_NAMES, VinDrMammoDataset, load_vindr_annotations  # noqa: E402
from losses import LossWrapper  # noqa: E402
from metrics import summarize  # noqa: E402
from models import build_model  # noqa: E402


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int, train: bool):
    import torchvision.transforms as T
    # Normalisation ImageNet (le backbone est pré-entraîné dessus)
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(),   # gauche/droite : sein L vs R, augmentation valide
            T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            norm,
        ])
    return norm


def make_loaders(args):
    csv_name = args.csv_name
    df_train = load_vindr_annotations(args.data_root, csv_name, split="training")
    df_test = load_vindr_annotations(args.data_root, csv_name, split="test")

    # Split validation interne au training, AU NIVEAU ÉTUDE (pas image) pour
    # éviter toute fuite : les 4 vues d'une patiente restent du même côté.
    rng = np.random.default_rng(args.seed)
    studies = df_train["study_id"].unique()
    rng.shuffle(studies)
    n_val = max(1, int(len(studies) * args.val_frac))
    val_studies = set(studies[:n_val])
    df_val = df_train[df_train["study_id"].isin(val_studies)].reset_index(drop=True)
    df_tr = df_train[~df_train["study_id"].isin(val_studies)].reset_index(drop=True)

    ds_tr = VinDrMammoDataset(args.data_root, img_size=args.img_size,
                               transform=build_transforms(args.img_size, True),
                               dataframe=df_tr)
    ds_val = VinDrMammoDataset(args.data_root, img_size=args.img_size,
                                transform=build_transforms(args.img_size, False),
                                dataframe=df_val)
    ds_te = VinDrMammoDataset(args.data_root, img_size=args.img_size,
                               transform=build_transforms(args.img_size, False),
                               dataframe=df_test)

    print(f"[data] train={len(ds_tr)} val={len(ds_val)} test={len(ds_te)} images")
    print(f"[data] distribution train: {ds_tr.class_counts()}")

    sampler = None
    shuffle = True
    if args.balance_classes:
        # VinDr-Mammo est déséquilibré (B et C dominent, A et D rares).
        # Rééchantillonnage inversement proportionnel à la fréquence.
        labels = df_tr["density_label"].to_numpy()
        counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
        weights = 1.0 / np.maximum(counts[labels], 1)
        sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                         num_samples=len(labels), replacement=True)
        shuffle = False

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                   pin_memory=torch.cuda.is_available())
    return (
        DataLoader(ds_tr, shuffle=shuffle, sampler=sampler, **common),
        DataLoader(ds_val, shuffle=False, **common),
        DataLoader(ds_te, shuffle=False, **common),
    )


@torch.no_grad()
def evaluate(model, loader, device, predict_fn):
    model.eval()
    all_true, all_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = predict_fn(logits.float().cpu())
        all_true.append(y.numpy())
        all_pred.append(preds.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return summarize(y_true, y_pred, N_CLASSES, CLASS_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "corn", "coral"], required=True)
    ap.add_argument("--data-root", required=True)
    # Par défaut le CSV filtré produit par scripts/download_vindr_subset.py :
    # utiliser le CSV complet alors que seul un sous-ensemble d'images est sur
    # le disque provoque un FileNotFoundError au premier batch.
    ap.add_argument("--csv-name", default="breast-level_annotations_subset.csv")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    # Rééquilibrage activé par défaut mais désactivable (pour l'ablation) :
    # `action="store_true"` + `default=True` rendrait le flag inopérant.
    ap.add_argument("--no-balance-classes", dest="balance_classes",
                     action="store_false", help="désactive le WeightedRandomSampler")
    ap.set_defaults(balance_classes=True)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out-dir", default="./results")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} model={args.model} backbone={args.backbone}")

    train_loader, val_loader, test_loader = make_loaders(args)

    model = build_model(args.model, N_CLASSES, args.backbone,
                         pretrained=not args.no_pretrained).to(device)
    predict_fn = type(model).predict_labels

    loss_kind = {"baseline": "ce", "corn": "corn", "coral": "coral"}[args.model]
    criterion = LossWrapper(loss_kind, N_CLASSES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    best_qwk = -1.0
    history = []
    ckpt_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, predict_fn)
        entry = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_qwk": val_metrics["qwk"],
            "val_ordinal_mae": val_metrics["ordinal_mae"],
            "seconds": time.time() - t0,
        }
        history.append(entry)
        print(f"[ep {epoch:3d}] loss={entry['train_loss']:.4f} "
              f"val_acc={entry['val_accuracy']:.4f} val_QWK={entry['val_qwk']:.4f} "
              f"val_MAE={entry['val_ordinal_mae']:.4f} ({entry['seconds']:.0f}s)")

        # Sélection du modèle sur le QWK (métrique ordinale), pas sur l'accuracy
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, predict_fn)
    print(f"\n[TEST {args.model}] acc={test_metrics['accuracy']:.4f} "
          f"QWK={test_metrics['qwk']:.4f} ordinal_MAE={test_metrics['ordinal_mae']:.4f}")
    print("Matrice de confusion (lignes=vérité A,B,C,D):")
    for name, row in zip(CLASS_NAMES, test_metrics["confusion_matrix"]):
        print(f"  {name}: {row}")

    out = {"args": vars(args), "history": history, "test_metrics": test_metrics,
           "best_val_qwk": best_qwk}
    out_path = os.path.join(args.out_dir, f"metrics_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] métriques -> {out_path}")


if __name__ == "__main__":
    main()
