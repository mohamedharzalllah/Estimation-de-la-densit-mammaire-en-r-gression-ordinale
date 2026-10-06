#!/usr/bin/env python3
"""Télécharge un SOUS-ENSEMBLE stratifié de VinDr-Mammo depuis PhysioNet.

Pourquoi : le dataset complet fait ~50 Go (5000 études x 4 vues DICOM). Pour
un prototype de classification de densité, 300-600 études stratifiées sur
A/B/C/D suffisent (~3-6 Go).

Étape 1 : récupérer seulement les CSV d'annotations (quelques Mo)
Étape 2 : choisir un sous-ensemble stratifié d'études
Étape 3 : télécharger uniquement les dossiers images/<study_id>/ retenus

Prérequis : compte PhysioNet credentialed + DUA VinDr-Mammo signée.

Usage :
    # étape 1+2+3 en une fois
    python scripts/download_vindr_subset.py \
        --user TON_LOGIN_PHYSIONET \
        --out ./data/vindr_subset \
        --n-train-studies 400 --n-test-studies 100

Le mot de passe est demandé de façon interactive (jamais stocké en clair).
"""
from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys

BASE = "https://physionet.org/files/vindr-mammo/1.0.0"
CSV_FILES = ["breast-level_annotations.csv", "finding_annotations.csv", "metadata.csv"]


def download_csvs(out_dir: str, user: str, password: str):
    os.makedirs(out_dir, exist_ok=True)
    for name in CSV_FILES:
        target = os.path.join(out_dir, name)
        if os.path.exists(target):
            print(f"[skip] {name} déjà présent")
            continue
        print(f"[csv] téléchargement de {name}")
        rc = subprocess.run([
            "wget", "--quiet", "--show-progress",
            "--user", user, "--password", password,
            "-O", target, f"{BASE}/{name}",
        ]).returncode
        if rc != 0 or os.path.getsize(target) == 0:
            os.path.exists(target) and os.remove(target)
            sys.exit(f"Échec du téléchargement de {name} — vérifie identifiants et DUA signée.")


def pick_studies(csv_path: str, n_train: int, n_test: int, seed: int = 42):
    """Sélection stratifiée d'études par catégorie de densité.

    On stratifie au niveau ÉTUDE (pas image) pour éviter toute fuite entre
    train et test : les 4 vues d'une même patiente restent du même côté.
    La densité d'une étude est prise comme la densité modale de ses images.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "breast_density" not in df.columns:
        sys.exit(f"Colonne breast_density absente de {csv_path}")

    study_density = (df.groupby(["split", "study_id"])["breast_density"]
                       .agg(lambda s: s.mode().iloc[0])
                       .reset_index())

    selected = {}
    for split_name, n_target in (("training", n_train), ("test", n_test)):
        sub = study_density[study_density["split"] == split_name]
        if len(sub) == 0:
            sys.exit(f"Aucune étude pour split={split_name}")
        n_target = min(n_target, len(sub))
        # répartition proportionnelle à la distribution réelle des densités
        frac = n_target / len(sub)
        picked = (sub.groupby("breast_density", group_keys=False)
                    .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))),
                                              random_state=seed)))
        # Mélanger AVANT de tronquer : sans cela, head() rognerait
        # systématiquement le dernier groupe trié (la densité D, la plus rare),
        # ce qui biaiserait la stratification.
        picked = picked.sample(frac=1.0, random_state=seed).head(n_target)
        selected[split_name] = picked
        print(f"[select] {split_name}: {len(picked)} études")
        print(picked["breast_density"].value_counts().sort_index().to_string())
    return selected


def download_studies(study_ids, out_dir: str, user: str, password: str):
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    total = len(study_ids)
    for i, sid in enumerate(study_ids, 1):
        dest = os.path.join(images_dir, sid)
        if os.path.isdir(dest) and os.listdir(dest):
            continue
        print(f"[{i}/{total}] étude {sid}")
        subprocess.run([
            "wget", "--quiet", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=5",
            "--user", user, "--password", password,
            "-P", dest, f"{BASE}/images/{sid}/",
        ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="identifiant PhysioNet")
    ap.add_argument("--out", default="./data/vindr_subset")
    ap.add_argument("--n-train-studies", type=int, default=400)
    ap.add_argument("--n-test-studies", type=int, default=100)
    ap.add_argument("--csv-only", action="store_true",
                     help="ne télécharger que les CSV (rapide, pour inspecter les labels)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    password = os.environ.get("PHYSIONET_PASSWORD") or getpass.getpass("Mot de passe PhysioNet: ")

    download_csvs(args.out, args.user, password)
    if args.csv_only:
        print("CSV téléchargés. Relance sans --csv-only pour les images.")
        return

    csv_path = os.path.join(args.out, "breast-level_annotations.csv")
    selected = pick_studies(csv_path, args.n_train_studies, args.n_test_studies, args.seed)

    all_ids = []
    for split_name, picked in selected.items():
        all_ids.extend(picked["study_id"].astype(str).tolist())
    print(f"\n[download] {len(all_ids)} études (~{len(all_ids) * 4} images DICOM, "
          f"estimé {len(all_ids) * 4 * 2.5 / 1024:.1f} Go)")
    download_studies(all_ids, args.out, args.user, password)

    # CSV filtré sur les images RÉELLEMENT présentes sur le disque : filtrer
    # sur les study_id ne suffit pas, un téléchargement partiel laisserait des
    # lignes pointant vers des fichiers absents -> FileNotFoundError au 1er batch.
    import pandas as pd
    df = pd.read_csv(csv_path)
    img_col = "image_id" if "image_id" in df.columns else None
    if img_col is None:
        sys.exit("Colonne 'image_id' absente du CSV : impossible de vérifier les fichiers.")

    df_sub = df[df["study_id"].astype(str).isin(set(all_ids))].copy()
    exists = df_sub.apply(
        lambda r: os.path.exists(os.path.join(images_dir_of(args.out),
                                              str(r["study_id"]),
                                              f"{r[img_col]}.dicom")), axis=1)
    n_missing = int((~exists).sum())
    df_sub = df_sub[exists]
    sub_path = os.path.join(args.out, "breast-level_annotations_subset.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"[done] annotations du sous-ensemble : {sub_path} ({len(df_sub)} images)")
    if n_missing:
        print(f"[warn] {n_missing} lignes écartées (fichier DICOM absent) — "
              f"relance le script pour compléter le téléchargement.")
    print(f"[next] python src/train.py --model baseline --data-root {args.out}")


def images_dir_of(out_dir: str) -> str:
    return os.path.join(out_dir, "images")


if __name__ == "__main__":
    main()
