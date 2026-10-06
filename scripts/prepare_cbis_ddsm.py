#!/usr/bin/env python3
"""Convertit CBIS-DDSM (miroir Kaggle awsaf49) au format de cache du projet.

Pourquoi CBIS-DDSM : MIAS n'a que 322 images et 3 niveaux de densité, ce qui
s'est révélé insuffisant pour trancher entre cross-entropy et CORN (delta
+0,005, IC 95 % [-0,024 ; +0,034]). CBIS-DDSM offre une densité ACR sur
**4 niveaux** et environ dix fois plus d'images, sous licence CC BY 3.0.
C'est le meilleur terrain disponible tant que VinDr-Mammo reste inaccessible.

STRUCTURE ATTENDUE (miroir awsaf49) :
    <racine>/csv/mass_case_description_train_set.csv
             csv/mass_case_description_test_set.csv
             csv/calc_case_description_train_set.csv
             csv/calc_case_description_test_set.csv
             csv/dicom_info.csv            (facultatif, sert de repli)
             jpeg/<SeriesInstanceUID>/*.jpg

DEUX PIEGES CONNUS, tous deux gérés :

1. Le nom de la colonne de densité DIFFERE entre les fichiers : les jeux « mass »
   utilisent `breast_density`, les jeux « calc » utilisent `breast density`
   (avec une espace). La détection se fait donc par normalisation du nom.

2. Les CSV référencent des chemins DICOM (`image file path`) alors que le miroir
   ne contient que des JPEG rangés par SeriesInstanceUID. On extrait donc l'UID
   du chemin DICOM et on le rapproche du nom de dossier sous `jpeg/`. En repli,
   on utilise `dicom_info.csv`.

Densité ACR (colonne breast_density), ordinale croissante :
    1 -> 0 (A, presque entièrement graisseux)
    2 -> 1 (B)
    3 -> 2 (C)
    4 -> 3 (D, extrêmement dense)

On ne garde que les **mammographies complètes**, pas les vignettes recadrées ni
les masques de ROI : la densité est une propriété globale du sein.

Usage :
    # diagnostic, n'écrit rien
    python scripts/prepare_cbis_ddsm.py --src /kaggle/input/<slug> --inspect

    # conversion
    python scripts/prepare_cbis_ddsm.py --src /kaggle/input/<slug> \
        --out /kaggle/working/cbis_cache --img-size 512

Sortie, identique au cache MIAS et VinDr-Mammo :
    <out>/cache/<image_id>.npy   images img_size x img_size uint8
    <out>/labels.csv             image_id, patient_id, laterality, view,
                                  density_label, split
    <out>/dataset_info.json      n_classes, class_names, colonnes utilisées

Citation : Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., &
Rubin, D. L. (2017). A curated mammography data set for use in computer-aided
detection and diagnosis research. Scientific Data, 4, 170177.
Licence CC BY 3.0.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

CLASS_NAMES = ["A", "B", "C", "D"]      # ACR 1..4 renumérotés 0..3
N_CLASSES = 4
IMG_EXT = (".jpg", ".jpeg", ".png")
UID_RE = re.compile(r"\b(\d+(?:\.\d+){4,})\b")     # UID DICOM : 1.3.6.1.4...

# Noms de colonnes, apres normalisation (minuscules, separateurs -> espace)
DENSITY_KEYS = ("breast density", "breast_density", "density")
PATIENT_KEYS = ("patient id", "patient_id")
PATH_KEYS = ("image file path", "image_file_path")
LATERALITY_KEYS = ("left or right breast", "left_or_right_breast", "laterality")
VIEW_KEYS = ("image view", "image_view", "view")


def canon(s) -> str:
    return re.sub(r"[\s_]+", " ", str(s)).strip().lower()


def pick(cols: dict, keys) -> str | None:
    for k in keys:
        if k in cols:
            return cols[k]
    return None


def norm_density(v) -> int | None:
    """ACR 1..4 -> rang 0..3. Tolère aussi A/B/C/D et 0..3."""
    s = str(v).strip().upper()
    if s in {"1", "2", "3", "4"}:
        return int(s) - 1
    if s in {"1.0", "2.0", "3.0", "4.0"}:
        return int(float(s)) - 1
    if s in {"A", "B", "C", "D"}:
        return "ABCD".index(s)
    if s == "0":                      # quelques lignes ont 0 = densite manquante
        return None
    return None


# ------------------------------------------------------------------ inventaire
def index_jpeg(src: str) -> tuple[dict[str, str], int]:
    """Indexe les images par nom de dossier parent (= SeriesInstanceUID).

    Un dossier peut contenir plusieurs JPEG ; on garde le plus volumineux, qui
    correspond à la mammographie pleine résolution.
    """
    best: dict[str, tuple[int, str]] = {}
    n_files = 0
    for root, _dirs, files in os.walk(src):
        for name in files:
            if not name.lower().endswith(IMG_EXT):
                continue
            n_files += 1
            uid = os.path.basename(root)
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            if uid not in best or size > best[uid][0]:
                best[uid] = (size, path)
    return {k: v[1] for k, v in best.items()}, n_files


def find_case_csvs(src: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(src):
        for name in files:
            low = name.lower()
            if low.endswith(".csv") and "case_description" in low:
                out.append(os.path.join(root, name))
    if out:
        return sorted(out)
    # repli : tout CSV contenant une colonne de densite
    for root, _dirs, files in os.walk(src):
        for name in files:
            if name.lower().endswith(".csv"):
                out.append(os.path.join(root, name))
    return sorted(out)


def split_of(path: str) -> str:
    low = os.path.basename(path).lower()
    if "test" in low:
        return "test"
    if "train" in low:
        return "training"
    return "training"


def uids_in(path_str: str) -> list[str]:
    return UID_RE.findall(str(path_str))


def build_dicom_fallback(src: str) -> dict[str, str]:
    """Rapprochement de secours via dicom_info.csv : {SeriesInstanceUID: chemin}."""
    import pandas as pd
    for root, _dirs, files in os.walk(src):
        for name in files:
            if name.lower() != "dicom_info.csv":
                continue
            try:
                df = pd.read_csv(os.path.join(root, name), dtype=str)
            except Exception:
                return {}
            cols = {canon(c): c for c in df.columns}
            uid_c = pick(cols, ("seriesinstanceuid", "series instance uid"))
            pth_c = pick(cols, ("image path", "image_path"))
            des_c = pick(cols, ("seriesdescription", "series description"))
            if not (uid_c and pth_c):
                return {}
            if des_c:      # ne garder que les mammographies completes
                df = df[df[des_c].astype(str).str.contains("full", case=False, na=False)]
            return {str(u): str(p) for u, p in zip(df[uid_c], df[pth_c])}
    return {}


# ------------------------------------------------------------------- lecture
def load_cases(src: str, verbose=True):
    """Fusionne les CSV mass + calc, train + test, en un tableau unique.

    Une image apparaît plusieurs fois s'il y a plusieurs anomalies : on
    déduplique sur l'identifiant d'image.
    """
    import pandas as pd

    csvs = find_case_csvs(src)
    if verbose:
        print(f"[csv] {len(csvs)} fichiers de description trouvés")
    frames = []
    for path in csvs:
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as e:
            if verbose:
                print(f"[warn] illisible : {os.path.basename(path)} ({e})")
            continue
        cols = {canon(c): c for c in df.columns}
        dens_c = pick(cols, DENSITY_KEYS)
        path_c = pick(cols, PATH_KEYS)
        if not (dens_c and path_c):
            if verbose:
                print(f"[skip] {os.path.basename(path)} : "
                      f"densite={dens_c} chemin={path_c}")
            continue
        pat_c = pick(cols, PATIENT_KEYS)
        lat_c = pick(cols, LATERALITY_KEYS)
        vue_c = pick(cols, VIEW_KEYS)
        sub = pd.DataFrame({
            "density_raw": df[dens_c],
            "dicom_path": df[path_c],
            "patient_id": df[pat_c] if pat_c else "",
            "laterality": df[lat_c] if lat_c else "",
            "view": df[vue_c] if vue_c else "",
        })
        sub["split"] = split_of(path)
        sub["source"] = os.path.basename(path)
        frames.append(sub)
        if verbose:
            print(f"[csv] {os.path.basename(path):45s} {len(sub):5d} lignes | "
                  f"densite='{dens_c}' | split={split_of(path)}")
    if not frames:
        sys.exit("Aucun CSV exploitable. Lance --inspect et envoie la sortie.")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------- images
def load_image(path: str, img_size: int, keep_aspect: bool = True) -> np.ndarray:
    """JPEG CBIS-DDSM -> uint8 [img_size, img_size].

    Pas de VOI LUT : le miroir fournit des JPEG 8 bits déjà fenêtrés.
    Normalisation min-max puis letterbox — écraser une mammographie
    (~3000x5000) en carré déformerait la surface de tissu dense, qui est le
    signal recherché.
    """
    from PIL import Image

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    if keep_aspect:
        w, h = img.size
        s = img_size / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        canvas = Image.new("L", (img_size, img_size), color=0)
        canvas.paste(img, ((img_size - img.size[0]) // 2,
                            (img_size - img.size[1]) // 2))
        img = canvas
    else:
        img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def inspect(src: str):
    import pandas as pd
    print(f"\n=== DIAGNOSTIC de {src} ===")
    print("\nArborescence (2 niveaux) :")
    for entry in sorted(os.scandir(src), key=lambda e: e.name):
        if entry.is_dir():
            subs = sorted(os.listdir(entry.path))
            print(f"  {entry.name}/  ({len(subs)} entrées) ex. {subs[:3]}")
        else:
            print(f"  {entry.name}  {entry.stat().st_size/1e6:.1f} Mo")

    jpg, n_files = index_jpeg(src)
    print(f"\n{n_files} fichiers image, {len(jpg)} dossiers (UID) indexés")
    for k in list(jpg)[:3]:
        print(f"  {k[:60]} -> {os.path.basename(jpg[k])}")

    print("\nCSV de description :")
    for path in find_case_csvs(src):
        try:
            df = pd.read_csv(path, dtype=str, nrows=3)
        except Exception as e:
            print(f"  {os.path.basename(path)} : illisible ({e})")
            continue
        cols = {canon(c): c for c in df.columns}
        print(f"\n  {os.path.basename(path)}  ({len(df.columns)} colonnes)")
        print(f"    colonnes : {list(df.columns)}")
        print(f"    densite detectee : {pick(cols, DENSITY_KEYS)}")
        print(f"    chemin  detecte  : {pick(cols, PATH_KEYS)}")
        if len(df):
            pc = pick(cols, PATH_KEYS)
            if pc:
                print(f"    1er chemin : {str(df[pc].iloc[0])[:110]}")
    fb = build_dicom_fallback(src)
    print(f"\ndicom_info.csv : {len(fb)} entrées de repli")
    print("\nSi la conversion echoue, envoie-moi ce diagnostic complet.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="racine du dataset CBIS-DDSM")
    ap.add_argument("--out", default="/kaggle/working/cbis_cache")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--views", default="",
                     help="filtrer les vues, ex. 'CC' (divise le volume par 2 "
                          "sans perdre d'etiquette). Vide = toutes.")
    ap.add_argument("--limit", type=int, default=0,
                     help="limiter le nombre d'images (test rapide)")
    ap.add_argument("--no-keep-aspect", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    try:
        from PIL import Image  # noqa: F401
        import pandas as pd
    except ImportError as e:
        sys.exit(f"Dependance manquante ({e.name}). pip install pillow pandas numpy")

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        sys.exit(f"Dossier introuvable : {src}")

    if args.inspect:
        inspect(src)
        return

    jpg, n_files = index_jpeg(src)
    print(f"[scan] {n_files} fichiers image, {len(jpg)} dossiers UID")
    if not jpg:
        inspect(src)
        sys.exit("Aucune image trouvee.")

    cases = load_cases(src)
    print(f"[csv] {len(cases)} lignes au total")

    cases["density_label"] = cases["density_raw"].map(norm_density)
    n_before = len(cases)
    cases = cases.dropna(subset=["density_label"]).copy()
    cases["density_label"] = cases["density_label"].astype(int)
    if len(cases) < n_before:
        print(f"[warn] {n_before - len(cases)} lignes sans densite exploitable")

    if args.views:
        keep = [v.strip().upper() for v in args.views.split(",")]
        before = len(cases)
        cases = cases[cases["view"].astype(str).str.upper().isin(keep)]
        print(f"[filtre] vues {keep} : {len(cases)}/{before} lignes")

    # Rapprochement CSV (chemins DICOM) <-> JPEG (dossiers par UID)
    fallback = build_dicom_fallback(src)
    def resolve(dicom_path):
        for uid in reversed(uids_in(dicom_path)):    # le dernier UID = la serie
            if uid in jpg:
                return uid, jpg[uid]
        for uid in uids_in(dicom_path):
            if uid in fallback:
                cand = os.path.join(src, fallback[uid].lstrip("/"))
                if os.path.exists(cand):
                    return uid, cand
                base = os.path.basename(fallback[uid])
                for u2, p2 in jpg.items():
                    if os.path.basename(p2) == base:
                        return u2, p2
        return None, None

    res = cases["dicom_path"].map(resolve)
    cases["image_id"] = [r[0] for r in res]
    cases["file"] = [r[1] for r in res]
    n_nores = int(cases["image_id"].isna().sum())
    cases = cases.dropna(subset=["image_id", "file"])
    print(f"[match] {len(cases)} lignes rapprochees d'un JPEG"
          + (f" ({n_nores} sans correspondance)" if n_nores else ""))
    if cases.empty:
        inspect(src)
        sys.exit("Aucun rapprochement CSV <-> JPEG. Envoie le diagnostic.")

    # Une image peut porter plusieurs anomalies -> une seule ligne par image.
    # En cas de densites contradictoires on garde la plus elevee (prudence).
    cases = (cases.sort_values("density_label", ascending=False)
                   .drop_duplicates(subset=["image_id"], keep="first")
                   .reset_index(drop=True))
    print(f"[dedup] {len(cases)} images uniques")

    if args.limit:
        cases = cases.groupby("density_label", group_keys=False).apply(
            lambda g: g.head(max(1, args.limit // N_CLASSES))).reset_index(drop=True)
        print(f"[limit] reduit a {len(cases)} images")

    # patient_id : indispensable pour eviter la fuite entre splits (une patiente
    # a jusqu'a 4 images : gauche/droite x CC/MLO)
    cases["patient_id"] = cases["patient_id"].fillna("").astype(str)
    manquants = cases["patient_id"].str.strip() == ""
    if manquants.any():
        cases.loc[manquants, "patient_id"] = cases.loc[manquants, "image_id"]
        print(f"[warn] {int(manquants.sum())} lignes sans patient_id : "
              f"chaque image compte comme une patiente distincte")

    # Le split officiel CBIS-DDSM est deja separe par patient : on le respecte,
    # mais on verifie qu'aucune patiente ne figure des deux cotes.
    croisees = (set(cases[cases.split == "training"]["patient_id"])
                & set(cases[cases.split == "test"]["patient_id"]))
    if croisees:
        print(f"[ATTENTION] {len(croisees)} patientes presentes dans train ET test : "
              f"elles sont retirees du test pour eviter la fuite")
        cases = cases[~((cases.split == "test")
                        & (cases.patient_id.isin(croisees)))]

    out_dir = os.path.abspath(args.out)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    rows, echecs = [], []
    for n, r in enumerate(cases.itertuples(), 1):
        out_npy = os.path.join(cache_dir, f"{r.image_id}.npy")
        if not os.path.exists(out_npy):
            try:
                arr = load_image(r.file, args.img_size, not args.no_keep_aspect)
            except Exception as e:
                echecs.append(f"{r.image_id}: {type(e).__name__}: {e}")
                continue
            tmp = out_npy + ".part"
            with open(tmp, "wb") as fh:      # np.save renommerait un str sans .npy
                np.save(fh, arr)
            os.replace(tmp, out_npy)         # ecriture atomique -> reprise sure
        rows.append({"image_id": r.image_id, "patient_id": r.patient_id,
                      "laterality": r.laterality, "view": r.view,
                      "density_label": r.density_label, "split": r.split})
        if n % 200 == 0:
            print(f"  converti {n}/{len(cases)}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    json.dump({"dataset": "CBIS-DDSM", "n_classes": N_CLASSES,
               "class_names": CLASS_NAMES, "scale": "acr",
               "label_column": "breast_density", "label_file": "case_description CSV",
               "img_size": args.img_size, "keep_aspect": not args.no_keep_aspect,
               "n_images": len(df), "n_patients": df["patient_id"].nunique(),
               "authentic_labels": True,
               "note": "Densite ACR 1-4 annotee, renumerotee 0-3 (A/B/C/D). "
                        "Split officiel CBIS-DDSM (train/test par patiente)."},
              open(os.path.join(out_dir, "dataset_info.json"), "w"), indent=2)

    print(f"\n[fin] {len(df)} images | {df['patient_id'].nunique()} patientes "
          f"-> {cache_dir}")
    if echecs:
        print(f"[warn] {len(echecs)} echecs de conversion :")
        for e in echecs[:5]:
            print("   " + e)

    print(f"\nDistribution (lignes = split, colonnes = {CLASS_NAMES}) :")
    print(df.groupby(["split", "density_label"]).size().unstack(fill_value=0).to_string())
    for split_name, g in df.groupby("split"):
        c = g["density_label"].value_counts()
        absentes = [k for k in range(N_CLASSES) if k not in c.index]
        faibles = {int(k): int(v) for k, v in c.items() if v < 30}
        if absentes:
            print(f"[ATTENTION] {split_name}: classes absentes {absentes}")
        if faibles:
            print(f"[warn] {split_name}: classes sous 30 images {faibles} "
                  f"-> QWK instable pour ces classes")

    mb = sum(e.stat().st_size for e in os.scandir(cache_dir)) / 1e6
    print(f"\n[cache] {mb:.0f} Mo")
    print("\n[suite] Dans le notebook d'entrainement : ROOT = "
          f"'{out_dir}'  -> N_CLASSES=4 lu depuis dataset_info.json")


if __name__ == "__main__":
    main()
