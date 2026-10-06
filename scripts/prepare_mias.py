#!/usr/bin/env python3
"""Convertit le dataset MIAS au format de cache du projet : cache/*.npy + labels.csv.

Pourquoi : le débit PhysioNet plafonne à ~86 Ko/s pour des DICOM VinDr-Mammo de
19,7 Mo, soit ~76 h pour un sous-ensemble utilisable. MIAS est petit et sous
licence libre, ce qui permet de valider toute la chaîne sur de vraies
mammographies immédiatement.

=============================================================================
QUELLE COLONNE D'ÉTIQUETTE UTILISER — point méthodologique important
=============================================================================
Le miroir Kaggle (orvile/mias-dataset) fournit deux fichiers :

  mias_info.csv          annotations ORIGINALES du MIAS
                         MIAS No, BG, CLASS, SEVERITY, x, y, RADIUS
  mias_derived_info.csv  colonnes AJOUTÉES par l'auteur du miroir
                         DENSITY, BI-RADS, CLASS_FULL, CLASS_GROUP

`BG` (background tissue) est la vraie variable de densité MIAS, ordinale à
3 niveaux :  F = fatty  <  G = fatty-glandular  <  D = dense-glandular

`DENSITY` (A/B/C/D) est DÉRIVÉE de BG par l'auteur du miroir. Or on ne peut pas
reconstruire 4 catégories BI-RADS à partir de 3 catégories MIAS : c'est un
recodage, pas une annotation de radiologue. MIAS n'a jamais contenu
d'annotation BI-RADS de densité. La colonne `BI-RADS` du même fichier est
également dérivée (de CLASS/SEVERITY), et ne mesure pas la densité mais la
suspicion de malignité.

=> Par défaut ce script utilise BG (3 classes authentiques).
=> `--scale birads` permet d'utiliser DENSITY, mais affiche un avertissement et
   le tableau croisé BG x DENSITY, qui montre qu'aucune information nouvelle
   n'est apportée. À n'utiliser que pour une vérification, pas pour un résultat
   publiable.

Le script écrit <out>/dataset_info.json avec n_classes et class_names, que le
notebook lit automatiquement : aucune constante à modifier à la main.
=============================================================================

Usage :
    # diagnostic seul, n'écrit rien
    python scripts/prepare_mias.py --src /kaggle/input/mias-dataset --inspect

    # conversion (BG, 3 classes -- recommandé)
    python scripts/prepare_mias.py --src /kaggle/input/mias-dataset \
        --out ./mias_cache --img-size 512

    # forcer une colonne précise
    python scripts/prepare_mias.py --src ... --out ... --label-col BG

Sortie, identique au cache VinDr-Mammo de fetch_and_cache_local.py :
    <out>/cache/<image_id>.npy   images img_size x img_size uint8
    <out>/labels.csv             image_id, patient_id, density_label, split
    <out>/dataset_info.json      n_classes, class_names, colonne utilisée

Sources et citation :
  Kaggle (miroir)   : https://www.kaggle.com/datasets/orvile/mias-dataset
  Source officielle : https://www.repository.cam.ac.uk/handle/1810/250394
  DOI               : https://doi.org/10.17863/CAM.105113
  Licence           : CC BY 2.0 UK (usage et redistribution libres, avec
                      attribution — aucune DUA, contrairement à VinDr-Mammo)
  Citation : Suckling, J., Parker, J., Dance, D., Astley, S., Hutt, I. et al.
  (2015). Mammographic Image Analysis Society (MIAS) database v1.21. Apollo,
  University of Cambridge. https://doi.org/10.17863/CAM.105113
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

# --------------------------------------------------------------- échelles
# BG : la vraie densité MIAS, 3 niveaux ordinaux croissants
BG_ALIASES: dict[str, int] = {
    "F": 0, "FATTY": 0,
    "G": 1, "FATTY-GLANDULAR": 1, "FATTY GLANDULAR": 1, "FATTYGLANDULAR": 1,
    "FATTY_GLANDULAR": 1, "GLANDULAR": 1,
    "D": 2, "DENSE": 2, "DENSE-GLANDULAR": 2, "DENSE GLANDULAR": 2,
    "DENSEGLANDULAR": 2, "DENSE_GLANDULAR": 2,
}
# DENSITY dérivée du miroir Kaggle, présentée comme du BI-RADS A/B/C/D
BIRADS_ALIASES: dict[str, int] = {
    "A": 0, "B": 1, "C": 2, "D": 3,
    "DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3,
}

SCALES = {
    "bg":     {"aliases": BG_ALIASES,     "names": ["F", "G", "D"],      "n": 3},
    "birads": {"aliases": BIRADS_ALIASES, "names": ["A", "B", "C", "D"], "n": 4},
}

# Noms de colonnes attendus, par ordre de préférence
BG_COL_HINTS = ("BG", "BACKGROUND", "BACKGROUND_TISSUE", "BACKGROUND TISSUE",
                "TISSUE", "BGTISSUE")
BIRADS_COL_HINTS = ("DENSITY", "BREAST_DENSITY", "BREAST DENSITY")
ID_COL_HINTS = ("MIAS NO", "MIAS_NO", "MIASNO", "REFNUM", "REF", "IMAGE_ID",
                "IMAGE", "ID", "FILENAME", "NAME")

IMG_EXT = (".pgm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
REF_RE = re.compile(r"(mdb\d{1,4})", re.I)
REF_RE_NOGROUP = re.compile(r"mdb\d{1,4}", re.I)
NUM_RE = re.compile(r"^\s*(\d{1,4})\s*$")


def canon(name) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().upper()


def norm_with(aliases: dict[str, int], v) -> int | None:
    if v is None:
        return None
    s = canon(v)
    if not s or s in {"NAN", "NONE", "-", ""}:
        return None
    return aliases.get(s)


def norm_id(v) -> str | None:
    """'mdb001', 'MDB001.pgm', '1' -> 'mdb001'."""
    s = str(v).strip()
    m = REF_RE.search(s)
    if m:
        num = re.sub(r"\D", "", m.group(1))
        return f"mdb{int(num):03d}"
    m = NUM_RE.match(s)
    if m:
        return f"mdb{int(m.group(1)):03d}"
    return None


# ------------------------------------------------------------------ inventaire
def scan_images(src: str) -> dict[str, str]:
    """Indexe les images par identifiant mdbNNN, récursivement.

    En cas de doublon (plusieurs formats ou tailles du même mdbNNN), on privilégie
    le PGM d'origine, puis le fichier le plus volumineux.
    """
    best: dict[str, tuple[int, int, str]] = {}   # key -> (priorité, taille, chemin)
    for root, _dirs, files in os.walk(src):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMG_EXT:
                continue
            key = norm_id(os.path.splitext(name)[0])
            if key is None:
                continue
            path = os.path.join(root, name)
            prio = 1 if ext == ".pgm" else 0
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            cur = best.get(key)
            if cur is None or (prio, size) > (cur[0], cur[1]):
                best[key] = (prio, size, path)
    return {k: v[2] for k, v in best.items()}


def candidate_label_files(src: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(src):
        for name in files:
            if name.lower().endswith((".csv", ".txt", ".tsv", ".xlsx")):
                out.append(os.path.join(root, name))
    # mias_info.csv d'abord (annotations originales), puis les autres CSV
    def rank(p):
        b = os.path.basename(p).lower()
        return (0 if b == "mias_info.csv" else
                1 if ("info" in b and "derived" not in b) else
                2 if b.endswith(".csv") else 3, len(p))
    return sorted(out, key=rank)


def read_tables(path: str) -> list:
    import pandas as pd
    out = []
    if path.lower().endswith(".xlsx"):
        try:
            out.append(pd.read_excel(path, dtype=str))
        except Exception:
            pass
        return out
    for header in ("infer", None):
        try:
            df = pd.read_csv(path, sep=None, engine="python", header=header,
                             comment="#", skip_blank_lines=True,
                             on_bad_lines="skip", dtype=str)
            if len(df.columns) >= 2 and len(df) >= 10:
                df.columns = [str(c) for c in df.columns]
                out.append(df)
        except Exception:
            continue
    return out


def pick_id_column(df) -> str | None:
    """Colonne identifiant : d'abord par nom connu, sinon par contenu."""
    for c in df.columns:
        if canon(c) in ID_COL_HINTS:
            if df[c].map(norm_id).notna().mean() >= 0.8:
                return c
    best, best_frac = None, 0.0
    for c in df.columns:
        frac = float(df[c].map(norm_id).notna().mean())
        if frac > best_frac:
            best, best_frac = c, frac
    return best if best_frac >= 0.8 else None


def pick_label_column(df, scale: str, forced: str | None):
    """Retourne (colonne, échelle). Priorité : forcée > nom connu > contenu."""
    if forced:
        match = [c for c in df.columns if canon(c) == canon(forced)]
        if not match:
            return None
        c = match[0]
        for sc in (scale, "bg", "birads"):
            if df[c].map(lambda v: norm_with(SCALES[sc]["aliases"], v)).notna().mean() >= 0.8:
                return c, sc
        return None

    hints = BG_COL_HINTS if scale == "bg" else BIRADS_COL_HINTS
    for c in df.columns:
        if canon(c) in hints:
            ok = df[c].map(lambda v: norm_with(SCALES[scale]["aliases"], v))
            if ok.notna().mean() >= 0.8 and ok.dropna().nunique() >= 2:
                return c, scale
    # repli : n'importe quelle colonne dont le contenu correspond à l'échelle
    for c in df.columns:
        ok = df[c].map(lambda v: norm_with(SCALES[scale]["aliases"], v))
        if ok.notna().mean() >= 0.8 and ok.dropna().nunique() >= 2:
            return c, scale
    return None


def show_crosstab(src: str):
    """Affiche BG x DENSITY quand les deux existent : preuve que DENSITY est
    un simple recodage de BG et n'apporte aucune information nouvelle."""
    import pandas as pd
    bg_map, dens_map = {}, {}
    for path in candidate_label_files(src):
        for df in read_tables(path):
            idc = pick_id_column(df)
            if idc is None:
                continue
            for c in df.columns:
                if canon(c) in BG_COL_HINTS:
                    for i, v in zip(df[idc], df[c]):
                        k, d = norm_id(i), norm_with(BG_ALIASES, v)
                        if k and d is not None:
                            bg_map[k] = d
                if canon(c) in BIRADS_COL_HINTS:
                    for i, v in zip(df[idc], df[c]):
                        k, d = norm_id(i), norm_with(BIRADS_ALIASES, v)
                        if k and d is not None:
                            dens_map[k] = d
    communs = sorted(set(bg_map) & set(dens_map))
    if len(communs) < 10:
        return
    t = pd.crosstab(pd.Series([BG_ALIASES and ["F", "G", "D"][bg_map[k]] for k in communs],
                               name="BG (original, 3 niveaux)"),
                     pd.Series([["A", "B", "C", "D"][dens_map[k]] for k in communs],
                               name="DENSITY (derivee, 4 niveaux)"))
    print("\n[verif] Tableau croise BG x DENSITY sur", len(communs), "images :")
    print(t.to_string())
    n_dens = t.shape[1]
    if n_dens <= 3 or (t > 0).sum(axis=1).max() == 1:
        print("[verif] Chaque niveau BG correspond a UN seul niveau DENSITY :")
        print("        DENSITY est un renommage de BG, sans information nouvelle.")
        print("        Utiliser BG (3 classes authentiques) est le choix defendable.")


# --------------------------------------------------------------------- images
def load_image(path: str, img_size: int, keep_aspect: bool = True) -> np.ndarray:
    """Image MIAS -> uint8 [img_size, img_size].

    Pas de VOI LUT ici contrairement aux DICOM VinDr : les fichiers MIAS sont
    déjà des images 8 bits fenêtrées. Normalisation min-max, puis
    redimensionnement préservant le rapport d'aspect (letterbox) — écraser une
    image en carré déformerait la surface de tissu dense, qui est le signal.
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
        canvas.paste(img, ((img_size - img.size[0]) // 2, (img_size - img.size[1]) // 2))
        img = canvas
    else:
        img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def patient_id_of(key: str) -> str:
    """MIAS livre les seins par paires — gauche et droite d'une même patiente :
    mdb001+mdb002, mdb003+mdb004, etc. On les regroupe pour que les deux images
    d'une patiente restent du même côté du split, sinon il y a fuite de données.
    """
    m = re.fullmatch(r"mdb(\d+)", key)
    if not m:
        return key
    n = int(m.group(1))
    return f"p{(n + 1) // 2:03d}"


def make_split(patients, patient_density, n_classes, test_frac, seed):
    """Split stratifié AU NIVEAU PATIENTE, jamais au niveau image."""
    rng = np.random.default_rng(seed)
    assign = {}
    for k in range(n_classes):
        grp = sorted([p for p in patients if patient_density[p] == k])
        rng.shuffle(grp)
        n_test = max(1, int(round(len(grp) * test_frac))) if grp else 0
        for p in grp[:n_test]:
            assign[p] = "test"
        for p in grp[n_test:]:
            assign[p] = "training"
    return assign


def inspect(src: str, images: dict[str, str]):
    print(f"\n=== DIAGNOSTIC de {src} ===")
    print(f"{len(images)} images indexees")
    exts, dirs = {}, {}
    for p in images.values():
        e = os.path.splitext(p)[1].lower(); exts[e] = exts.get(e, 0) + 1
        d = os.path.relpath(os.path.dirname(p), src); dirs[d] = dirs.get(d, 0) + 1
    print("extensions :", exts)
    print("dossiers   :", dict(list(dirs.items())[:10]))
    print("exemples   :", sorted(images)[:5])

    cands = candidate_label_files(src)
    print(f"\n{len(cands)} fichiers tabulaires candidats (ordre de priorite) :")
    for p in cands[:10]:
        print("  -", os.path.relpath(p, src))
        for df in read_tables(p)[:1]:
            print("    colonnes  :", list(df.columns)[:12])
            print("    1re ligne :", df.iloc[0].tolist()[:12])
            idc = pick_id_column(df)
            print("    id detecte:", idc)
            for sc in ("bg", "birads"):
                r = pick_label_column(df, sc, None)
                if r:
                    print(f"    echelle {sc:6s} -> colonne '{r[0]}'")
    show_crosstab(src)
    print("\nSi la detection echoue, envoie-moi ce diagnostic complet.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="racine du dataset MIAS")
    ap.add_argument("--out", default="./mias_cache")
    ap.add_argument("--scale", choices=["bg", "birads"], default="bg",
                     help="bg = densite MIAS originale F/G/D, 3 classes (defaut) ; "
                          "birads = colonne DENSITY derivee A/B/C/D, 4 classes "
                          "(NON authentique, voir docstring)")
    ap.add_argument("--label-col", default=None,
                     help="forcer un nom de colonne, ex. BG ou DENSITY")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--no-keep-aspect", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", action="store_true", help="diagnostiquer et s'arreter")
    args = ap.parse_args()

    try:
        from PIL import Image  # noqa: F401
        import pandas as pd
    except ImportError as e:
        sys.exit(f"Dependance manquante ({e.name}). pip install pillow pandas numpy")

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        sys.exit(f"Dossier introuvable : {src}")

    images = scan_images(src)
    print(f"[scan] {len(images)} images indexees")
    if args.inspect:
        inspect(src, images)
        return
    if not images:
        inspect(src, images)
        sys.exit("Aucune image trouvee. Verifie --src.")

    if args.scale == "birads":
        print("\n" + "!" * 74)
        print("ATTENTION : --scale birads utilise la colonne DENSITY, qui est DERIVEE")
        print("de BG par l'auteur du miroir Kaggle. MIAS ne contient AUCUNE annotation")
        print("BI-RADS de densite : 4 categories ne peuvent pas etre reconstruites a")
        print("partir des 3 categories originales. A ne pas presenter comme du BI-RADS.")
        print("!" * 74)
        show_crosstab(src)

    # Recherche des étiquettes
    labels, used_path, used_col, used_scale = None, None, None, None
    for path in candidate_label_files(src):
        for df in read_tables(path):
            idc = pick_id_column(df)
            if idc is None:
                continue
            res = pick_label_column(df, args.scale, args.label_col)
            if res is None:
                continue
            col, sc = res
            aliases = SCALES[sc]["aliases"]
            lab, conflits = {}, 0
            for raw_i, raw_d in zip(df[idc], df[col]):
                k, d = norm_id(raw_i), norm_with(aliases, raw_d)
                if k is None or d is None:
                    continue
                if k in lab and lab[k] != d:
                    conflits += 1
                lab[k] = d
            if len(lab) >= 10:
                labels, used_path, used_col, used_scale = lab, path, col, sc
                print(f"\n[detect] fichier   : {os.path.relpath(path, src)}")
                print(f"[detect] identifiant: '{idc}'")
                print(f"[detect] etiquette  : '{col}' (echelle {sc}, "
                      f"{SCALES[sc]['n']} classes {SCALES[sc]['names']})")
                if conflits:
                    print(f"[warn]   {conflits} lignes en conflit (plusieurs anomalies "
                          f"par image) : la derniere l'emporte")
                break
        if labels:
            break

    if labels is None:
        inspect(src, images)
        sys.exit("Etiquettes introuvables. Lance --inspect et envoie la sortie.")

    n_classes = SCALES[used_scale]["n"]
    class_names = SCALES[used_scale]["names"]

    communs = sorted(set(labels) & set(images))
    sans_img = sorted(set(labels) - set(images))
    sans_lab = sorted(set(images) - set(labels))
    print(f"\n[match] {len(communs)} images etiquetees et presentes")
    if sans_img:
        print(f"[warn]  {len(sans_img)} etiquettes sans image : {sans_img[:5]}")
    if sans_lab:
        print(f"[warn]  {len(sans_lab)} images sans etiquette : {sans_lab[:5]}")
    if not communs:
        inspect(src, images)
        sys.exit("Aucune correspondance entre etiquettes et images.")

    # Densité d'une patiente : la plus élevée de ses deux seins (prudence clinique)
    patient_density = {}
    for k in communs:
        p = patient_id_of(k)
        patient_density[p] = max(patient_density.get(p, -1), labels[k])
    patients = sorted(patient_density)
    print(f"[match] {len(patients)} patientes (paires gauche/droite regroupees)")

    assign = make_split(patients, patient_density, n_classes, args.test_frac, args.seed)

    out_dir = os.path.abspath(args.out)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    rows, echecs = [], []
    for i, key in enumerate(communs, 1):
        out_npy = os.path.join(cache_dir, f"{key}.npy")
        if not os.path.exists(out_npy):
            try:
                arr = load_image(images[key], args.img_size, not args.no_keep_aspect)
            except Exception as e:
                echecs.append(f"{key}: {type(e).__name__}: {e}")
                continue
            tmp = out_npy + ".part"
            # np.save ajoute .npy si le nom n'en a pas : on passe un descripteur
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, out_npy)          # ecriture atomique -> reprise sure
        p = patient_id_of(key)
        rows.append({"image_id": key, "patient_id": p,
                      "density_label": labels[key], "split": assign[p]})
        if i % 50 == 0:
            print(f"  {i}/{len(communs)}")

    df_out = pd.DataFrame(rows)
    labels_path = os.path.join(out_dir, "labels.csv")
    df_out.to_csv(labels_path, index=False)

    info = {"dataset": "MIAS", "n_classes": n_classes, "class_names": class_names,
            "scale": used_scale, "label_column": used_col,
            "label_file": os.path.relpath(used_path, src),
            "img_size": args.img_size, "keep_aspect": not args.no_keep_aspect,
            "n_images": len(df_out), "n_patients": len(patients), "seed": args.seed,
            "authentic_labels": used_scale == "bg",
            "note": ("BG est l'annotation MIAS originale (3 niveaux ordinaux)."
                      if used_scale == "bg" else
                      "DENSITY est DERIVEE de BG par le miroir Kaggle : MIAS ne "
                      "contient pas d'annotation BI-RADS de densite. Ne pas "
                      "presenter ce resultat comme du BI-RADS 4 classes.")}
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n[fin] {len(df_out)} images en cache -> {cache_dir}")
    print(f"[fin] labels -> {labels_path}")
    print(f"[fin] info   -> {os.path.join(out_dir, 'dataset_info.json')}")
    if echecs:
        print(f"[warn] {len(echecs)} echecs de conversion :")
        for e in echecs[:5]:
            print("   " + e)

    print(f"\nDistribution (lignes = split, colonnes = {class_names}) :")
    print(df_out.groupby(["split", "density_label"]).size()
                .unstack(fill_value=0).to_string())

    for split_name, g in df_out.groupby("split"):
        c = g["density_label"].value_counts()
        absentes = [k for k in range(n_classes) if k not in c.index]
        faibles = {int(k): int(v) for k, v in c.items() if v < 15}
        if absentes:
            print(f"[ATTENTION] {split_name}: classes absentes {absentes} — "
                  f"comparaison impossible en l'etat")
        if faibles:
            print(f"[warn] {split_name}: classes sous 15 images {faibles} "
                  f"-> QWK instable, rapporter plusieurs seeds")

    mb = sum(e.stat().st_size for e in os.scandir(cache_dir)) / 1e6
    print(f"\n[cache] {mb:.0f} Mo")
    print("\n[suite] Ouvre notebooks/mias_ordinal_density.ipynb et mets")
    print(f"        ROOT = '{out_dir}'")
    print("        Le notebook lit dataset_info.json : N_CLASSES est automatique.")


if __name__ == "__main__":
    main()
