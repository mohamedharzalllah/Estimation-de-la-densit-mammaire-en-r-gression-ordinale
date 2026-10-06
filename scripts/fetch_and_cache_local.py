#!/usr/bin/env python3
"""Télécharge VinDr-Mammo depuis PhysioNet ET convertit en cache NPY, en un passage.

Pourquoi ce script : les DICOM de VinDr-Mammo pèsent ~19,7 Mo chacun (mesuré),
soit ~46 Go pour quelques centaines d'études. Le cache NPY en 512px uint8 ne
pèse que 262 Ko par image, soit un facteur 75. Le débit PhysioNet depuis Google
Colab plafonne autour de 86 Ko/s (76 h pour un sous-ensemble utilisable), alors
qu'une connexion domestique est souvent bien plus rapide.

Stratégie : télécharger + convertir + SUPPRIMER le DICOM immédiatement. Le disque
ne contient jamais plus de quelques fichiers temporaires, et le résultat final
(quelques centaines de Mo) s'envoie facilement vers Colab ou Kaggle.

Prérequis : compte PhysioNet credentialed + DUA VinDr-Mammo signée.
    pip install requests pydicom pillow numpy pandas

Exemple (sous-ensemble équilibré, vues CC seules, prototype rapide) :
    python scripts/fetch_and_cache_local.py --user MON_LOGIN \
        --out ./vindr_cache --views CC --cap-train 40 --cap-test 15 \
        --img-size 512 --workers 8

Le script est REPRENABLE : relance-le après une coupure, il saute ce qui est déjà
converti. Ctrl+C est propre.

Sortie :
    <out>/cache/<image_id>.npy      images 512x512 uint8
    <out>/labels.csv                image_id, study_id, laterality, view_position,
                                     density_label (0=A..3=D), split
    <out>/breast-level_annotations.csv   annotations complètes (référence)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import io
import os
import sys
import threading
import time

import numpy as np

BASE = "https://physionet.org/files/vindr-mammo/1.0.0"
# PhysioNet renvoie 403 aux User-Agent de bibliothèques HTTP ; il faut se
# présenter comme un navigateur (constaté en testant requests contre wget).
HEADERS = {"User-Agent": "Mozilla/5.0"}
MIN_DICOM_BYTES = 100_000

_print_lock = threading.Lock()


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------- réseau
def make_session(user: str, password: str):
    import requests
    s = requests.Session()
    s.auth = (user, password)
    s.headers.update(HEADERS)
    return s


def download_bytes(session, url: str, timeout=180) -> bytes | None:
    try:
        r = session.get(url, stream=True, timeout=timeout)
        if r.status_code != 200:
            return None
        buf = io.BytesIO()
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                buf.write(chunk)
        return buf.getvalue()
    except Exception:
        return None


def download_csv(session, name: str, out_dir: str) -> str:
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        log(f"[csv] {name} déjà présent")
        return path
    log(f"[csv] téléchargement de {name}")
    data = download_bytes(session, f"{BASE}/{name}", timeout=120)
    if not data:
        sys.exit(f"Échec sur {name}. Vérifie identifiants PhysioNet et DUA signée.")
    with open(path, "wb") as f:
        f.write(data)
    return path


# ------------------------------------------------------------------ sélection
def parse_density(v) -> int | None:
    s = str(v).strip().upper().replace("_", " ")
    return "ABCD".index(s[-1]) if s and s[-1] in "ABCD" else None


def select_studies(csv_path: str, views: str | None, cap_train: int, cap_test: int,
                    seed: int):
    """Sélection stratifiée AVEC enrichissement des classes rares.

    VinDr-Mammo est très déséquilibré (population de dépistage vietnamienne :
    densité C ~77 %, densité A <1 %). Une stratification proportionnelle donne
    2 études de classe A sur 400, ce qui rend la classe inévaluable. On prend
    donc jusqu'à `cap` études PAR CLASSE, soit tout ce qui existe si c'est moins.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["density_label"] = df["breast_density"].apply(parse_density)
    df = df.dropna(subset=["density_label"]).copy()
    df["density_label"] = df["density_label"].astype(int)

    if views:
        keep = [v.strip().upper() for v in views.split(",")]
        before = len(df)
        df = df[df["view_position"].str.upper().isin(keep)].reset_index(drop=True)
        log(f"[select] vues {keep} : {len(df)} images conservées sur {before}")

    img_col = "image_id" if "image_id" in df.columns else None
    if img_col is None:
        sys.exit("Colonne 'image_id' absente du CSV : chemins DICOM impossibles à construire.")

    # densité d'une étude = densité modale de ses images
    study_density = (df.groupby(["split", "study_id"])["density_label"]
                       .agg(lambda s: s.mode().iloc[0]).reset_index())

    log("\n[select] études disponibles par classe dans tout VinDr-Mammo :")
    log(study_density.groupby(["split", "density_label"]).size()
                     .unstack(fill_value=0).to_string())

    picked_ids = []
    for split_name, cap in (("training", cap_train), ("test", cap_test)):
        sub = study_density[study_density["split"] == split_name]
        parts = []
        for k in range(4):
            g = sub[sub["density_label"] == k]
            parts.append(g.sample(min(cap, len(g)), random_state=seed))
        picked = pd.concat(parts).sample(frac=1.0, random_state=seed)
        counts = picked["density_label"].value_counts().sort_index().to_dict()
        log(f"[select] {split_name}: {len(picked)} études | densités {counts}")
        faibles = {k: v for k, v in counts.items() if v < 8}
        if faibles:
            log(f"[warn]   classes sous 8 études {faibles} : métriques instables. "
                f"Envisage une analyse secondaire non-dense (A+B) vs dense (C+D).")
        picked_ids += picked["study_id"].astype(str).tolist()

    sel = set(picked_ids)
    todo = df[df["study_id"].astype(str).isin(sel)].reset_index(drop=True)
    est_gb = len(todo) * 19.7 / 1024
    log(f"\n[select] {len(todo)} images à traiter (~{est_gb:.1f} Go téléchargés, "
        f"~{len(todo) * 0.262:.0f} Mo de cache en sortie)")
    return todo, img_col


# ---------------------------------------------------------------- conversion
def dicom_bytes_to_array(data: bytes, img_size: int, keep_aspect: bool) -> np.ndarray:
    """DICOM en mémoire -> array uint8 [img_size, img_size].

    Ordre des opérations (voir src/data.py, identique) :
      1. VOI LUT / fenêtrage DICOM  -> restaure le contraste diagnostique.
         Sans cela l'image est plate, or la densité BI-RADS EST une information
         de contraste : on dégraderait la cible elle-même.
      2. inversion si MONOCHROME1   -> tissu dense = clair
      3. normalisation min-max
      4. letterbox, pour ne pas déformer une image ~2800x3518 en carré
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    from PIL import Image

    dcm = pydicom.dcmread(io.BytesIO(data))
    try:
        arr = apply_voi_lut(dcm.pixel_array, dcm).astype(np.float32)
    except Exception:
        arr = dcm.pixel_array.astype(np.float32)

    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

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


# --------------------------------------------------------------------- worker
class Stats:
    def __init__(self, total: int):
        self.total = total
        self.lock = threading.Lock()
        self.ok = self.skip = self.fail = 0
        self.bytes = 0
        self.t0 = time.time()
        self.errors: list[str] = []

    def update(self, status: str, nbytes: int = 0, err: str = ""):
        with self.lock:
            setattr(self, status, getattr(self, status) + 1)
            self.bytes += nbytes
            if err and len(self.errors) < 20:
                self.errors.append(err)
            done = self.ok + self.skip + self.fail
            if done % 10 == 0 or done == self.total:
                el = time.time() - self.t0
                mbs = self.bytes / 1e6 / max(el, 1)
                eta = (el / done * (self.total - done) / 60) if done else 0
                print(f"  {done}/{self.total}  ok={self.ok} skip={self.skip} "
                      f"fail={self.fail}  {mbs:.2f} Mo/s  "
                      f"{el/60:.1f} min écoulées, ~{eta:.0f} min restantes",
                      flush=True)


def process_one(args):
    session, row, img_col, cache_dir, img_size, keep_aspect, stats, retries = args
    iid = str(row[img_col])
    out = os.path.join(cache_dir, f"{iid}.npy")
    if os.path.exists(out):
        stats.update("skip")
        return
    url = f"{BASE}/images/{row['study_id']}/{iid}.dicom"

    for attempt in range(retries):
        data = download_bytes(session, url)
        if data and len(data) >= MIN_DICOM_BYTES:
            try:
                arr = dicom_bytes_to_array(data, img_size, keep_aspect)
            except Exception as e:
                stats.update("fail", err=f"{iid}: conversion {type(e).__name__}: {e}")
                return
            tmp = out + ".part"
            # np.save ajoute l'extension .npy si le nom n'en a pas : on passe un
            # descripteur de fichier pour garder exactement le nom voulu.
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, out)          # écriture atomique -> reprise sûre
            stats.update("ok", nbytes=len(data))
            return
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    stats.update("fail", err=f"{iid}: téléchargement échoué après {retries} essais")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="identifiant PhysioNet")
    ap.add_argument("--out", default="./vindr_cache")
    ap.add_argument("--views", default="CC",
                     help="vues à garder, ex. 'CC' (2 images/étude, volume divisé "
                          "par 2 sans perte d'étiquette) ou 'CC,MLO' pour tout")
    ap.add_argument("--cap-train", type=int, default=40,
                     help="études par classe de densité (entraînement)")
    ap.add_argument("--cap-test", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--no-keep-aspect", action="store_true",
                     help="redimensionner en carré (déforme, non recommandé)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import requests  # noqa: F401
        import pydicom   # noqa: F401
        from PIL import Image  # noqa: F401
        import pandas    # noqa: F401
    except ImportError as e:
        sys.exit(f"Dépendance manquante ({e.name}). "
                 f"Lance : pip install requests pydicom pillow numpy pandas")

    password = os.environ.get("PHYSIONET_PASSWORD") or getpass.getpass("Mot de passe PhysioNet: ")

    out_dir = os.path.abspath(args.out)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    session = make_session(args.user, password)
    csv_path = download_csv(session, "breast-level_annotations.csv", out_dir)

    todo, img_col = select_studies(csv_path, args.views, args.cap_train,
                                   args.cap_test, args.seed)

    # labels.csv écrit AVANT le téléchargement : utilisable même si on interrompt
    cols = [c for c in [img_col, "study_id", "laterality", "view_position",
                         "density_label", "split"] if c in todo.columns]
    labels_path = os.path.join(out_dir, "labels.csv")
    todo[cols].to_csv(labels_path, index=False)
    log(f"[labels] {labels_path}")

    already = sum(1 for _ in os.scandir(cache_dir) if _.name.endswith(".npy"))
    if already:
        log(f"[reprise] {already} fichiers déjà en cache, ils seront sautés")

    stats = Stats(len(todo))
    keep_aspect = not args.no_keep_aspect
    jobs = [(session, row, img_col, cache_dir, args.img_size, keep_aspect,
             stats, args.retries) for _, row in todo.iterrows()]

    log(f"\n[go] {args.workers} connexions parallèles, Ctrl+C pour arrêter proprement\n")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(process_one, jobs))
    except KeyboardInterrupt:
        log("\n[stop] interrompu — relance la même commande pour reprendre.")

    el = (time.time() - stats.t0) / 60
    log(f"\n[fin] {el:.1f} min | ok={stats.ok} skip={stats.skip} fail={stats.fail}")
    log(f"[fin] {stats.bytes/1e9:.2f} Go téléchargés -> cache {cache_dir}")
    if stats.errors:
        log(f"\n[erreurs] {len(stats.errors)} premières :")
        for e in stats.errors[:10]:
            log("   " + e)
        log("   Relance la commande pour réessayer les manquants.")

    n_npy = sum(1 for _ in os.scandir(cache_dir) if _.name.endswith(".npy"))
    mb = sum(_.stat().st_size for _ in os.scandir(cache_dir)) / 1e6
    log(f"\n[cache] {n_npy} fichiers .npy, {mb:.0f} Mo au total")
    log("[suite] compresse le dossier et envoie-le vers Colab/Drive :")
    log(f'         tar -czf vindr_cache.tar.gz -C "{out_dir}" cache labels.csv')


if __name__ == "__main__":
    main()
