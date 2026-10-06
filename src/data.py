"""Chargement des données pour la classification de densité mammaire BI-RADS A/B/C/D.

Deux sources :
- SyntheticBreastDensityDataset : générateur numpy/PIL pur, sans dépendance
  lourde, utilisé pour valider le pipeline avant d'avoir accès à VinDr-Mammo.
- VinDrMammoDataset : Dataset PyTorch qui lit les DICOM + le CSV de
  métadonnées de VinDr-Mammo. Nécessite torch, torchvision et pydicom
  (imports différés pour ne pas casser le mode synthétique/numpy).

Densité BI-RADS : A=0 (peu dense) < B=1 < C=2 < D=3 (très dense) — ordinal.
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ["A", "B", "C", "D"]
N_CLASSES = 4


def make_synthetic_breast_density(
    n_samples: int = 800,
    img_size: int = 64,
    n_classes: int = N_CLASSES,
    label_noise: float = 0.15,
    seed: int = 42,
):
    """Génère des images synthétiques en niveaux de gris + labels ordinaux A/B/C/D.

    Idée : la "densité" simulée est encodée par (a) le niveau moyen de gris
    (plus dense = plus clair, comme du tissu fibro-glandulaire sur une vraie
    mammographie) et (b) la texture (plus dense = plus de texture/bruit
    structuré). On ajoute du bruit de label pour éviter une séparation
    triviale et forcer les modèles à vraiment apprendre une frontière —
    utile pour observer des différences de comportement baseline vs ordinal.

    Retourne (images [N,H,W] float32 in [0,1], labels [N] int64 in [0..K-1]).
    """
    rng = np.random.default_rng(seed)
    true_labels = rng.integers(0, n_classes, size=n_samples)

    images = np.zeros((n_samples, img_size, img_size), dtype=np.float32)
    xx, yy = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size))
    radial = np.sqrt(xx**2 + yy**2)

    for i, label in enumerate(true_labels):
        base_level = 0.2 + 0.6 * (label / (n_classes - 1))  # A sombre -> D clair
        texture_freq = 3 + 2 * label  # plus dense = texture plus fine/fréquente
        texture = np.sin(texture_freq * xx * np.pi) * np.cos(texture_freq * yy * np.pi)
        texture_amp = 0.05 + 0.05 * label
        breast_mask = np.clip(1.2 - radial, 0, 1) ** 1.5  # forme ~ovale de sein
        img = base_level + texture_amp * texture
        img = img * breast_mask
        noise = rng.normal(0, 0.04, size=img.shape)
        img = np.clip(img + noise, 0, 1)
        images[i] = img.astype(np.float32)

    # Bruit de label : simule l'ambiguïté inter-observateur réelle du BI-RADS density
    # (accord inter-lecteur imparfait, confusions surtout entre classes adjacentes).
    # On tire le sens du décalage parmi les directions VALIDES : sinon un clip
    # annulerait le bruit pour les classes extrêmes A et D, et le taux de bruit
    # effectif serait nettement inférieur au taux demandé.
    noisy_labels = true_labels.copy()
    n_noisy = int(round(label_noise * n_samples))
    noisy_idx = rng.choice(n_samples, size=n_noisy, replace=False)
    for idx in noisy_idx:
        y = noisy_labels[idx]
        if y == 0:
            shift = 1
        elif y == n_classes - 1:
            shift = -1
        else:
            shift = rng.choice([-1, 1])
        noisy_labels[idx] = y + shift

    return images, noisy_labels.astype(np.int64)


def train_val_test_split(n_samples: int, val_frac=0.15, test_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_val = int(n_samples * val_frac)
    n_test = int(n_samples * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx, val_idx, test_idx


def load_vindr_annotations(data_root: str, csv_name: str = "breast-level_annotations.csv",
                            split: str | None = None):
    """Charge et nettoie breast-level_annotations.csv de VinDr-Mammo.

    Colonnes documentées par PhysioNet pour ce fichier :
      study_id, series_id, image_id, laterality, view_position, height, width,
      breast_birads, breast_density, split
    (`split` vaut "training" ou "test" — split officiel stratifié 4000/1000 études.)

    breast_density est fourni sous la forme "DENSITY A" ... "DENSITY D".
    """
    import os
    import pandas as pd

    csv_path = os.path.join(data_root, csv_name)
    df = pd.read_csv(csv_path)

    if split is not None:
        if "split" not in df.columns:
            raise KeyError("Colonne 'split' absente du CSV VinDr-Mammo")
        df = df[df["split"] == split].reset_index(drop=True)

    df["density_label"] = df["breast_density"].apply(parse_density)
    n_before = len(df)
    df = df.dropna(subset=["density_label"]).reset_index(drop=True)
    df["density_label"] = df["density_label"].astype(int)
    if len(df) < n_before:
        print(f"[data] {n_before - len(df)} lignes écartées (densité illisible)")
    return df


def parse_density(value) -> int | None:
    """'DENSITY A' / 'A' / 'density_a' -> 0 ; ... ; 'DENSITY D' -> 3."""
    if value is None:
        return None
    s = str(value).strip().upper().replace("_", " ")
    if not s or s in {"NAN", "NONE"}:
        return None
    last = s[-1]
    if last in "ABCD":
        return "ABCD".index(last)
    return None


class VinDrMammoDataset:
    """Dataset PyTorch pour les vraies données VinDr-Mammo (PhysioNet).

    Arborescence attendue (telle que publiée par PhysioNet) :
      data_root/
        breast-level_annotations.csv
        finding_annotations.csv
        metadata.csv
        images/<study_id>/<image_id>.dicom

    Note DICOM (critique pour la densité) : les mammographies VinDr sont en
    16 bits. Il faut appliquer la VOI LUT / le fenêtrage stocké dans les tags
    DICOM (WindowCenter/WindowWidth) AVANT toute normalisation, sinon on obtient
    des images plates et peu contrastées — or la densité BI-RADS est justement
    une information de contraste et d'intensité. On gère ensuite
    PhotometricInterpretation = MONOCHROME1 (échelle inversée).
    """

    def __init__(self, data_root: str, csv_name: str = "breast-level_annotations.csv",
                 split: str | None = None, img_size: int = 512, transform=None,
                 dataframe=None, keep_aspect: bool = True):
        self.data_root = data_root
        self.img_size = img_size
        self.transform = transform
        self.keep_aspect = keep_aspect
        self.df = (dataframe if dataframe is not None
                    else load_vindr_annotations(data_root, csv_name, split))
        if "image_id" not in self.df.columns:
            raise KeyError(
                "Colonne 'image_id' absente du CSV. Elle est nécessaire pour "
                "construire le chemin images/<study_id>/<image_id>.dicom. "
                "Ne pas se rabattre sur 'series_id' : plusieurs images "
                "partagent une même série, ce qui créerait des collisions.")
        self.image_col = "image_id"

    def __len__(self):
        return len(self.df)

    def class_counts(self):
        return self.df["density_label"].value_counts().sort_index().to_dict()

    def __getitem__(self, i):
        import os
        import torch

        row = self.df.iloc[i]
        dicom_path = os.path.join(self.data_root, "images", str(row["study_id"]),
                                   f"{row[self.image_col]}.dicom")
        arr = load_mammogram_dicom(dicom_path, self.img_size, self.keep_aspect)
        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)  # 1ch -> 3ch RGB

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, int(row["density_label"])


def load_mammogram_dicom(dicom_path: str, img_size: int = 512,
                          keep_aspect: bool = True) -> np.ndarray:
    """Lit un DICOM de mammographie et retourne un array float32 [H,W] dans [0,1].

    Ordre des opérations (important) :
      1. VOI LUT / fenêtrage DICOM  -> restaure le contraste diagnostique
      2. inversion si MONOCHROME1   -> tissu dense = clair
      3. normalisation min-max
      4. redimensionnement, en préservant le rapport d'aspect par défaut
         (une mammographie fait ~2800x3500 ; l'écraser en carré déforme la
         surface de tissu dense, qui est précisément le signal recherché)
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    from PIL import Image

    dcm = pydicom.dcmread(dicom_path)
    try:
        arr = apply_voi_lut(dcm.pixel_array, dcm).astype(np.float32)
    except Exception:
        # Certains DICOM n'ont ni VOI LUT ni fenêtrage exploitable
        arr = dcm.pixel_array.astype(np.float32)

    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    if keep_aspect:
        # letterbox : on redimensionne au plus grand côté puis on centre sur
        # un fond noir carré -> pas de distorsion géométrique
        w, h = img.size
        scale = img_size / max(w, h)
        img = img.resize((max(1, int(round(w * scale))),
                          max(1, int(round(h * scale)))), Image.BILINEAR)
        canvas = Image.new("L", (img_size, img_size), color=0)
        canvas.paste(img, ((img_size - img.size[0]) // 2,
                            (img_size - img.size[1]) // 2))
        img = canvas
    else:
        img = img.resize((img_size, img_size), Image.BILINEAR)

    return np.asarray(img, dtype=np.float32) / 255.0
