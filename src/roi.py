"""Breast ROI cropping.

Primary: the YOLOX breast detector from the RSNA winning solution
(dangnh0611/kaggle_rsna_breast_cancer) - credited in README / third_party/NOTICE.md.
Fallback: classic Otsu + largest connected component.

The YOLOX call itself is injected as `detector`, a callable:
    detector(img_uint8) -> (x1, y1, x2, y2, conf) or None
Wire it to the inference code you already ran in rsna_inference.ipynb.
"""
from __future__ import annotations

import cv2
import numpy as np

try:  # pydicom >= 3
    from pydicom.pixels import apply_voi_lut
except ImportError:  # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_voi_lut


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def load_image(path: str) -> np.ndarray:
    """Read a PNG/JPG or a DICOM (.dcm) as a 2-D uint8 image.

    DICOM: apply VOI LUT when present, un-invert MONOCHROME1, min-max to 8-bit.
    NOTE: if the YOLOX boxes look wrong, replace this with the exact DICOM->PNG
    conversion used in your rsna_inference notebook (the winner's pipeline).
    """
    if not path.lower().endswith(".dcm"):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        return to_uint8(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    import pydicom

    ds = pydicom.dcmread(path)
    x = ds.pixel_array.astype(np.float32)
    try:
        x = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        x = x.max() - x
    return to_uint8(x)


def fallback_box(img_u8: np.ndarray, margin: float = 0.02):
    """Bounding box of the largest bright component (Otsu). Returns x1,y1,x2,y2."""
    g = img_u8 if img_u8.ndim == 2 else cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    # some exports are inverted (bright background): detect via border brightness
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    if np.median(border) > 127:
        g = 255 - g
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return 0, 0, w, h
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = (int(stats[idx, k]) for k in (0, 1, 2, 3))
    mx, my = int(margin * w), int(margin * h)
    return max(x - mx, 0), max(y - my, 0), min(x + bw + mx, w), min(y + bh + my, h)


def crop_roi(img: np.ndarray, detector=None, min_conf: float = 0.3):
    """Return (crop_u8, info). Uses the detector if confident, else the fallback."""
    u8 = to_uint8(img)
    h, w = u8.shape[:2]
    box, conf, source = None, None, "fallback"
    if detector is not None:
        res = detector(u8)
        if res is not None and res[4] >= min_conf:
            x1, y1, x2, y2, conf = res
            box = (max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h))
            if box[2] - box[0] > 10 and box[3] - box[1] > 10:
                source = "yolo"
            else:
                box = None
    if box is None:
        box = fallback_box(u8)
    x1, y1, x2, y2 = box
    return u8[y1:y2, x1:x2], {"source": source, "conf": conf, "box": box}
