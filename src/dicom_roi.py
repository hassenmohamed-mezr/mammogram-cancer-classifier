"""DICOM -> breast ROI -> 8-bit crop, following the RSNA 1st-place pipeline.

Ported (MIT licence) from dangnh0611/kaggle_rsna_breast_cancer:
  src/tools/_prepair_classification_dataset_stage2.py, src/roi_det/roi_extract.py,
  src/utils/windowing.py.  See third_party/NOTICE.md.

Pipeline per image (same order as the winner):
  1. raw pixels -> min-max -> 0..255 (inverted if MONOCHROME1)     = detection image
  2. YOLOX-nano (416x416) box if conf >= 0.5 and area >= 4% of the image,
     else Otsu fallback, else the full frame
  3. crop the RAW pixels with that box, apply the DICOM window (first window),
     invert if MONOCHROME1, cast to uint8
Differences from the original: pydicom instead of dicomsdl/DALI, and the plain
PyTorch checkpoint instead of a TensorRT engine.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from .roi import fallback_box


# --------------------------------------------------------------------------- DICOM
@dataclass
class DicomData:
    raw: np.ndarray                      # 2-D float32, original pixel values
    invert: bool = False                 # MONOCHROME1
    centers: list = field(default_factory=list)
    widths: list = field(default_factory=list)
    voi_func: str = "LINEAR"


def _as_float_list(v) -> list:
    if v is None:
        return []
    try:
        return [float(x) for x in v]     # pydicom MultiValue
    except TypeError:
        try:
            return [float(v)]
        except (TypeError, ValueError):
            return []
    except ValueError:
        return []


def load_dicom(path: str) -> DicomData:
    """Read a DICOM (or, for tests, a PNG/JPG) into raw pixels + window metadata."""
    if not path.lower().endswith(".dcm"):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return DicomData(raw=img.astype(np.float32))
    import pydicom

    ds = pydicom.dcmread(path)
    raw = ds.pixel_array.astype(np.float32)
    centers = _as_float_list(ds.get("WindowCenter"))
    widths = _as_float_list(ds.get("WindowWidth"))
    if len(centers) != len(widths):
        centers, widths = [], []
    return DicomData(
        raw=raw,
        invert=(getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1"),
        centers=centers,
        widths=widths,
        voi_func=str(ds.get("VOILUTFunction", "LINEAR")).upper(),
    )


def min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else x - lo


def apply_window(arr, center, width, func="LINEAR", y_min=0, y_max=255):
    """DICOM VOI windowing (same maths as the winner's windowing.py)."""
    arr = arr.astype(np.float32)
    y_range = y_max - y_min
    if func in ("LINEAR", "LINEAR_EXACT"):
        if func == "LINEAR":
            if width < 1:
                raise ValueError("window width must be >= 1 for LINEAR")
            center -= 0.5
            width -= 1
        s = y_range / width
        b = (-center / width + 0.5) * y_range + y_min
        return np.clip(arr * s + b, y_min, y_max)
    if func == "SIGMOID":
        return y_range / (1 + np.exp(-4 * (arr - center) / width)) + y_min
    raise ValueError(f"unsupported VOI LUT function {func}")


# --------------------------------------------------------------------------- Otsu fallback (winner's)
def _extract_roi_otsu(img_u8: np.ndarray, gkernel=(5, 5)):
    """Winner's fallback. Returns ([x0,y0,x1,y1], area_pct) or (None, None). Modifies input."""
    ori_h, ori_w = img_u8.shape[:2]
    upper = np.percentile(img_u8, 95)
    img_u8[img_u8 > upper] = np.min(img_u8)
    img = cv2.GaussianBlur(img_u8, gkernel, 0) if gkernel else img_u8
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = cv2.dilate(img_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1)))
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, None
    areas = np.array([cv2.contourArea(c) for c in cnts])
    k = int(np.argmax(areas))
    x0, y0, w, h = cv2.boundingRect(cnts[k])
    x1, y1 = min(max(x0 + w, 0), ori_w), min(max(y0 + h, 0), ori_h)
    x0, y0 = min(max(x0, 0), ori_w), min(max(y0, 0), ori_h)
    return [x0, y0, x1, y1], areas[k] / (ori_h * ori_w)


# --------------------------------------------------------------------------- YOLOX detector
class WinnerRoi:
    """YOLOX-nano breast detector from the winning solution (PyTorch checkpoint).

    winner_repo : path of the cloned dangnh0611/kaggle_rsna_breast_cancer repo
    ckpt_path   : yolox_nano_416_roi_torch.pth (from the best-ckpts Kaggle dataset)
    """

    def __init__(self, winner_repo, ckpt_path, device=None, input_size=(416, 416),
                 conf_thres=0.5, nms_thres=0.9, area_pct_thres=0.04):
        import torch

        root = f"{winner_repo}/src/roi_det/YOLOX"
        if root not in sys.path:
            sys.path.insert(0, root)
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_h, self.input_w = input_size
        self.conf_thres, self.nms_thres, self.area_pct_thres = conf_thres, nms_thres, area_pct_thres

        # architecture = yolox_nano_bre_416.py: 1 class, depth 0.33, width 0.25, depthwise
        ch = [256, 512, 1024]
        backbone = YOLOPAFPN(0.33, 0.25, in_channels=ch, act="silu", depthwise=True)
        head = YOLOXHead(1, 0.25, in_channels=ch, act="silu", depthwise=True)
        model = YOLOX(backbone, head)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

    def detect(self, img: np.ndarray):
        """img: 2-D float32 in 0..255 (dark background). Returns (xyxy|None, source, conf)."""
        torch = self.torch
        import torch.nn.functional as F
        from yolox.utils import postprocess

        H, W = self.input_h, self.input_w
        ori_h, ori_w = img.shape
        with torch.no_grad():
            t = torch.from_numpy(img).to(self.device)
            ratio = min(H / ori_h, W / ori_w)
            resized = F.interpolate(t.view(1, 1, ori_h, ori_w), mode="bilinear",
                                    scale_factor=ratio, recompute_scale_factor=True)[0, 0]
            padded = torch.full((H, W), 114, dtype=resized.dtype, device=self.device)
            padded[: resized.size(0), : resized.size(1)] = resized
            x = padded[None, None].expand(1, 3, H, W).contiguous()
            out = postprocess(self.model(x), 1, self.conf_thres, self.nms_thres)[0]

            if out is not None:
                boxes = out[:, :4] / ratio
                score = out[:, 4] * out[:, 5]
                k = int(score.argmax())
                x0, y0, x1, y1 = (int(v) for v in boxes[k].tolist())
                x0, x1 = min(max(x0, 0), ori_w), min(max(x1, 0), ori_w)
                y0, y1 = min(max(y0, 0), ori_h), min(max(y1, 0), ori_h)
                if (x1 - x0) * (y1 - y0) / (ori_h * ori_w) >= self.area_pct_thres:
                    return [x0, y0, x1, y1], "yolo", float(score[k])
            small = resized.to(torch.uint8).cpu().numpy()

        # winner's fallbacks: Otsu on the resized image, else full frame
        xyxy, area_pct = _extract_roi_otsu(small)
        if xyxy is not None and area_pct >= self.area_pct_thres:
            x0, y0, x1, y1 = (min(max(int(v / ratio), 0), lim)
                              for v, lim in zip(xyxy, (ori_w, ori_h, ori_w, ori_h)))
            return [x0, y0, x1, y1], "otsu", None
        return None, "full", None


# --------------------------------------------------------------------------- one image
def process(data: DicomData, roi: WinnerRoi | None = None):
    """DicomData -> (crop_uint8, info). roi=None uses only the simple fallback."""
    raw = data.raw
    h, w = raw.shape
    det = min_max(raw) * 255.0
    if data.invert:
        det = 255.0 - det

    if roi is not None:
        box, source, conf = roi.detect(det.astype(np.float32))
    else:
        box, source, conf = list(fallback_box(det.astype(np.uint8))), "fallback", None

    if box is None or box[2] - box[0] < 10 or box[3] - box[1] < 10:
        box, source = [0, 0, w, h], "full"
    x0, y0, x1, y1 = box
    crop = raw[y0:y1, x0:x1]

    if data.widths:
        try:
            crop = apply_window(crop, data.centers[0], data.widths[0], data.voi_func)
        except ValueError:
            crop = min_max(crop) * 255.0
    else:
        crop = min_max(crop) * 255.0
    if data.invert:
        crop = 255.0 - crop
    crop = np.clip(crop, 0, 255).astype(np.uint8)
    return crop, {"source": source, "conf": conf, "box": tuple(int(v) for v in box)}
