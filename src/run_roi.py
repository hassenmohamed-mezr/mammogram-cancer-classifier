"""Batch ROI extraction over roi_manifest.csv (resumable).

For every image: load DICOM -> winner-style ROI (YOLOX, else Otsu, else full frame) -> window
-> resize so the long side <= max_side -> save PNG.
Writes roi_results.csv incrementally; re-running skips images already done,
so a dead Kaggle session never costs you the whole run.

    from src.run_roi import run
    run("/kaggle/working/splits/roi_manifest.csv", "/kaggle/working/roi_out", roi=roi)
"""
from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

from .dicom_roi import load_dicom, process


def _resize_long_side(img, max_side: int):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def run(manifest_csv, out_dir, roi=None, max_side=1024,
        prefetch=4, flush_every=200, limit=None):
    out = Path(out_dir)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    res_path, err_path = out / "roi_results.csv", out / "roi_errors.csv"

    man = pd.read_csv(manifest_csv)
    done = set(pd.read_csv(res_path)["image_id"]) if res_path.exists() else set()
    todo = man[~man["image_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    total = len(todo)
    print(f"{len(done)} already done, {total} to process")

    rows, errors, n, t0 = [], [], 0, time.time()

    def flush():
        nonlocal rows, errors
        if rows:
            pd.DataFrame(rows).to_csv(res_path, mode="a", header=not res_path.exists(), index=False)
            rows = []
        if errors:
            pd.DataFrame(errors).to_csv(err_path, mode="a", header=not err_path.exists(), index=False)
            errors = []

    with ThreadPoolExecutor(max_workers=prefetch) as ex:
        it = iter(todo.itertuples(index=False))
        q = deque()

        def submit():
            r = next(it, None)
            if r is not None:
                q.append((r, ex.submit(load_dicom, r.path)))

        for _ in range(prefetch):          # bounded read-ahead: decode while the GPU works
            submit()
        while q:
            r, fut = q.popleft()
            submit()
            try:
                data = fut.result()
                crop, info = process(data, roi)
                crop = _resize_long_side(crop, max_side)
                p = out / "crops" / f"{r.image_id}.png"
                cv2.imwrite(str(p), crop)
                rows.append(dict(image_id=r.image_id, patient_id=r.patient_id, cancer=r.cancer,
                                 roi_path=str(p), source=info["source"], conf=info["conf"],
                                 box=str(info["box"]), orig_h=data.raw.shape[0], orig_w=data.raw.shape[1]))
            except Exception as e:  # keep going; failed images are listed in roi_errors.csv
                errors.append(dict(image_id=r.image_id, path=r.path, error=repr(e)))
            n += 1
            if n % flush_every == 0:
                flush()
                el = time.time() - t0
                print(f"{n}/{total} | {el/60:.1f} min | eta {(el/n)*(total-n)/60:.1f} min")
    flush()
    print(f"finished: {n} processed, results -> {res_path}")
    if err_path.exists():
        print(f"some images failed, see {err_path}")
