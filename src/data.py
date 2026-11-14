"""Data utilities for the mammogram classifier (RSNA screening mammography).

- Patient-level train/val/test split (no patient appears in two splits).
- Class balancing for TRAIN only (all positives + N negatives per positive).
- val/test keep the natural distribution unless --eval-neg-ratio is given.
- Writes one ROI manifest so the (slow) ROI step runs exactly once.

Usage (Kaggle):
    python -m src.data --csv /kaggle/input/<comp>/train.csv \
        --img-root /kaggle/input/<png-dataset> \
        --img-template "{root}/{patient_id}/{image_id}.png" \
        --out /kaggle/working/splits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLS = {"patient_id", "image_id", "cancer"}


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    return df


def patient_split(df: pd.DataFrame, fractions=(0.70, 0.15, 0.15), seed: int = 42) -> dict:
    """Split by patient_id, stratified on 'patient has any cancer image'."""
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("fractions must sum to 1")
    pat = df.groupby("patient_id", as_index=False)["cancer"].max()
    train_p, rest_p = train_test_split(
        pat, test_size=1 - fractions[0], stratify=pat["cancer"], random_state=seed
    )
    val_share = fractions[1] / (fractions[1] + fractions[2])
    val_p, test_p = train_test_split(
        rest_p, train_size=val_share, stratify=rest_p["cancer"], random_state=seed
    )
    splits = {
        name: df[df["patient_id"].isin(p["patient_id"])].reset_index(drop=True)
        for name, p in (("train", train_p), ("val", val_p), ("test", test_p))
    }
    check_no_leakage(splits)
    return splits


def check_no_leakage(splits: dict) -> None:
    ids = {k: set(v["patient_id"]) for k, v in splits.items()}
    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ids[names[i]] & ids[names[j]]
            assert not overlap, f"patient leakage between {names[i]} and {names[j]}: {len(overlap)}"


def balance(df: pd.DataFrame, neg_per_pos: float, seed: int = 42) -> pd.DataFrame:
    """All positives + round(neg_per_pos * n_pos) negatives.

    Negatives are shuffled once and the head is taken, so a smaller ratio
    is always a subset of a larger one (same seed) -> ROI runs once.
    """
    pos = df[df["cancer"] == 1]
    neg = df[df["cancer"] == 0].sample(frac=1.0, random_state=seed)
    n = min(len(neg), int(round(len(pos) * neg_per_pos)))
    out = pd.concat([pos, neg.head(n)])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def add_path(df: pd.DataFrame, img_root: str, template: str) -> pd.DataFrame:
    df = df.copy()
    df["path"] = [
        template.format(root=img_root, patient_id=p, image_id=i)
        for p, i in zip(df["patient_id"], df["image_id"])
    ]
    return df


def summarize(name: str, df: pd.DataFrame) -> dict:
    pos = int((df["cancer"] == 1).sum())
    return {
        "split": name,
        "images": len(df),
        "patients": int(df["patient_id"].nunique()),
        "positive_images": pos,
        "positive_rate": round(pos / max(len(df), 1), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-root", default="")
    ap.add_argument("--img-template", default="{root}/{patient_id}/{image_id}.png")
    ap.add_argument("--ratios", type=float, nargs="+", default=[1, 3],
                    help="negatives per positive for train subsets")
    ap.add_argument("--eval-neg-ratio", type=float, default=None,
                    help="optional: cap negatives per positive in val/test to save ROI time "
                         "(distorts prevalence; AUC unaffected, PR-AUC is)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.csv)
    splits = patient_split(df, seed=args.seed)

    files = {}
    for r in args.ratios:
        tag = f"train_{r:g}to1"
        files[tag] = balance(splits["train"], r, args.seed)

    for name in ("val", "test"):
        d = splits[name]
        if args.eval_neg_ratio is not None:
            d = balance(d, args.eval_neg_ratio, args.seed)
        files[name] = d

    files = {k: add_path(v, args.img_root, args.img_template) for k, v in files.items()}
    for k, v in files.items():
        v.to_csv(out / f"{k}.csv", index=False)

    manifest = (
        pd.concat(files.values())
        .drop_duplicates("image_id")
        .reset_index(drop=True)
    )
    manifest.to_csv(out / "roi_manifest.csv", index=False)

    summary = [summarize(k, v) for k, v in files.items()]
    summary.append(summarize("roi_manifest (unique images)", manifest))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
