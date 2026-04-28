"""
Sinh mien phu tro 'rainy' cho triplet loss cua DA-Detect bang OpenCV,
khong phu thuoc rainy mask ngoai.

Moi ty le mua = streak layer ngau nhien + do mo nhe + giam sang ->
blend vao anh source. Label giu nguyen (copy tu source labels).

Dung:
  python tools/make_rainy.py \
    --src_img datasets/source_real/source_real/train/images \
    --src_lbl datasets/source_real/source_real/train/labels \
    --dst     datasets/rainy_real/train
"""
import argparse
import glob
import os
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm


def make_rain_layer(h, w, density=0.02, length=20, angle=75, thickness=1):
    """Tao mot mat na streak mua trang tren nen den."""
    layer = np.zeros((h, w), dtype=np.uint8)
    n_drops = int(h * w * density / max(1, length))
    angle_rad = np.deg2rad(angle)
    dx = int(np.cos(angle_rad) * length)
    dy = int(np.sin(angle_rad) * length)
    for _ in range(n_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        brightness = random.randint(180, 255)
        cv2.line(layer, (x, y), (x + dx, y + dy), brightness, thickness)
    # mo nhe de giong streak that
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    return layer


def apply_rain(img, severity=0.5, seed=None):
    """
    severity 0..1: 0 = chi toi mo, 1 = mua rat nang.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    h, w = img.shape[:2]
    # do suong nen
    fog_alpha = 0.08 + 0.18 * severity
    fog = np.full_like(img, 200, dtype=np.uint8)
    out = cv2.addWeighted(img, 1.0 - fog_alpha, fog, fog_alpha, 0)
    # mat na streak
    density = 0.008 + 0.035 * severity
    length = random.randint(14, 24)
    angle = random.uniform(65, 90)
    thickness = 1 if severity < 0.6 else random.choice([1, 2])
    streak = make_rain_layer(h, w, density=density, length=length,
                             angle=angle, thickness=thickness)
    streak_rgb = cv2.cvtColor(streak, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out_f = out.astype(np.float32) / 255.0
    # lighten blending: out + streak - out*streak
    out_f = out_f + streak_rgb - out_f * streak_rgb
    out_f = np.clip(out_f, 0.0, 1.0)
    # giam sang + giam tuong phan mo phong troi am
    out_f = out_f * (0.85 - 0.15 * severity)
    out_f = np.clip(out_f, 0.0, 1.0)
    # mo nhe
    k = 3 if severity < 0.5 else 5
    out_f = cv2.GaussianBlur(out_f, (k, k), 0)
    return (out_f * 255.0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_img", required=True, help="thu muc anh nguon")
    ap.add_argument("--src_lbl", required=True, help="thu muc label YOLO nguon")
    ap.add_argument("--dst", required=True, help="thu muc dich, se sinh ra dst/images va dst/labels")
    ap.add_argument("--severity_min", type=float, default=0.3)
    ap.add_argument("--severity_max", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dst_img = os.path.join(args.dst, "images")
    dst_lbl = os.path.join(args.dst, "labels")
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    paths = sorted(
        [p for p in glob.glob(os.path.join(args.src_img, "*"))
         if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    )
    if not paths:
        raise RuntimeError(f"Khong co anh trong {args.src_img}")

    random.seed(args.seed)
    for i, p in enumerate(tqdm(paths, desc="rainy-aug")):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print("skip:", p)
            continue
        sev = random.uniform(args.severity_min, args.severity_max)
        out = apply_rain(img, severity=sev, seed=args.seed + i)
        name = os.path.basename(p)
        cv2.imwrite(os.path.join(dst_img, name), out)
        # copy label
        stem = os.path.splitext(name)[0]
        src_lbl = os.path.join(args.src_lbl, stem + ".txt")
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, os.path.join(dst_lbl, stem + ".txt"))

    print(f"[make_rainy] Done. {len(paths)} images -> {args.dst}")


if __name__ == "__main__":
    main()
