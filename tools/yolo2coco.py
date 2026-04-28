"""
Convert YOLO-format dataset (cls cx cy w h, normalized) sang COCO JSON
yeu cau boi DA-Detect / maskrcnn-benchmark.

Sinh polygon gia = hinh chu nhat cua bbox vi COCODataset yeu cau segmentation
du chi train detection.

Dung:
  python tools/yolo2coco.py \
    --img_dir datasets/source_real/source_real/train/images \
    --lbl_dir datasets/source_real/source_real/train/labels \
    --out     datasets/annotations/source_train.json
"""
import argparse, glob, json, os
from PIL import Image

# YOLO label -> COCO category_id (COCO bat dau tu 1, 0 la background)
CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
]
YOLO2COCO = {0: 1, 1: 2}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def convert(img_dir, lbl_dir, out_json):
    images, annotations = [], []
    img_id, ann_id = 0, 0

    paths = sorted(
        [p for p in glob.glob(os.path.join(img_dir, "*"))
         if p.lower().endswith(IMG_EXT)]
    )
    if not paths:
        raise RuntimeError(f"Khong tim thay anh trong {img_dir}")

    for img_path in paths:
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]
        with Image.open(img_path) as im:
            W, H = im.size
        images.append({
            "id": img_id,
            "file_name": name,
            "width": W,
            "height": H,
        })

        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    if cls not in YOLO2COCO:
                        continue
                    cx, cy, bw_n, bh_n = map(float, parts[1:5])
                    x = (cx - bw_n / 2.0) * W
                    y = (cy - bh_n / 2.0) * H
                    bw = bw_n * W
                    bh = bh_n * H
                    # clamp
                    x = max(0.0, min(W - 1.0, x))
                    y = max(0.0, min(H - 1.0, y))
                    bw = max(1.0, min(W - x, bw))
                    bh = max(1.0, min(H - y, bh))
                    # polygon gia: 4 goc hcn
                    seg = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": YOLO2COCO[cls],
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "segmentation": seg,
                        "iscrowd": 0,
                    })
                    ann_id += 1
        img_id += 1

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": CATEGORIES,
        }, f)
    print(f"[yolo2coco] {out_json}: {len(images)} images, {len(annotations)} annotations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--lbl_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert(args.img_dir, args.lbl_dir, args.out)
