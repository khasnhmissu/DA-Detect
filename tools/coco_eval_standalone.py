"""
Doc bbox.json tu inference/ va in day du chi so COCO:
  mAP (AP @ IoU=0.5:0.95), mAP50, mAP75, APs, APm, APl, AR @1/10/100
+ AP per class (person / car).

Dung:
  python tools/coco_eval_standalone.py \
    --gt  datasets/annotations/target_test.json \
    --dt  output/da_triplet/inference/fusionda_target_test_cocostyle/bbox.json
"""
import argparse
import json
import os

# Monkey-patch np.float cho pycocotools <= 2.0.7
import numpy as np
if not hasattr(np, "float"):
    np.float = float  # type: ignore
if not hasattr(np, "int"):
    np.int = int  # type: ignore
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="COCO GT annotation JSON")
    ap.add_argument("--dt", required=True, help="Detection bbox.json")
    args = ap.parse_args()

    if not os.path.exists(args.gt):
        raise FileNotFoundError(args.gt)
    if not os.path.exists(args.dt):
        raise FileNotFoundError(args.dt)

    coco_gt = COCO(args.gt)
    with open(args.dt) as f:
        dt_data = json.load(f)
    if len(dt_data) == 0:
        print("[WARN] file detection rong.")
        return
    coco_dt = coco_gt.loadRes(dt_data)

    print("\n===== COCO EVAL (overall) =====")
    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.evaluate(); E.accumulate(); E.summarize()

    stats = E.stats  # 12 numbers
    print("\n===== Tom tat =====")
    keys = [
        ("mAP (0.5:0.95)", 0),
        ("mAP50         ", 1),
        ("mAP75         ", 2),
        ("AP_small      ", 3),
        ("AP_medium     ", 4),
        ("AP_large      ", 5),
        ("AR@1          ", 6),
        ("AR@10         ", 7),
        ("AR@100        ", 8),
        ("AR_small      ", 9),
        ("AR_medium     ", 10),
        ("AR_large      ", 11),
    ]
    for name, i in keys:
        print(f"  {name}: {stats[i]*100:.2f}")

    # per-class
    print("\n===== AP per class (IoU=0.5:0.95) =====")
    cat_ids = coco_gt.getCatIds()
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    for cid in cat_ids:
        E2 = COCOeval(coco_gt, coco_dt, "bbox")
        E2.params.catIds = [cid]
        E2.evaluate(); E2.accumulate(); E2.summarize()
        print(f"  {cats[cid]:<10s} mAP={E2.stats[0]*100:.2f}  mAP50={E2.stats[1]*100:.2f}")


if __name__ == "__main__":
    main()
