"""
Standalone evaluator — bypass test_net.py + maskrcnn_benchmark.engine.inference.
Tu load checkpoint, chay forward tren tap test, sinh bbox.json va in toan bo
COCO metrics (mAP, mAP50, AP_S/M/L) + per-class AP.

Dung:
  python tools/eval_standalone.py \
    --config-file configs/da_faster_rcnn/fusionda_source_only_c4.yaml \
    --weight ./output/source_only_c4/model_final.pth \
    --test-dataset fusionda_target_test_cocostyle \
    --gt-json datasets/annotations/target_test.json \
    --out-dir ./output/source_only_c4/inference/fusionda_target_test_cocostyle
"""
import argparse
import json
import os
import sys

# Monkey-patch numpy de tuong thich pycocotools <= 2.0.7 (van dung np.float deprecated).
# Phai chay TRUOC khi import pycocotools.
import numpy as np
if not hasattr(np, "float"):
    np.float = float  # type: ignore
if not hasattr(np, "int"):
    np.int = int  # type: ignore
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore

import torch
from tqdm import tqdm

# noqa: setup_environment phai import truoc moi thu cua maskrcnn_benchmark
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weight", required=True, help="path to model_final.pth")
    ap.add_argument("--test-dataset", required=True,
                    help="ten dataset key trong paths_catalog.py (vd: fusionda_target_test_cocostyle)")
    ap.add_argument("--gt-json", required=True, help="COCO GT JSON tuong ung")
    ap.add_argument("--out-dir", required=True, help="thu muc luu bbox.json + metrics.txt")
    return ap.parse_args()


@torch.no_grad()
def run_inference(model, data_loader, device):
    """
    Tra ve dict {image_id: BoxList prediction}.
    """
    model.eval()
    cpu = torch.device("cpu")
    results = {}
    for batch in tqdm(data_loader, desc="inference"):
        images, targets, image_ids = batch
        images = images.to(device)
        outputs = model(images)
        outputs = [o.to(cpu) for o in outputs]
        for img_id, pred in zip(image_ids, outputs):
            results[img_id] = pred
    return results


def predictions_to_coco(predictions, dataset):
    """
    Convert dict {image_id: BoxList} -> list[{image_id, category_id, bbox, score}]
    theo dinh dang COCO detection.
    """
    coco_results = []
    for image_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        original_id = dataset.id_to_img_map[image_id]
        info = dataset.get_img_info(image_id)
        prediction = prediction.resize((info["width"], info["height"])).convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        for k, box in enumerate(boxes):
            coco_results.append({
                "image_id": original_id,
                "category_id": mapped_labels[k],
                "bbox": box,
                "score": scores[k],
            })
    return coco_results


def coco_eval(gt_json, dt_data, out_path):
    """In du 12 chi so COCO + per-class. Ghi log xuong out_path neu khac None."""
    lines = []
    def echo(s=""):
        print(s)
        lines.append(s)

    coco_gt = COCO(gt_json)

    if not dt_data:
        echo("[WARN] khong co detection nao -> mAP = -1.")
        if out_path:
            with open(out_path, "w") as f:
                f.write("\n".join(lines))
        return

    coco_dt = coco_gt.loadRes(dt_data)

    echo("\n===== COCO EVAL (overall) =====")
    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.evaluate(); E.accumulate(); E.summarize()

    s = E.stats
    echo("\n===== Tom tat =====")
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
        echo(f"  {name}: {s[i]*100:.2f}")

    echo("\n===== AP per class (IoU=0.5:0.95) =====")
    cat_ids = coco_gt.getCatIds()
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    for cid in cat_ids:
        E2 = COCOeval(coco_gt, coco_dt, "bbox")
        E2.params.catIds = [cid]
        E2.evaluate(); E2.accumulate(); E2.summarize()
        echo(f"  {cats[cid]:<10s} mAP={E2.stats[0]*100:.2f}  mAP50={E2.stats[1]*100:.2f}")

    if out_path:
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\n[SAVED] {out_path}")


def main():
    args = parse_args()

    # ---- Load config ----
    cfg.merge_from_file(args.config_file)
    # Override TEST dataset
    cfg.DATASETS.TEST = (args.test_dataset,)
    # Override checkpoint
    cfg.MODEL.WEIGHT = args.weight
    cfg.freeze()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Build model + load weights ----
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_detection_model(cfg).to(device)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=args.out_dir)
    extra = checkpointer.load(cfg.MODEL.WEIGHT)
    print(f"[OK] loaded weights: {cfg.MODEL.WEIGHT}")

    # ---- Build dataloader ----
    data_loaders = make_data_loader(cfg, is_train=False, is_distributed=False)
    if not data_loaders:
        print("[ERROR] khong co dataloader nao tao ra!", file=sys.stderr)
        sys.exit(1)
    data_loader = data_loaders[0]
    dataset = data_loader.dataset
    print(f"[OK] dataset {args.test_dataset}: {len(dataset)} images")

    # ---- Inference ----
    predictions = run_inference(model, data_loader, device)
    print(f"[OK] sinh {len(predictions)} predictions")

    # ---- Save predictions.pth (raw BoxList format) ----
    pred_pth = os.path.join(args.out_dir, "predictions.pth")
    pred_list = [predictions[i] for i in sorted(predictions.keys())]
    torch.save(pred_list, pred_pth)
    print(f"[SAVED] {pred_pth}  ({os.path.getsize(pred_pth)} bytes)")

    # ---- Convert to COCO format + save bbox.json ----
    coco_dets = predictions_to_coco(predictions, dataset)
    bbox_json = os.path.join(args.out_dir, "bbox.json")
    with open(bbox_json, "w") as f:
        json.dump(coco_dets, f)
    print(f"[SAVED] {bbox_json}  ({len(coco_dets)} detections)")

    # ---- COCO eval ----
    metrics_txt = os.path.join(args.out_dir, "metrics.txt")
    coco_eval(args.gt_json, coco_dets, metrics_txt)


if __name__ == "__main__":
    main()
