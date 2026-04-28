#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh — chuyen YOLO -> COCO JSON va sinh mien phu tro rainy
# Dung: bash scripts/prepare_data.sh
# Gia dinh du lieu da tai ve qua scripts/download_data.sh
# =============================================================================
set -e

ROOT="${DATA_ROOT:-datasets}"
ANN="$ROOT/annotations"
mkdir -p "$ANN"

# Duong dan sau giai nen (dung tu cau truc gdown cua user)
SRC_TRAIN_IMG="$ROOT/source_real/source_real/train/images"
SRC_TRAIN_LBL="$ROOT/source_real/source_real/train/labels"
SRC_VAL_IMG="$ROOT/source_real/source_real/val/images"
SRC_VAL_LBL="$ROOT/source_real/source_real/val/labels"
TGT_TRAIN_IMG="$ROOT/target_real/target_real/train/images"
TGT_TRAIN_LBL="$ROOT/target_real/target_real/train/labels"
TGT_VAL_IMG="$ROOT/target_real/target_real/val/images"
TGT_VAL_LBL="$ROOT/target_real/target_real/val/labels"
SRC_TEST_IMG="$ROOT/source_test/source_test/val/images"
SRC_TEST_LBL="$ROOT/source_test/source_test/val/labels"
TGT_TEST_IMG="$ROOT/target_test/target_test/val/images"
TGT_TEST_LBL="$ROOT/target_test/target_test/val/labels"

check_dir() {
  if [ ! -d "$1" ]; then
    echo "[ERROR] Khong thay thu muc $1. Hay chay scripts/download_data.sh truoc." >&2
    exit 1
  fi
}
for d in "$SRC_TRAIN_IMG" "$SRC_TRAIN_LBL" "$TGT_TRAIN_IMG" "$TGT_TRAIN_LBL" \
         "$SRC_TEST_IMG" "$SRC_TEST_LBL" "$TGT_TEST_IMG" "$TGT_TEST_LBL"; do
  check_dir "$d"
done

echo "[1/8] source_train..."
python tools/yolo2coco.py --img_dir "$SRC_TRAIN_IMG" --lbl_dir "$SRC_TRAIN_LBL" --out "$ANN/source_train.json"

echo "[2/8] source_val..."
python tools/yolo2coco.py --img_dir "$SRC_VAL_IMG" --lbl_dir "$SRC_VAL_LBL" --out "$ANN/source_val.json"

echo "[3/8] target_train..."
python tools/yolo2coco.py --img_dir "$TGT_TRAIN_IMG" --lbl_dir "$TGT_TRAIN_LBL" --out "$ANN/target_train.json"

echo "[4/8] target_val..."
python tools/yolo2coco.py --img_dir "$TGT_VAL_IMG" --lbl_dir "$TGT_VAL_LBL" --out "$ANN/target_val.json"

echo "[5/8] source_test..."
python tools/yolo2coco.py --img_dir "$SRC_TEST_IMG" --lbl_dir "$SRC_TEST_LBL" --out "$ANN/source_test.json"

echo "[6/8] target_test..."
python tools/yolo2coco.py --img_dir "$TGT_TEST_IMG" --lbl_dir "$TGT_TEST_LBL" --out "$ANN/target_test.json"

echo "[7/8] Sinh mien phu tro rainy tu source_train..."
if [ ! -d "$ROOT/rainy_real/train/images" ]; then
  python tools/make_rainy.py --src_img "$SRC_TRAIN_IMG" --src_lbl "$SRC_TRAIN_LBL" --dst "$ROOT/rainy_real/train"
else
  echo "Da co $ROOT/rainy_real/train/images, bo qua."
fi

echo "[8/8] rainy_train JSON..."
python tools/yolo2coco.py \
  --img_dir "$ROOT/rainy_real/train/images" \
  --lbl_dir "$ROOT/rainy_real/train/labels" \
  --out     "$ANN/rainy_train.json"

echo ""
echo "[DONE] Annotation JSON da tao:"
ls -la "$ANN"
