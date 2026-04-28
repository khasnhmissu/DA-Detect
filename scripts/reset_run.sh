#!/usr/bin/env bash
# =============================================================================
# reset_run.sh — xoa MOI ket qua train/eval cu de chay lai tu dau.
# KHONG xoa env conda, source data, build extension.
#
# Dung:
#   bash scripts/reset_run.sh           # hoi confirm truoc khi xoa
#   bash scripts/reset_run.sh --yes     # bo qua confirm
# =============================================================================
set -e

YES=0
for arg in "$@"; do
  [ "$arg" = "--yes" ] && YES=1
done

echo "==============================================================="
echo "RESET RUN — se xoa cac thu sau:"
echo "==============================================================="
echo "  - output/                          (toan bo checkpoint, eval, log)"
echo "  - datasets/annotations/            (regen tu YOLO bang prepare_data.sh)"
echo "  - datasets/rainy_real/             (regen tu source bang make_rainy.py)"
echo "  - run.log, triplet_*.log           (log file cu)"
echo ""
echo "GIU LAI:"
echo "  - conda env 'da_detect'            (van ngon)"
echo "  - datasets/source_real/, target_real/, source_test/, target_test/"
echo "  - build/, maskrcnn_benchmark.egg-info  (extension da build)"
echo "  - ~/.torch/models/R-50.pkl         (pretrained cache, tranh tai lai)"
echo ""

if [ "$YES" != "1" ]; then
  read -p "Tiep tuc? (y/N) " ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "Huy."; exit 0 ;;
  esac
fi

echo ""
echo "[1/4] Xoa output/..."
rm -rf output

echo "[2/4] Xoa datasets/annotations/ va rainy_real/..."
rm -rf datasets/annotations datasets/rainy_real

echo "[3/4] Xoa log files..."
rm -f run.log run.pid triplet_c4.log triplet_fpn.log

echo "[4/4] Don dep cache..."
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "[DONE] Da reset. Bay gio chay lai tu dau:"
echo "  bash scripts/prepare_data.sh        # regen annotations + rainy"
echo "  bash scripts/run_compare.sh         # train + eval 4 thi nghiem"
