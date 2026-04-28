#!/usr/bin/env bash
# =============================================================================
# cleanup_disk.sh — don dep disk sau khi train/eval xong.
#
# Dung:
#   bash scripts/cleanup_disk.sh             # don dep an toan (mac dinh)
#   bash scripts/cleanup_disk.sh --aggressive # them: xoa rainy_real, pretrained cache
#   bash scripts/cleanup_disk.sh --dry-run   # chi xem se xoa gi, khong thuc su xoa
#   bash scripts/cleanup_disk.sh --inference # chi xoa inference_progress (khi eval lai nhieu lan)
# =============================================================================
set -e

DRY=0
MODE="safe"

for arg in "$@"; do
  case "$arg" in
    --dry-run)    DRY=1 ;;
    --aggressive) MODE="aggressive" ;;
    --inference)  MODE="inference" ;;
  esac
done

run() {
  echo "$@"
  if [ "$DRY" != "1" ]; then
    eval "$@"
  fi
}

print_header() {
  echo ""
  echo "==============================================================="
  echo "$1"
  echo "==============================================================="
}

print_size() {
  local label="$1" path="$2"
  if [ -e "$path" ]; then
    local sz
    sz=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
    printf "  %-40s %s\n" "$label" "$sz"
  fi
}

print_header "DISK USAGE TRUOC CLEANUP"
print_size "output/ (toan bo)"        "output"
for tag in source_only_c4 source_only_fpn triplet_c4 triplet_fpn; do
  if [ -d "output/$tag" ]; then
    print_size "  output/$tag/"        "output/$tag"
    n_int=$(ls output/$tag/model_*.pth 2>/dev/null | grep -v model_final.pth | wc -l)
    echo "    └── intermediate ckpts: $n_int file"
  fi
done
print_size "datasets/rainy_real/"     "datasets/rainy_real"
print_size "~/.torch/models/"         "$HOME/.torch/models"
print_size "build/"                   "build"
print_size "dist/"                    "dist"

# ============================================================
# 1. Inference-only: xoa eval_progress folders
# ============================================================
if [ "$MODE" = "inference" ]; then
  print_header "[INFERENCE MODE] Xoa inference_progress/"
  for tag in source_only_c4 source_only_fpn triplet_c4 triplet_fpn; do
    if [ -d "output/$tag/inference_progress" ]; then
      run "rm -rf output/$tag/inference_progress"
    fi
  done
  exit 0
fi

# ============================================================
# 2. Safe mode (mac dinh): xoa intermediate checkpoints
# ============================================================
print_header "Xoa intermediate checkpoints (model_XXXXXXX.pth, giu lai model_final.pth)"
for tag in source_only_c4 source_only_fpn triplet_c4 triplet_fpn; do
  if [ -d "output/$tag" ]; then
    for f in output/$tag/model_*.pth; do
      [ -f "$f" ] || continue
      base=$(basename "$f")
      if [ "$base" = "model_final.pth" ]; then continue; fi
      # Chi match model_<digits>.pth
      stem="${base%.pth}"
      digits="${stem#model_}"
      if [[ "$digits" =~ ^[0-9]+$ ]]; then
        run "rm -f '$f'"
      fi
    done
    # Xoa luon last_checkpoint pointer (de eval_standalone tu tim model_final.pth)
    if [ -f "output/$tag/last_checkpoint" ]; then
      run "rm -f output/$tag/last_checkpoint"
    fi
  fi
done

print_header "Xoa cache build (build/, *.egg-info, __pycache__)"
run "rm -rf build maskrcnn_benchmark.egg-info"
run "find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true"

# ============================================================
# 3. Aggressive: them xoa rainy_real, pretrained cache, zip cu
# ============================================================
if [ "$MODE" = "aggressive" ]; then
  print_header "[AGGRESSIVE] Xoa rainy_real (~5.6GB)"
  echo "  rainy_real co the regen tu source_real bang tools/make_rainy.py"
  if [ -d "datasets/rainy_real" ]; then
    run "rm -rf datasets/rainy_real"
  fi
  if [ -f "datasets/annotations/rainy_train.json" ]; then
    run "rm -f datasets/annotations/rainy_train.json"
  fi

  print_header "[AGGRESSIVE] Xoa pretrained cache (~98MB)"
  echo "  R-50.pkl se duoc tu dong tai lai khi train mot model moi"
  if [ -f "$HOME/.torch/models/R-50.pkl" ]; then
    run "rm -f '$HOME/.torch/models/R-50.pkl'"
  fi

  print_header "[AGGRESSIVE] Xoa zip cu trong datasets/"
  for f in datasets/*.zip; do
    [ -f "$f" ] && run "rm -f '$f'"
  done
fi

print_header "DISK USAGE SAU CLEANUP"
print_size "output/ (toan bo)"        "output"
print_size "datasets/rainy_real/"     "datasets/rainy_real"
print_size "~/.torch/models/"         "$HOME/.torch/models"
df -h / | head -2

if [ "$DRY" = "1" ]; then
  echo ""
  echo "[DRY-RUN] Khong co file nao bi xoa. Chay lai khong co --dry-run de thuc su xoa."
fi
