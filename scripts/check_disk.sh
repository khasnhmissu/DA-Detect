#!/usr/bin/env bash
# =============================================================================
# check_disk.sh — phan tich disk usage cua repo + don nhanh.
#
# Dung:
#   bash scripts/check_disk.sh           # chi xem
#   bash scripts/check_disk.sh --clean   # xoa cache pip/conda + build artifacts
#   bash scripts/check_disk.sh --emergency  # xoa moi cache + zip dataset cu
# =============================================================================
set -e

MODE="info"
for arg in "$@"; do
  case "$arg" in
    --clean) MODE="clean" ;;
    --emergency) MODE="emergency" ;;
  esac
done

print_size() {
  local label="$1" path="$2"
  if [ -e "$path" ]; then
    local sz
    sz=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
    printf "  %-45s %s\n" "$label" "$sz"
  else
    printf "  %-45s (khong co)\n" "$label"
  fi
}

echo "==============================================================="
echo "DISK STATUS"
echo "==============================================================="
df -h / 2>/dev/null | head -2 || df -h .

echo ""
echo "=== /workspace/DA-Detect ==="
print_size "datasets/source_real/"        datasets/source_real
print_size "datasets/target_real/"        datasets/target_real
print_size "datasets/source_test/"        datasets/source_test
print_size "datasets/target_test/"        datasets/target_test
print_size "datasets/rainy_real/"         datasets/rainy_real
print_size "datasets/annotations/"        datasets/annotations
print_size "datasets/*.zip (zip thua)"    datasets/source_real.zip
print_size "  (target zip)"                datasets/target_real.zip
print_size "build/"                       build
print_size "maskrcnn_benchmark.egg-info/" maskrcnn_benchmark.egg-info
print_size "output/"                      output
print_size "*.log"                        run.log

echo ""
echo "=== Cache (ngoai repo) ==="
print_size "~/.cache/pip/"                "$HOME/.cache/pip"
print_size "~/.conda/pkgs/"               "$HOME/.conda/pkgs"
print_size "/opt/conda/pkgs/"             "/opt/conda/pkgs"
print_size "~/.torch/models/"             "$HOME/.torch/models"

# Conda env size
ENV_PATH=$(conda info --envs 2>/dev/null | awk '/da_detect/{print $NF}' | head -1)
if [ -n "$ENV_PATH" ] && [ -d "$ENV_PATH" ]; then
  echo ""
  echo "=== Conda env 'da_detect' ==="
  print_size "$ENV_PATH"                  "$ENV_PATH"
fi

if [ "$MODE" = "info" ]; then
  echo ""
  echo "------------------------------------------------------"
  echo "Don dep:  bash scripts/check_disk.sh --clean"
  echo "Khan cap: bash scripts/check_disk.sh --emergency"
  exit 0
fi

# =============================================================================
# Clean mode
# =============================================================================
echo ""
echo "==============================================================="
echo "[CLEAN] Don dep cache pip/conda + build artifacts"
echo "==============================================================="

echo "[1/5] pip cache purge..."
pip cache purge 2>/dev/null || true

echo "[2/5] conda clean -afy..."
conda clean -afy 2>/dev/null || true

echo "[3/5] xoa /tmp/pip-* va /tmp/tmp*..."
rm -rf /tmp/pip-* /tmp/tmp* 2>/dev/null || true

echo "[4/5] xoa build/ va egg-info/..."
rm -rf build maskrcnn_benchmark.egg-info

echo "[5/5] xoa __pycache__/..."
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

if [ "$MODE" = "emergency" ]; then
  echo ""
  echo "==============================================================="
  echo "[EMERGENCY] Xoa nhung thu lam tan disk lon"
  echo "==============================================================="

  echo "[1/3] Xoa zip thua trong datasets/..."
  rm -f datasets/*.zip 2>/dev/null || true

  echo "[2/3] Xoa pretrained R-50 cache (~98MB, se tu tai lai)..."
  rm -f "$HOME/.torch/models/R-50.pkl" 2>/dev/null || true

  echo "[3/3] Xoa intermediate ckpt trong output/..."
  if [ -d output ]; then
    for f in output/*/model_*.pth; do
      [ -f "$f" ] || continue
      base=$(basename "$f")
      [ "$base" = "model_final.pth" ] && continue
      stem="${base%.pth}"
      digits="${stem#model_}"
      if [[ "$digits" =~ ^[0-9]+$ ]]; then
        rm -f "$f"
      fi
    done
  fi
fi

echo ""
echo "==============================================================="
echo "DISK SAU CLEAN"
echo "==============================================================="
df -h / 2>/dev/null | head -2 || df -h .
