#!/usr/bin/env bash
# =============================================================================
# download_data.sh — tai + giai nen 4 bo du lieu FusionDA tu Google Drive,
# sau do xoa .zip de tiet kiem disk.
# Dung: bash scripts/download_data.sh
#
# Bien moi truong:
#   DATA_ROOT=datasets  (mac dinh)
#   KEEP_ZIP=1          giu lai cac file .zip (bo qua buoc xoa)
# =============================================================================
set -e

ROOT="${DATA_ROOT:-datasets}"
KEEP_ZIP="${KEEP_ZIP:-0}"
mkdir -p "$ROOT"
cd "$ROOT"

if ! command -v gdown >/dev/null 2>&1; then
  pip install gdown
fi

# Helper: tai (neu chua co), giai nen (neu chua co), roi xoa .zip (tru khi KEEP_ZIP=1)
fetch_one() {
  local label="$1" gdrive_id="$2" zip_name="$3" dst_dir="$4"
  echo "$label"
  if [ -d "$dst_dir" ]; then
    echo "  -> $dst_dir/ da co, bo qua download."
  else
    [ -f "$zip_name" ] || gdown "$gdrive_id" -O "$zip_name"
    unzip -q "$zip_name" -d "$dst_dir"
  fi
  if [ "$KEEP_ZIP" != "1" ] && [ -f "$zip_name" ]; then
    rm -f "$zip_name"
    echo "  -> da xoa $zip_name"
  fi
}

fetch_one "[1/4] source_real (Cityscapes 2800 train + 175 val)..." \
  1p_pJjOIngZ6ylbPSs2fD3Eo7QR8q5C2b source_real.zip source_real

fetch_one "[2/4] target_real (Foggy Cityscapes 2800 + 175)..." \
  1Y8GA79Hng9MD6llcSi3ys4NgWouxTnZb target_real.zip target_real

fetch_one "[3/4] source_test (500 anh source)..." \
  1zK5SqDdXekeFMYNtONSOcavMW2vfhJBq source_test.zip source_test

fetch_one "[4/4] target_test (500 anh target)..." \
  1TRnmVqgujZucqvwLDCvKA55z_j3usUtm target_test.zip target_test

cd ..
echo ""
echo "[DONE] Du lieu da tai ve va giai nen:"
find "$ROOT" -maxdepth 4 -type d | sort
echo ""
echo "Disk usage:"
du -sh "$ROOT"
