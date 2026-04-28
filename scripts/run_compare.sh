#!/usr/bin/env bash
# =============================================================================
# run_compare.sh — chay 4 thi nghiem so sanh:
#   {source_only, triplet} x {R50-C4, R50-FPN}
# Moi model sau khi train se duoc eval tren CA source_test va target_test,
# in day du COCO mAP, mAP50, mAP50-95, APs/m/l, AR + per-class.
#
# Dung:
#   bash scripts/run_compare.sh             # full: cai env + tai data + train 4 + eval
#   bash scripts/run_compare.sh train_only  # bo qua setup, chi train + eval
#   bash scripts/run_compare.sh eval_only   # chi eval (4 model phai da co .pth)
# =============================================================================
set -e

MODE="${1:-full}"
ENV_NAME="${ENV_NAME:-da_detect}"

# ---- Activate env ----
if [ "$MODE" = "full" ]; then
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    bash scripts/setup_env.sh
  fi
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---- Data prep (only in full mode) ----
if [ "$MODE" = "full" ]; then
  bash scripts/download_data.sh
  bash scripts/prepare_data.sh
fi

# ---- 4 thi nghiem ----
# tag=ten thu muc output, cfg=config yaml, script=train_net.py hoac train_net_triplet.py
declare -a EXPS=(
  "source_only_c4|configs/da_faster_rcnn/fusionda_source_only_c4.yaml|train_net.py"
  "source_only_fpn|configs/da_faster_rcnn/fusionda_source_only_fpn.yaml|train_net.py"
  "triplet_c4|configs/da_faster_rcnn/fusionda_triplet_c4.yaml|train_net_triplet.py"
  "triplet_fpn|configs/da_faster_rcnn/fusionda_triplet_fpn.yaml|train_net_triplet.py"
)

train_one() {
  local tag="$1" cfg="$2" script="$3"
  if [ -f "./output/${tag}/model_final.pth" ]; then
    echo "[SKIP TRAIN] da co ./output/${tag}/model_final.pth"
    return 0
  fi
  echo ""
  echo "###########################################################"
  echo "# TRAIN  ${tag}  ($script)"
  echo "###########################################################"
  python "tools/${script}" --config-file "$cfg" --skip-test
}

eval_one() {
  local tag="$1" cfg="$2"
  local weight="./output/${tag}/model_final.pth"
  if [ ! -f "$weight" ]; then
    weight="$(ls -t ./output/${tag}/model_*.pth 2>/dev/null | head -n 1 || true)"
  fi
  if [ -z "$weight" ] || [ ! -f "$weight" ]; then
    echo "[SKIP EVAL] khong co checkpoint trong ./output/${tag}/" >&2
    return 1
  fi
  for split in target_test source_test; do
    ds="fusionda_${split}_cocostyle"
    out_dir="./output/${tag}/inference/${ds}"
    echo ""
    echo "==========================================================="
    echo "EVAL  model=${tag}   test=${split}"
    echo "==========================================================="
    # eval_standalone.py: bypass test_net.py, sinh predictions.pth + bbox.json
    # + metrics.txt trong 1 lan chay. Robust hon.
    python tools/eval_standalone.py \
      --config-file "$cfg" \
      --weight     "$weight" \
      --test-dataset "$ds" \
      --gt-json    "datasets/annotations/${split}.json" \
      --out-dir    "$out_dir"
  done
}

# ---- Train tat ca (skip neu da co .pth) ----
if [ "$MODE" != "eval_only" ]; then
  for entry in "${EXPS[@]}"; do
    IFS='|' read -r tag cfg script <<< "$entry"
    train_one "$tag" "$cfg" "$script"
  done
fi

# ---- Eval tat ca ----
for entry in "${EXPS[@]}"; do
  IFS='|' read -r tag cfg script <<< "$entry"
  eval_one "$tag" "$cfg"
done

# ---- Tom tat cuoi ----
echo ""
echo "###########################################################"
echo "# TOM TAT KET QUA (xem ./output/<tag>/inference/*/metrics.txt)"
echo "###########################################################"
for entry in "${EXPS[@]}"; do
  IFS='|' read -r tag cfg script <<< "$entry"
  for split in target_test source_test; do
    ds="fusionda_${split}_cocostyle"
    f="./output/${tag}/inference/${ds}/metrics.txt"
    if [ -f "$f" ]; then
      echo ""
      echo "--- ${tag}  on  ${split} ---"
      grep -E "mAP|AP_small|AP_medium|AP_large|^  person|^  car" "$f" || cat "$f"
    fi
  done
done
echo ""
echo "[ALL DONE]"
