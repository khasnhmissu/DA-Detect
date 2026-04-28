#!/usr/bin/env bash
# =============================================================================
# evaluate.sh — eval model da train tren 2 tap test (source_test va target_test)
# Dung tools/eval_standalone.py (bypass test_net.py de tranh silent fail).
#
# Dung:
#   bash scripts/evaluate.sh                    # eval source_only_c4 (mac dinh)
#   bash scripts/evaluate.sh source_only_c4
#   bash scripts/evaluate.sh source_only_fpn
#   bash scripts/evaluate.sh triplet_c4
#   bash scripts/evaluate.sh triplet_fpn
#   bash scripts/evaluate.sh all                # eval ca 4
# =============================================================================
set -e

MODE="${1:-source_only_c4}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

config_of() {
  case "$1" in
    source_only_c4)  echo "configs/da_faster_rcnn/fusionda_source_only_c4.yaml" ;;
    source_only_fpn) echo "configs/da_faster_rcnn/fusionda_source_only_fpn.yaml" ;;
    triplet_c4)      echo "configs/da_faster_rcnn/fusionda_triplet_c4.yaml" ;;
    triplet_fpn)     echo "configs/da_faster_rcnn/fusionda_triplet_fpn.yaml" ;;
    *) return 1 ;;
  esac
}

eval_model() {
  local tag="$1"
  local cfg
  cfg="$(config_of "$tag")"
  local weight="./output/${tag}/model_final.pth"
  if [ ! -f "$weight" ]; then
    weight="$(ls -t ./output/${tag}/model_*.pth 2>/dev/null | head -n 1 || true)"
  fi
  if [ -z "$weight" ] || [ ! -f "$weight" ]; then
    echo "[SKIP EVAL] khong co checkpoint trong ./output/${tag}/" >&2
    return 1
  fi
  echo "[$tag] checkpoint: $weight"

  for split in target_test source_test; do
    ds="fusionda_${split}_cocostyle"
    out_dir="./output/${tag}/inference/${ds}"
    echo ""
    echo "==================================================================="
    echo "EVAL  model=${tag}   test=${split}"
    echo "==================================================================="
    python tools/eval_standalone.py \
      --config-file "$cfg" \
      --weight     "$weight" \
      --test-dataset "$ds" \
      --gt-json    "datasets/annotations/${split}.json" \
      --out-dir    "$out_dir"
  done
}

case "$MODE" in
  source_only_c4|source_only_fpn|triplet_c4|triplet_fpn)
    eval_model "$MODE" ;;
  all)
    for tag in source_only_c4 source_only_fpn triplet_c4 triplet_fpn; do
      eval_model "$tag" || true
    done ;;
  *)
    echo "Usage: $0 {source_only_c4|source_only_fpn|triplet_c4|triplet_fpn|all}" >&2
    exit 1 ;;
esac

echo ""
echo "[DONE] Ket qua tai:"
ls -la ./output/*/inference/*/metrics.txt 2>/dev/null || true
