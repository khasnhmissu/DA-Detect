#!/usr/bin/env bash
# =============================================================================
# patch_csrc.sh — patch tat ca legacy C++/CUDA code de build duoc voi PyTorch 2.x.
# Idempotent: chay nhieu lan deu OK.
# =============================================================================
set -e

CSRC=maskrcnn_benchmark/csrc
[ -d "$CSRC" ] || { echo "khong thay $CSRC, hay chay tu repo root"; exit 1; }

echo "[patch] Step 1: AT_CHECK -> TORCH_CHECK"
perl -i -pe 's/\bAT_CHECK\b/TORCH_CHECK/g' \
  $CSRC/cuda/*.cu $CSRC/cuda/*.h $CSRC/cpu/*.cpp $CSRC/cpu/*.h $CSRC/*.h 2>/dev/null || true

echo "[patch] Step 2: bo header THC, them ATen/Atomic.cuh + CUDACachingAllocator"
# Xoa cac dong include THC (THC.h, THCDeviceUtils.cuh, THCNumerics.cuh,...)
perl -i -ne 'print unless /^\s*#\s*include\s*<THC\/(?!THCAtomics)/' $CSRC/cuda/*.cu
# Doi THCAtomics.cuh -> ATen/cuda/Atomic.cuh
perl -i -pe 's|<THC/THCAtomics\.cuh>|<ATen/cuda/Atomic.cuh>|g' $CSRC/cuda/*.cu

echo "[patch] Step 3: them include cho CUDACachingAllocator + Exception (chi vao file dung den)"
for f in $CSRC/cuda/nms.cu; do
  if ! grep -q "CUDACachingAllocator.h" "$f"; then
    perl -i -pe 's|(<ATen/cuda/CUDAContext.h>)|$1\n#include <c10/cuda/CUDACachingAllocator.h>\n#include <c10/cuda/CUDAException.h>|' "$f"
  fi
done

echo "[patch] Step 4a: THCudaCheck -> AT_CUDA_CHECK (tat ca file .cu)"
perl -i -pe 's|\bTHCudaCheck\b|AT_CUDA_CHECK|g' $CSRC/cuda/*.cu

echo "[patch] Step 4b: THCState/THCudaMalloc/Free (chi nms.cu su dung)"
for f in $CSRC/cuda/nms.cu; do
  perl -i -pe 's|THCState\s*\*\s*state\s*=\s*at::globalContext\(\)\.lazyInitCUDA\(\);|// state init removed|g' "$f"
  perl -i -pe 's|THCudaMalloc\s*\(\s*state\s*,\s*([^)]+)\)|c10::cuda::CUDACachingAllocator::raw_alloc($1)|g' "$f"
  perl -i -pe 's|THCudaFree\s*\(\s*state\s*,\s*([^)]+)\)|c10::cuda::CUDACachingAllocator::raw_delete($1)|g' "$f"
done

# THCCeilDiv co the xuat hien o nhieu file -> them macro va thay
echo "[patch] Step 5: THCCeilDiv -> macro local"
for f in $CSRC/cuda/*.cu; do
  if grep -q "THCCeilDiv" "$f"; then
    if ! grep -q "MASKRCNN_CEIL_DIV" "$f"; then
      # Chen macro dau file (sau dong include cuoi cung)
      perl -i -0pe 's|(\#include[^\n]*\n)(?!\#include)|$1\n#define MASKRCNN_CEIL_DIV(a, b) (((a) + (b) - 1) / (b))\n|m' "$f"
    fi
    perl -i -pe 's|\bTHCCeilDiv\b|MASKRCNN_CEIL_DIV|g' "$f"
  fi
done

echo "[patch] Step 6: tensor.type().is_cuda() -> tensor.is_cuda()"
perl -i -pe 's|\.type\(\)\.is_cuda\(\)|\.is_cuda()|g' \
  $CSRC/cuda/*.cu $CSRC/cpu/*.cpp $CSRC/*.h 2>/dev/null || true

echo "[patch] Step 7: tensor.data<T>() -> tensor.data_ptr<T>()"
perl -i -pe 's|\.data<([^>]+)>\(\)|.data_ptr<$1>()|g' \
  $CSRC/cuda/*.cu $CSRC/cpu/*.cpp $CSRC/*.h 2>/dev/null || true

echo "[patch] Step 7b: fix Tensor.type() deprecated trong AT_DISPATCH va so sanh"
# AT_DISPATCH_xxx(tensor.type(), ...) -> AT_DISPATCH_xxx(tensor.scalar_type(), ...)
perl -i -pe 's|(AT_DISPATCH_[A-Z_]+\s*\(\s*[A-Za-z_][A-Za-z0-9_]*)\.type\(\)|$1.scalar_type()|g' \
  $CSRC/cuda/*.cu $CSRC/cpu/*.cpp 2>/dev/null || true
# tensor.type() == other.type() -> tensor.scalar_type() == other.scalar_type()
perl -i -pe 's|([A-Za-z_][A-Za-z0-9_]*)\.type\(\)\s*==\s*([A-Za-z_][A-Za-z0-9_]*)\.type\(\)|$1.scalar_type() == $2.scalar_type()|g' \
  $CSRC/cuda/*.cu $CSRC/cpu/*.cpp $CSRC/*.h 2>/dev/null || true

echo "[patch] Step 8: torch._six.PY37 -> True (idempotent, file da fix san)"
perl -i -pe 's|torch\._six\.PY37|True|g' \
  maskrcnn_benchmark/utils/imports.py \
  maskrcnn_benchmark/utils/c2_model_loading.py 2>/dev/null || true

# Step 9: cac fix khac da commit truc tiep vao file:
# - maskrcnn_benchmark/utils/imports.py            (bo torch._six.PY37 check)
# - maskrcnn_benchmark/utils/c2_model_loading.py   (bo torch._six.PY37 check)
# - maskrcnn_benchmark/utils/model_zoo.py          (download_url_to_file public API)
# - maskrcnn_benchmark/data/datasets/evaluation/coco/coco_eval.py  ('is 1' -> '== 1')
# - maskrcnn_benchmark/modeling/rpn/anchor_generator.py            (np.float -> np.float64)
# - maskrcnn_benchmark/modeling/detector/generalized_rcnn.py       (source-only branch + B-aware slicing)
# - maskrcnn_benchmark/modeling/da_heads/da_heads.py               (FPN-aware avgpool)
# - maskrcnn_benchmark/modeling/da_heads/loss.py                   (multi-level FPN BCE)
# - maskrcnn_benchmark/modeling/rpn/inference.py                   (always add GT proposals)
# - maskrcnn_benchmark/utils/checkpoint.py                         (auto-cleanup intermediate)

echo "[patch] Done."
echo ""
echo "Verify (khong nen con dong nao xuat hien):"
echo "  AT_CHECK     : $(grep -rc '\bAT_CHECK\b' $CSRC | grep -v ':0$' | wc -l)"
echo "  THC/THC.h    : $(grep -rc '<THC/THC.h>' $CSRC | grep -v ':0$' | wc -l)"
echo "  THCudaCheck  : $(grep -rc '\bTHCudaCheck\b' $CSRC | grep -v ':0$' | wc -l)"
echo "  THCCeilDiv   : $(grep -rc '\bTHCCeilDiv\b' $CSRC | grep -v ':0$' | wc -l)"
echo "  .type().is_cuda(): $(grep -rc '\.type()\.is_cuda()' $CSRC | grep -v ':0$' | wc -l)"
echo "  .data<T>()   : $(grep -rE '\.data<[A-Za-z_]+>\(\)' $CSRC -rc | grep -v ':0$' | wc -l)"
echo "  AT_DISPATCH(.type()): $(grep -rE 'AT_DISPATCH_[A-Z_]+\s*\(\s*[A-Za-z_]+\.type\(\)' $CSRC -rc | grep -v ':0$' | wc -l)"
