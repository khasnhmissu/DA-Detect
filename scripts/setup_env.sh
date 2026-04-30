#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — cai toan bo moi truong cho DA-Detect tren Linux server (conda).
#
# Yeu cau driver NVIDIA ho tro CUDA >= 12.1 (driver >= 530).
# Su dung PyTorch 2.2.2 + cu121 (forward-compat voi system CUDA 12.4-12.8).
#
# Cach dung:
#   bash scripts/setup_env.sh                        # tao conda env 'da_detect' (mac dinh)
#   ENV_NAME=myenv bash scripts/setup_env.sh         # ten env khac
#   FORCE_RECREATE=1 bash scripts/setup_env.sh       # xoa env cu va tao lai
# =============================================================================
set -e

ENV_NAME="${ENV_NAME:-da_detect}"
PY_VER="${PY_VER:-3.10}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

echo "[1/7] Tao/su dung conda env '$ENV_NAME' (python $PY_VER)..."
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] khong tim thay conda." >&2
  echo "  Cai Miniconda truoc: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  TORCH_VER=$(conda run -n "$ENV_NAME" python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
  if [ "$FORCE_RECREATE" = "1" ] || [[ "$TORCH_VER" != 2.2.* ]]; then
    echo "Env '$ENV_NAME' co torch=$TORCH_VER (can 2.2.x). Xoa va tao lai..."
    conda env remove -y -n "$ENV_NAME"
    conda create -y -n "$ENV_NAME" python="$PY_VER"
  else
    echo "Env '$ENV_NAME' da co torch $TORCH_VER, kich hoat lai."
  fi
else
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"

echo "[2/7] Cai PyTorch 2.2.2 + CUDA build tools 12.1 (toi thieu, tiet kiem disk)..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir "setuptools<70" wheel

# Cai NVCC + CUDA libraries dev (du cublas, curand, cusolver, cusparse cho PyTorch
# build extension). Trung gian giua cuda-toolkit (3GB) va minimal (fail vi thieu cublas_v2.h).
echo "  Cai cuda-nvcc + libraries-dev (~1.2GB, mat 2-3 phut)..."
conda install -y -n "$ENV_NAME" -c "nvidia/label/cuda-12.1.0" \
  cuda-nvcc cuda-cudart-dev cuda-cccl cuda-libraries-dev

# QUAN TRONG: cai gcc/g++ 12 trong env. CUDA 12.1 nvcc tu choi gcc >= 13
# (system Ubuntu 24.04 / Debian 13 mac dinh la gcc-13/14). Khong cai => build fail.
echo "  Cai gcc-12/g++-12 (host compiler cho nvcc 12.1)..."
conda install -y -n "$ENV_NAME" -c conda-forge \
  "gcc_linux-64=12.*" "gxx_linux-64=12.*" "sysroot_linux-64>=2.17"

# Don dep cache conda ngay sau install (tiet kiem 2-3GB)
conda clean -afy

# Cai PyTorch khong cache (tiet kiem ~2GB pip cache)
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[3/7] Cai cac goi Python phu thuoc (no cache)..."
pip install --no-cache-dir \
  "numpy<2" "protobuf<5" \
  ninja yacs==0.1.8 cython matplotlib tqdm \
  "opencv-python<4.10" timm scipy h5py \
  "pillow<11" "pycocotools>=2.0.8" gdown tensorboardX

echo "[4/7] Patch legacy C++/CUDA code cho PyTorch 2.x..."
bash scripts/patch_csrc.sh

echo "[5/7] Build maskrcnn_benchmark extension..."
# Cac compute capability pho bien (Volta -> Hopper).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0+PTX}"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# QUAN TRONG: ep dung nvcc trong conda env (12.1) thay cho system (co the la 13.0+).
# CONDA_PREFIX la duong dan env active.
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version 2>/dev/null | head -n4 || true

# QUAN TRONG: ep nvcc dung gcc-12 trong env (khong dung system gcc 13+).
# Conda goi gcc-12 la x86_64-conda-linux-gnu-gcc; 1 so phien ban activation script
# tu set CC/CXX, nhung khong tin cay khi pip --no-build-isolation. Set thu cong.
CONDA_CC="$(ls $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc 2>/dev/null || true)"
CONDA_CXX="$(ls $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ 2>/dev/null || true)"
if [ -z "$CONDA_CC" ] || [ -z "$CONDA_CXX" ]; then
  echo "[ERROR] khong tim thay gcc-12/g++-12 trong $CONDA_PREFIX/bin." >&2
  echo "  conda install -n $ENV_NAME -c conda-forge gcc_linux-64=12 gxx_linux-64=12" >&2
  exit 1
fi
export CC="$CONDA_CC"
export CXX="$CONDA_CXX"
export CUDAHOSTCXX="$CONDA_CXX"   # nvcc -ccbin se dung cai nay
echo "CC  = $CC ($($CC -dumpversion))"
echo "CXX = $CXX ($($CXX -dumpversion))"

# Xoa build cu de tranh cache mismatch
rm -rf build maskrcnn_benchmark.egg-info
pip install --no-build-isolation -e .

echo "[6/7] Don dep cache cuoi cung..."
# Tat ca cache khong can thiet sau khi install xong
pip cache purge 2>/dev/null || true
conda clean -afy 2>/dev/null || true
rm -rf /tmp/pip-* /tmp/tmp* 2>/dev/null || true
echo "  Disk usage cua env:"
du -sh "$(conda info --envs | awk -v env="$ENV_NAME" '$1==env {print $NF}')" 2>/dev/null || true

echo "[7/7] Smoke test..."
python - <<'PY'
import torch
print("torch     :", torch.__version__)
print("cuda      :", torch.version.cuda)
print("device cnt:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0  :", torch.cuda.get_device_name(0))
    print("compute   :", torch.cuda.get_device_capability(0))
from maskrcnn_benchmark.config import cfg
print("config OK :", cfg.MODEL.META_ARCHITECTURE)
from maskrcnn_benchmark import _C
print("ext OK    :", [a for a in dir(_C) if not a.startswith('_')][:5])
if torch.cuda.is_available():
    boxes  = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]], device="cuda")
    scores = torch.tensor([0.9, 0.8], device="cuda")
    keep = _C.nms(boxes, scores, 0.5)
    print("nms OK    :", keep)
PY

echo ""
echo "[DONE] Moi truong '$ENV_NAME' da san sang."
echo "De kich hoat lai: conda activate $ENV_NAME"
