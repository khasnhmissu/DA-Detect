# DA-Detect làm baseline so sánh với FusionDA

Hướng dẫn chạy DA-Detect (WACV 2023) trên bộ dữ liệu FusionDA (Cityscapes → Foggy Cityscapes, 2 class `person` / `car`) để sinh số liệu so sánh với pipeline [FusionDA](https://github.com/khasnhmissu/FusionDA).

Toàn bộ đóng gói trong 5 script bash và 3 config yaml. Chỉ cần Linux + 1 GPU (≥8GB VRAM), Miniconda.

---

## 0. Chuẩn bị máy

- Linux (Ubuntu 18.04/20.04/22.04 đều OK)
- 1 GPU NVIDIA, driver hỗ trợ **CUDA 11.3** (driver ≥ 465)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) đã cài
- ~30 GB đĩa trống (dataset + checkpoint)

Clone repo và vào thư mục:

```bash
git clone <your-fork-url> DA-Detect
cd DA-Detect
chmod +x scripts/*.sh
```

---

## 1. Chạy nhanh nhất — một lệnh duy nhất

```bash
bash scripts/run_all.sh triplet
```

Lệnh trên sẽ **tự** làm đầy đủ: cài env → tải data → convert → sinh rainy → train → eval. Kết quả mAP COCO chi tiết in ra cuối cùng.

Muốn chạy cả 3 mô hình (source-only + DA Faster + Triplet) để lập bảng so sánh:

```bash
bash scripts/run_all.sh all
```

Khi chạy xong, xem kết quả tại `./output/<tag>/inference/<test_set>/`.

---

## 2. Chạy từng bước (để debug dễ)

### Bước 1 — cài môi trường (chỉ làm 1 lần)

```bash
bash scripts/setup_env.sh
conda activate da_detect
```

Script này:
- tạo conda env `da_detect` với Python 3.8
- cài PyTorch 1.10.1 + CUDA 11.3
- patch 2 chỗ legacy của maskrcnn-benchmark (`AT_CHECK` → `TORCH_CHECK`, `torch._six.PY37` → `True`)
- build extension C++/CUDA (`python setup.py build develop`)

### Bước 2 — tải dữ liệu

```bash
bash scripts/download_data.sh
```

Tải 4 file `.zip` từ Google Drive về `datasets/` và giải nén:

```
datasets/
├── source_real/source_real/{train,val}/{images,labels}
├── target_real/target_real/{train,val}/{images,labels}
├── source_test/source_test/val/{images,labels}   # 500 ảnh
└── target_test/target_test/val/{images,labels}   # 500 ảnh
```

### Bước 3 — chuẩn bị (YOLO → COCO + sinh miền rainy)

```bash
bash scripts/prepare_data.sh
```

- `tools/yolo2coco.py` chuyển label YOLO `.txt` thành COCO JSON, bơm polygon giả từ bbox (repo yêu cầu trường `segmentation`).
- `tools/make_rainy.py` sinh miền phụ trợ **rainy** bằng OpenCV (streak mưa + sương + giảm sáng) từ `source_train`. Miền này làm **negative** cho triplet loss.

Đầu ra:

```
datasets/annotations/{source_train,source_val,target_train,target_val,source_test,target_test,rainy_train}.json
datasets/rainy_real/train/{images,labels}
```

### Bước 4 — train

```bash
# pipeline chính cua paper (AdvGRL + Triplet, 3 mien)
bash scripts/train.sh triplet

# DA Faster R-CNN co ban (Img+Obj+GRL)
bash scripts/train.sh da_faster

# source-only (lower bound)
bash scripts/train.sh source_only

# chay ca 3 tuan tu
bash scripts/train.sh all
```

Checkpoint lưu tại `./output/<OUTPUT_SAVE_NAME>/model_final.pth`.

Có thể mở file config trong `configs/da_faster_rcnn/fusionda_*.yaml` để chỉnh `MAX_ITER`, `BASE_LR`, `IMS_PER_BATCH` nếu GPU yếu.

### Bước 5 — đánh giá

```bash
# eval pipeline chinh tren ca target_test va source_test
bash scripts/evaluate.sh triplet

# eval ca 3 model
bash scripts/evaluate.sh all
```

Script sẽ in **12 số COCO** cho mỗi mô hình / mỗi tập test:

- `mAP (0.5:0.95)`
- `mAP50`, `mAP75`
- `AP_small`, `AP_medium`, `AP_large`
- `AR@1`, `AR@10`, `AR@100`, `AR_small/medium/large`
- **AP / AP50 từng lớp** (`person`, `car`)

---

## 3. Mô tả 3 cấu hình thí nghiệm

| Config | File | Mô tả | Ghi chú |
|---|---|---|---|
| Source-only | `fusionda_source_only.yaml` | Train Faster R-CNN chỉ trên Cityscapes, không DA | Lower bound — cho thấy domain gap |
| DA Faster | `fusionda_da_faster.yaml` | Thêm image+instance GRL, 2 domain | Baseline DA kinh điển (Chen et al. CVPR18) |
| **DA Triplet** | `fusionda_triplet.yaml` | AdvGRL + Triplet loss 3 domain (source + target + rainy) | **Pipeline chính của paper WACV23** |

Mặc định đều dùng ResNet-50 C4 backbone, input 600×1200, batch 2. Bạn có thể chỉnh `IMS_PER_BATCH` lên 4/8 nếu GPU có ≥16 GB.

---

## 4. Bảng so sánh đề xuất (điền vào paper của bạn)

Test trên `target_test.zip` (500 ảnh Foggy Cityscapes):

| Method | Backbone | mAP | mAP50 | mAP75 | APs | APm | APl |
|---|---|---|---|---|---|---|---|
| Source-only | R-50-C4 | ... | ... | ... | ... | ... | ... |
| DA Faster R-CNN | R-50-C4 | ... | ... | ... | ... | ... | ... |
| DA Faster + AdvGRL + Triplet (WACV23) | R-50-C4 | ... | ... | ... | ... | ... | ... |
| **FusionDA (ours)** | ... | ... | ... | ... | ... | ... | ... |

Nên thêm AP riêng `person` và `car` (mỗi cột 2 dòng) để thể hiện rõ khác biệt theo lớp.

---

## 5. Các lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `RuntimeError: Error compiling objects for extension` | Chắc chắn đã chạy `perl -i -pe 's/AT_CHECK/TORCH_CHECK/g' ...` (script `setup_env.sh` đã làm tự động). |
| `AttributeError: module 'torch._six' has no attribute 'PY37'` | Script setup đã patch; nếu tự clone khác, chạy lại `setup_env.sh` hoặc sửa `maskrcnn_benchmark/utils/imports.py` thay `torch._six.PY37` bằng `True`. |
| `IndexError: list index out of range` khi train | Thay tất cả `torch.bool` bằng `torch.uint8` trong các file `loss.py` (hiếm gặp với PyTorch 1.10). |
| `pycocotools` báo `DeprecationWarning: np.float` | Vô hại. Nếu muốn tắt: `pip install numpy==1.23.5`. |
| CUDA OOM | Giảm `IMS_PER_BATCH` xuống 1 hoặc giảm `MIN_SIZE_TRAIN`/`MAX_SIZE_TRAIN`. |

---

## 6. Tuỳ biến nâng cao

- **Thay backbone R-50 → R-101**: đổi `MODEL.WEIGHT` sang `catalog://ImageNetPretrained/MSRA/R-101` và thêm `MODEL.BACKBONE.CONV_BODY: "R-101-C4"`.
- **Sinh rainy khác**: chỉnh `--severity_min/--severity_max` trong [tools/make_rainy.py](tools/make_rainy.py), hoặc dùng [efficientderain-master/generate_rainy_cityscape.py](efficientderain-master/generate_rainy_cityscape.py) (cần tải `Streaks_Garg06.zip` từ [EfficientDerain](https://github.com/tsingqguo/efficientderain)).
- **Multi-GPU**: `python -m torch.distributed.launch --nproc_per_node=<N> tools/train_net_triplet.py ...` và tăng `IMS_PER_BATCH` tỉ lệ thuận.

---

## 7. File trong gói

```
scripts/
├── setup_env.sh         # 1. cài môi trường
├── download_data.sh     # 2. tải data
├── prepare_data.sh      # 3. YOLO→COCO + sinh rainy
├── train.sh             # 4. train
├── evaluate.sh          # 5. eval
└── run_all.sh           # all-in-one
tools/
├── yolo2coco.py         # YOLO txt -> COCO JSON với polygon giả
├── make_rainy.py        # OpenCV rain synthesis (miền phụ trợ)
└── coco_eval_standalone.py  # in 12 số COCO + per-class AP
configs/da_faster_rcnn/
├── fusionda_source_only.yaml
├── fusionda_da_faster.yaml
└── fusionda_triplet.yaml   # pipeline chính
maskrcnn_benchmark/config/paths_catalog.py   # đã thêm 7 key fusionda_*_cocostyle
```
