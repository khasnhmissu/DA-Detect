# Chạy lại từ đầu — sau khi đã có toàn bộ fix

Quy trình ngắn nhất + tiết kiệm disk nhất. Đã verify với:
- PyTorch 2.2.2 + cu121
- System CUDA 12.x (driver ≥ 530)
- GPU compute capability 7.0–9.0 (V100, T4, A100, RTX 30/40/50, H100)

## TL;DR — 1 lần copy paste

```bash
cd /workspace/DA-Detect
git pull

# 1. Reset moi ket qua train cu (giu data, env, build)
bash scripts/reset_run.sh --yes

# 2. Regen annotations + rainy_real
conda activate da_detect
bash scripts/prepare_data.sh

# 3. Train + eval ca 4 thi nghiem (chay nen ~10h)
nohup bash scripts/run_compare.sh > run.log 2>&1 &
echo $! > run.pid

# 4. Theo doi
tail -f run.log | grep -E "TRAIN|EVAL|iter:|loss:|mAP|Error|Traceback|ALL DONE"
```

Sau khi xong:

```bash
# 5. Don dep disk (giu lai model_final.pth + metrics.txt)
bash scripts/cleanup_disk.sh

# Hoac aggressive: xoa luon rainy_real + pretrained cache
bash scripts/cleanup_disk.sh --aggressive
```

---

## Disk usage ước tính

| Pha | Item | Size |
|---|---|---|
| Trước train | source_real + target_real + source_test + target_test | ~3.5 GB |
| Sau prepare_data | + rainy_real (sinh từ source) | +5.6 GB → ~9 GB |
| Sau prepare_data | + annotations/*.json (7 file) | +60 MB |
| Đang train | + 1 intermediate ckpt (auto-cleanup) | +500 MB peak |
| Sau train (per model) | model_final.pth | 250-500 MB |
| Sau train cả 4 model | 4 model_final.pth | ~2 GB |
| Sau eval | + predictions.pth, bbox.json, metrics.txt | +5 MB |
| **Tổng peak** | | **~12 GB** |
| **Sau aggressive cleanup** | | **~5 GB** (chỉ source/target + 4 model_final + metrics) |

---

## Auto-cleanup intermediate checkpoints

Tôi đã sửa `Checkpointer.save()` ở [maskrcnn_benchmark/utils/checkpoint.py](maskrcnn_benchmark/utils/checkpoint.py):

> Mỗi lần save một checkpoint mới (vd. `model_0014000.pth`), tự động xoá tất cả `model_XXXXXXX.pth` cũ, **trừ `model_final.pth`**.

Kết quả:
- Trong khi train: chỉ có **1 intermediate** trong folder bất kỳ lúc nào (~500MB)
- Sau train xong: chỉ còn `model_final.pth`

→ Không cần lo tràn disk khi train song song nhiều model.

---

## Các script

| Script | Chức năng |
|---|---|
| [scripts/setup_env.sh](scripts/setup_env.sh) | Cài conda env, PyTorch, build extension. **Chạy 1 lần** |
| [scripts/download_data.sh](scripts/download_data.sh) | Tải 4 zip + extract + xoá zip. Idempotent |
| [scripts/prepare_data.sh](scripts/prepare_data.sh) | YOLO→COCO 6 splits + sinh rainy_real |
| [scripts/run_compare.sh](scripts/run_compare.sh) | Train + eval **4 thí nghiệm** (skip cái đã train) |
| [scripts/evaluate.sh](scripts/evaluate.sh) | Eval 1 model trên 2 test set |
| [scripts/cleanup_disk.sh](scripts/cleanup_disk.sh) | **MỚI** — dọn intermediate ckpt + cache |
| [scripts/reset_run.sh](scripts/reset_run.sh) | **MỚI** — xoá output/, annotations/, rainy_real/ |

---

## Tools (Python)

| Tool | Chức năng |
|---|---|
| [tools/yolo2coco.py](tools/yolo2coco.py) | YOLO txt → COCO JSON (polygon dummy) |
| [tools/make_rainy.py](tools/make_rainy.py) | Sinh miền phụ trợ rainy bằng OpenCV |
| [tools/eval_standalone.py](tools/eval_standalone.py) | Eval đầy đủ → predictions.pth + bbox.json + metrics.txt |
| [tools/coco_eval_standalone.py](tools/coco_eval_standalone.py) | Standalone COCO eval từ bbox.json |
| [tools/eval_progress.py](tools/eval_progress.py) | Eval nhiều ckpt — chuẩn đoán hội tụ |

---

## Configs (4 thí nghiệm)

```
configs/da_faster_rcnn/
├── fusionda_source_only_c4.yaml      # R-50-C4, no DA, 28k iter (~40 epoch)
├── fusionda_source_only_fpn.yaml     # R-50-FPN, no DA, 28k iter (~40 epoch)
├── fusionda_triplet_c4.yaml          # R-50-C4, AdvGRL+Triplet, 50k iter (~71 epoch)
└── fusionda_triplet_fpn.yaml         # R-50-FPN, AdvGRL+Triplet, 50k iter (~71 epoch)
```

Tất cả: input 320×640, IMS_PER_BATCH=4, BASE_LR=0.001, NUM_CLASSES=3 (person+car+bg).

---

## Các fix đã commit (so với repo gốc)

Tất cả đã được commit thẳng vào file, **không cần patch lúc setup**:

| File | Vấn đề | Fix |
|---|---|---|
| `utils/imports.py` | `torch._six.PY37` (xoá ở torch ≥ 1.13) | Bỏ check, dùng thẳng `importlib.util` |
| `utils/c2_model_loading.py` | `torch._six.PY37` | Bỏ check |
| `utils/model_zoo.py` | `_download_url_to_file` private bị xoá | Dùng `download_url_to_file` public + `urllib.parse` |
| `utils/checkpoint.py` | Lưu hết intermediate ckpt → tràn disk | **Auto-cleanup** intermediate cũ khi save mới |
| `data/datasets/evaluation/coco/coco_eval.py` | `len(catIds) is 1` (Py 3.8 warn) | Đổi `== 1` |
| `modeling/rpn/anchor_generator.py` | `np.float` (numpy ≥ 1.24 xoá) | Đổi `np.float64` |
| `modeling/detector/generalized_rcnn.py` | (1) Source-only thiếu nhánh; (2) Slice sai khi IMS_PER_BATCH > 1; (3) Không hỗ trợ FPN | Thêm else branch + B-aware slicing + slice tất cả feature levels |
| `modeling/da_heads/da_heads.py` | `resnet_backbone` flag bật cho cả FPN → avgpool sai | Detect `'FPN' in CONV_BODY` đúng + `num_ins_inputs = MLP_HEAD_DIM` cho FPN |
| `modeling/da_heads/loss.py` | `cat dim=0` của multi-level FPN crash | BCE riêng từng level, average |
| `modeling/rpn/inference.py` | `add_gt_proposals` bỏ qua target → empty proposals → matcher crash | Luôn append GT (target's GT bị filter sau bởi `is_source`) |
| `csrc/cuda/*.cu` | `THC/THC.h`, `THCudaCheck`, `THCCeilDiv` (xoá ở torch ≥ 1.11) | Patch `nms.cu` trực tiếp + `patch_csrc.sh` cho các file còn lại |

Eval scripts có monkey-patch `np.float = float` cho `pycocotools < 2.0.8`.

---

## Khi gặp lỗi mới

Pipeline đã được sửa tới 8+ lỗi tương thích PyTorch 2.x. Có thể vẫn còn vài edge case. Nếu gặp lỗi:

1. **Capture log đầy đủ** quanh `Traceback`
2. Run `nvidia-smi` xem GPU OK không
3. Run `df -h` xem có hết disk không
4. Run `python -c "from maskrcnn_benchmark import _C; print('OK')"` confirm extension load được

Paste log + thông tin trên cho người support, fix sẽ nhanh.
