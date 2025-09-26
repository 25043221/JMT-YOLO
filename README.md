# JMT-YOLO

Multi-task Domain Adaptation via Segmentation-Guided Consistency Learning

## Installation

Environment (recommended):
* Python 3.8
* CUDA 11.1 / 11.2
* PyTorch 1.10.0 + cu111
* (Optional) Weights & Biases for logging

Clone:
```
git clone https://github.com/25043221/JMT-YOLO.git
cd JMT-YOLO
```

Install dependencies :
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
---

## Dataset Preparing

Prepare your own datasets in YOLO format and create a corresponding .yaml file in the folder (`data/yamls_sda/`)

Example YAMLs provided:
* `pascalvoc0712_clipart1k_VOC.yaml`
* `cityscapes_csfoggy_VOC.yaml`
* `KITTI_CityScapes_VOC.yaml`

### Example (Cityscapes → FoggyCityscapes):
Follow the preparation pipeline in [SSDA-YOLO](https://github.com/hnuzhy/SSDA-YOLO) to build the detection dataset (Cityscapes → FoggyCityscapes).

Segmentation dataset: Download `gtFine_trainvaltest.zip` from [website](https://www.cityscapes-dataset.com) and extract the fine annotations (train/val). Requirements for the segmentation branch in this project:
* Convert each gtFine annotation to a single‑channel 8‑bit index mask (each pixel = class id) aligned 1:1 with the detection image; place under `segData/images/<split>/`.
* Detection labels (YOLO txt) and segmentation masks must share the same filename stem (e.g. `frankfurt_000000_000294_leftImg8bit.png` ↔ `frankfurt_000000_000294_leftImg8bit.txt` ↔ `frankfurt_000000_000294_leftImg8bit.png` mask).
* Foggy images come from `leftImg8bit_trainval_foggyDBF.zip` (select fog density `beta=0.02` among `(0.01, 0.02, 0.005)`).

Keep directory hierarchy consistent with the YAML fields (`train_source_real`, `train_target_real`, and corresponding `train_*_seg_*`).

Quick setup steps:
1. Download & extract: `leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`, `leftImg8bit_trainval_foggyDBF.zip`.
2. Generate YOLO detection labels from original annotations (polygons → bounding boxes → YOLO txt).
3. Place Foggy and Normal style‑translated images into their respective real/fake domain folders.
4. Convert fine annotations to index mask PNGs and place them into the `segData` directory structure.
5. Update or verify paths and `nc / n_segcls` in `data/yamls_sda/cityscapes_csfoggy_VOC.yaml`.

---


## Training and Testing

### Training

Example (Cityscapes → FoggyCityscapes):
```
python -m torch.distributed.launch --nproc_per_node 1 \
  train.py \
  --weights weights/yolov5l.pt \
  --data data/yamls_sda/cityscapes_csfoggy_VOC.yaml \
  --name C2F \
  --img 960 --device 0 --batch-size 2 --epochs 200 \
  --lambda_weight 0.005 --consistency_loss --alpha_weight 2.0
```


### Testing

```
python test.py \
    --data data/yamls_sda/cityscapes_csfoggy_VOC.yaml \
    --weights run/train/best_student_C2F.pt \
    --name C2F \
    --img 960 --batch-size 4 --device 0
```
---


## References
* YOLOv5: https://github.com/ultralytics/yolov5
* SSDA-YOLO: https://github.com/hnuzhy/SSDA-YOLO
* MIC: https://github.com/lhoyer/MIC
* CUT: https://github.com/taesungp/contrastive-unpaired-translation

---

