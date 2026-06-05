# Training Recipes

This guide covers how to train and convert the models required for each use case.

- [Pipeline Defect Detection](#pipeline-defect-detection) — single-modality YOLOv8 object detection
- [Gas Detection (Multimodal)](#gas-detection-multimodal) — YOLOv8 image classifier + sensor MLP

---

## Pipeline Defect Detection

### 1. Train YOLOv8 Detection Model

```bash
conda activate pace
yolo detect train \
    data=datasets/pipeline_defects_detection/dataset.yaml \
    model=yolov8s.pt \
    imgsz=640 \
    epochs=50 \
    device=0,1,2,3,4,5
```

> Adjust `device=` to match your GPU setup (`device=0` for a single GPU,
> `device=cpu` for CPU-only).

### 2. Place the Trained Model

```bash
cp runs/detect/train/weights/best.pt models/pt_models/pipeline_defects_detection.pt
```

### 3. Convert to OpenVINO

```bash
conda activate pace
python setup/convert_to_openvino.py \
    --input  models/pt_models/pipeline_defects_detection.pt \
    --output models/ov_models/pipeline_defects_detection/image
```

Produces `models/ov_models/pipeline_defects_detection/image/best.xml` and `best.bin`.

---

## Gas Detection (Multimodal)

The gas detection use case requires **two separate models** that are fused at
inference time:

| Model | Input | Output | Framework |
|-------|-------|--------|-----------|
| YOLOv8 image classifier | 640×640 thermal image | 4-class probabilities | Ultralytics → OpenVINO |
| Sensor MLP | 7 MQ-sensor readings | 4-class probabilities | PyTorch → ONNX → OpenVINO |

Both models are independent and can be trained in any order.

### Step 1A — Train YOLOv8 Image Classifier

The image model is a YOLOv8 **classification** model (not detection). Use the
`yolo classify` command with the `train/` subdirectory as the data root:

```bash
conda activate pace
yolo classify train \
    data=datasets/gas_detection/images/train \
    model=yolov8s-cls.pt \
    imgsz=640 \
    epochs=50 \
    device=0
```

> `data=` must point to the parent of the per-class subdirectories, not to
> `dataset.yaml`. Ultralytics infers class names from subdirectory names automatically.

Place the best checkpoint:

```bash
cp runs/classify/train/weights/best.pt models/pt_models/gas_detection/yolov8s_cls_best.pt
```

Convert to OpenVINO:

```bash
conda activate pace
python setup/convert_to_openvino.py \
    --input  models/pt_models/gas_detection/yolov8s_cls_best.pt \
    --output models/ov_models/gas_detection/image
```

Produces `models/ov_models/gas_detection/image/best.xml` and `best.bin`.

---

### Step 1B — Train Sensor MLP

The sensor MLP is a small fully-connected network trained directly on the CSV
readings. Below is a sample training script.

```python
#!/usr/bin/env python3
"""Train sensor MLP for gas detection and export to ONNX."""
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH     = "datasets/gas_detection/sensor_data/Gas_Sensors_Measurements.csv"
ONNX_OUT     = "models/pt_models/gas_detection/sensor_mlp.onnx"
PT_OUT       = "models/pt_models/gas_detection/sensor_mlp_best.pt"
CLASSES      = ["Mixture", "NoGas", "Perfume", "Smoke"]
SENSOR_COLS  = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]
TRAIN_RATIO  = 0.8
EPOCHS       = 50
BATCH_SIZE   = 64
LR           = 1e-3
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

# Load CSV
X_raw, y = [], []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        X_raw.append([float(row[c]) for c in SENSOR_COLS])
        y.append(class_to_idx[row["Gas"]])

X_np = np.array(X_raw, dtype=np.float32)
y_np = np.array(y, dtype=np.int64)

# Z-score normalisation (computed from full dataset)
mean = X_np.mean(axis=0)
std  = X_np.std(axis=0)
std[std == 0] = 1.0
X_norm = (X_np - mean) / std

dataset = TensorDataset(torch.tensor(X_norm), torch.tensor(y_np))
n_train = int(len(dataset) * TRAIN_RATIO)
train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)


class SensorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4),
        )
    def forward(self, x):
        return self.net(x)


model     = SensorMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
    val_acc = correct / total
    print(f"Epoch {epoch:3d}/{EPOCHS}  val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), PT_OUT)

print(f"\nBest val accuracy: {best_val_acc:.4f}  →  {PT_OUT}")

# Export best checkpoint to ONNX
model.load_state_dict(torch.load(PT_OUT, weights_only=True))
model.eval()
Path(ONNX_OUT).parent.mkdir(parents=True, exist_ok=True)
dummy = torch.zeros(1, 7)
torch.onnx.export(
    model, dummy, ONNX_OUT,
    input_names=["sensor_input"],
    output_names=["logits"],
    dynamic_axes={"sensor_input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=13,
)
print(f"ONNX exported  →  {ONNX_OUT}")
```

Run it:

```bash
conda activate pace
python train_sensor_mlp.py
```

---

### Step 1C — Convert Sensor MLP to OpenVINO

The sensor MLP is exported via ONNX, so use the `ovc` command-line tool directly
(not `convert_to_openvino.py`, which is designed for YOLO `.pt` files):

```bash
conda activate pace
ovc models/pt_models/gas_detection/sensor_mlp.onnx \
    --output_model models/ov_models/gas_detection/sensor_mlp/sensor_mlp.xml
```

Produces `models/ov_models/gas_detection/sensor_mlp/sensor_mlp.xml` and `sensor_mlp.bin`.

---

### Step 2 — Verify Model Paths

Confirm the config points to the correct model files:

```yaml
# config/gas_detection/config.yaml
inference:
  model_path: models/ov_models/gas_detection/image/best.xml

sensor:
  model_path: models/ov_models/gas_detection/sensor_mlp/sensor_mlp.xml
  data_path:  datasets/gas_detection/sensor_data/Gas_Sensors_Measurements.csv
```

Both paths must exist before running inference.

---

## Next Steps

Once models are trained and converted, continue with the inference and agent pipeline:

```bash
conda activate pace
python run_complete_pipeline.py --device GPU
```

See [QUICKSTART.md](QUICKSTART.md) for the full end-to-end walkthrough.
