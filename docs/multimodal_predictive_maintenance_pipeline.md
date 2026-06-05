# Multimodal Predictive Maintenance Pipeline Blueprint

> **Blueprint Series — Edge AI Predictive Maintenance for Critical Infrastructure**
> Multimodal Sensor Fusion + Multi-Agent Reasoning + Edge Vision with Intel OpenVINO

This document describes the **multimodal extension** of the Predictive Maintenance
Pipeline, where visual image data and chemical sensor readings are fused at inference
time to produce a single, more reliable classification result. The gas detection
use case is used as the reference implementation throughout this document.

The three-unit architecture, agent reasoning layer, prompt system, configuration
hierarchy, and deployment principles documented in
[predictive_maintenance_pipeline.md](predictive_maintenance_pipeline.md) remain
unchanged. This document focuses exclusively on what is new or different in the
multimodal variant: the sensor modality, the fusion mechanism, and the changes to
the inference and data layers that support them.


## Multimodality

A single sensor modality can be misleading. A camera sees smoke but cannot
distinguish perfume from combustible gas. A chemical sensor detects elevated MQ2
readings but cannot localise the source. Fusing independent signals reduces
false positives and improves classification confidence under ambiguous conditions —
exactly the failure modes that matter most in safety-critical industrial environments.

The multimodal pipeline combines:

- **Image classifier** — captures the visual spectral signature of the gas cloud
- **Sensor MLP** — distils seven MQ-series electrochemical sensor readings into a
  class probability vector
- **Late fusion** — produces a single weighted-average classification per sample

Because the two modalities are physically independent (a camera fault does not affect
sensors, and sensor saturation does not affect image quality), the fused prediction
degrades gracefully under partial failure.


## The Dataset and Gas Classes

The gas detection dataset contains two aligned data sources that are the foundation of multimodal inference.

### Image Data

Images are organised into four class directories and a flat validation split:

```
datasets/gas_detection/images/
├── Mixture/       # Multiple gases present simultaneously
├── NoGas/         # Baseline / clean air readings
├── Perfume/       # Aromatic compound — non-toxic reference class
├── Smoke/         # Combustion products — elevated hazard
└── val/           # Mixed validation set (all four classes)
```

Image filenames encode their identity, e.g. `586_Perfume.png`. The stem
(`586_Perfume`) is the key used to look up the corresponding sensor row in the CSV.

| Class   | Description                                              |
|---------|----------------------------------------------------------|
| Mixture | Co-presence of multiple gas types — complex spectral pattern |
| NoGas   | Clean air baseline — used for false-positive calibration  |
| Perfume | Aromatic compound reference — non-toxic, narrow spectrum  |
| Smoke   | Combustion-product signature — safety-critical class      |

### Sensor Data

Chemical sensor measurements are stored in a single CSV:

```
datasets/gas_detection/sensor_data/Gas_Sensors_Measurements.csv
```

Each row corresponds to one image sample and records the raw ADC readings from seven
MQ-series electrochemical sensors:

| Column                   | Description                                    |
|--------------------------|------------------------------------------------|
| `Serial Number`          | Row index                                      |
| `MQ2`                    | Combustible gas / LPG / propane / hydrogen     |
| `MQ3`                    | Alcohol / ethanol / benzene                    |
| `MQ5`                    | LPG / natural gas / coal gas                  |
| `MQ6`                    | LPG / butane / propane                         |
| `MQ7`                    | Carbon monoxide                                |
| `MQ8`                    | Hydrogen                                       |
| `MQ135`                  | Air quality / ammonia / sulfide / benzene      |
| `Gas`                    | Ground-truth class label                       |
| `Corresponding Image Name` | Image stem used to join with image files     |

The join key between sensor rows and image files is the `Corresponding Image Name`
column, which matches the stem of the image filename (without extension). This
alignment enables per-sample fusion of image and sensor predictions at inference time.


## The AI Models

### Image Classification Model — YOLOv8 Classifier on OpenVINO

A YOLOv8 classification model (not a detection model) is trained on the gas
detection image dataset. Unlike the detection variant described in
[predictive_maintenance_pipeline.md](predictive_maintenance_pipeline.md#the-ai-models),
this model assigns a single class label per image rather than producing bounding boxes.

- **Task:** Image classification (4-class softmax output)
- **Input size:** 640×640
- **Output:** Class probability vector of length 4
- **Model path:** `models/ov_models/gas_detection/image/best.xml`
- **Inference device:** GPU (direct OpenVINO, not via DL Streamer)
- **Pre-processing:** Resize → float32 normalization [0,1] → HWC→NCHW transpose

> **Note on inference backend:** Because DL Streamer's `gvainference` element has
> limitations with FP32-input classification models, the image classification path
> uses direct OpenVINO Runtime inference (`run_image_classification` in
> `run_inference_oep.py`) rather than the DL Streamer Docker pipeline. The detection
> pipeline described in the base blueprint continues to use DL Streamer.

### Sensor MLP Model — Small Pretrained Network on OpenVINO

A compact Multi-Layer Perceptron is pre-trained on the tabular sensor data. It is
the primary contribution of the multimodal extension — a lightweight, purpose-built
network that processes seven sensor readings and emits a class probability vector
of the same shape as the image classifier output, enabling direct weighted fusion.

- **Task:** 4-class classification from 7-dimensional sensor input
- **Input:** Z-score-normalised MQ-sensor vector `[MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135]`
- **Output:** Class probability vector of length 4 (softmax; logits converted at runtime)
- **Model path:** `models/ov_models/gas_detection/sensor_mlp/sensor_mlp.xml`
- **Inference device:** CPU (the model is small; CPU avoids GPU memory pressure)
- **Pre-processing:** Z-score normalisation using dataset-wide mean and standard
  deviation computed inline at inference time from the full CSV

The MLP is deliberately small. Its role is not to replace the image model but to
contribute an independent, complementary signal. On samples where the image is
ambiguous (e.g., diffuse smoke that looks similar to clean air), the sensor readings
often provide the discriminating evidence. Conversely, on samples where sensors are
near saturation or noisy, the image provides the stabilising signal.

### Reasoning Models — LLMs on OpenVINO

Identical to the base pipeline. See
[predictive_maintenance_pipeline.md — Reasoning Models](predictive_maintenance_pipeline.md#reasoning-models----llms-on-openvino-genai).


## Architecture: The Three-Unit Stack

The three-unit structure is unchanged from the base blueprint. The multimodal
extensions affect Units 1 and 2. Unit 3 (agent reasoning) operates identically —
it reads structured records from SQLite regardless of how many modalities produced
them.

```
┌──────────────────────────────────────────────────────────────┐
│                          Web UI                              │
│  Pipeline Execution · Interactive Chat · Agent Outputs       │
├──────────────────────────────────────────────────────────────┤
│                   Unit 3: Agent Reasoning                    │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Policy  │  │ Analysis │  │ Evidence │  │ Ticketing│      │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       └─────────────┴─────────────┴──────────────┘           │
│                           ▲                                  │
│                      Meta-Agent                              │
│                     (Coordinator)                            │
├──────────────────────────────────────────────────────────────┤
│               Unit 2: Data / Storage Layer                   │
│                                                              │
│                   SQLite — detections.db                     │
│  image_id · source · label · confidence                      │
│  image_confidence · sensor_confidence                        │
├──────────────────────────────────────────────────────────────┤
│              Unit 1: Inference / Ingestion                   │
│                                                              │
│  ┌────────────────────┐    ┌──────────────────────────────┐  │
│  │   Image Modality   │    │      Sensor Modality         │  │
│  │                    │    │                              │  │
│  │  YOLOv8 Classifier │    │   MQ2 · MQ3 · MQ5 · MQ6      │  │
│  │  (OpenVINO, GPU)   │    │   MQ7 · MQ8 · MQ135          │  │
│  │  → P(class|image)  │    │   Z-score norm → MLP (CPU)   │  │
│  │                    │    │   → P(class|sensors)         │  │
│  └─────────┬──────────┘    └──────────────┬───────────────┘  │
│            └──────────────┬───────────────┘                  │
│                           ▼                                  │
│              Late Fusion: weighted average                   │
│          0.6 × P(image) + 0.4 × P(sensor) → argmax           │
└──────────────────────────────────────────────────────────────┘
                            ▲
                    Intel CPU/iGPU/NPU
```

### Unit 1: Inference and Ingestion — Multimodal

Unit 1 now runs three sequential inference stages for each batch of images.

#### Stage A — Image Classification

The image classifier processes each image in the validation set independently:

1. Load image → resize to 640×640 → normalise to float32 [0, 1]
2. Transpose HWC → NCHW and add batch dimension
3. Run OpenVINO compiled model on GPU
4. Collect softmax output vector `P_image[i]` of length 4 per image `i`
5. If the model outputs raw logits (detected by negative values or non-unit sum),
   apply softmax: `exp(x - max(x)) / sum(exp(x - max(x)))`

The result is a dictionary `image_probs: {image_name → [p0, p1, p2, p3]}`.

#### Stage B — Sensor MLP Inference

The sensor MLP processes the aligned CSV rows for the same images:

1. Load `Gas_Sensors_Measurements.csv` into a lookup keyed by image stem
2. Compute Z-score normalisation parameters from the **full dataset** (not just the
   current batch): `μ = mean(all_rows)`, `σ = std(all_rows)`, clipping `σ = 1` where
   `σ = 0` to prevent division by zero
3. For each image name, retrieve its sensor row, normalise:
   `x_norm = (x - μ) / σ`, and run through the compiled OpenVINO MLP on CPU
4. Apply softmax to convert logits to probabilities if needed
5. If no sensor row exists for an image (missing data), fall back to a uniform
   distribution `[0.25, 0.25, 0.25, 0.25]`

The result is a dictionary `sensor_probs: {image_name → [p0, p1, p2, p3]}`.

#### Stage C — Late Fusion

Late fusion combines the two probability vectors using a configurable weighted average:

```
P_fused[i] = w_image × P_image[i] + w_sensor × P_sensor[i]
P_fused[i] = P_fused[i] / sum(P_fused[i])   # re-normalise
predicted_class = argmax(P_fused[i])
```

Default weights for the gas detection use case: `w_image = 0.6`, `w_sensor = 0.4`.
These are set in `config/gas_detection/config.yaml` under `fusion_weights` and can
be tuned without retraining either model.

The fused result record for each image carries:
- `label` — final predicted class name
- `confidence` — `max(P_fused)`, the fused probability of the predicted class
- `image_confidence` — `P_image[predicted_class_idx]`, image branch contribution
- `sensor_confidence` — `P_sensor[predicted_class_idx]`, sensor branch contribution

This three-field confidence breakdown is written through to the SQLite database,
enabling downstream agents to reason about the contribution of each modality to
any given classification.

**Handling missing modalities at runtime:**
- If image inference fails for an image, `P_image` defaults to a uniform distribution
  before fusion — the sensor signal still contributes.
- If no sensor row is found for an image, `P_sensor` defaults to uniform — the image
  signal still contributes.
- In this way, the fused pipeline degrades gracefully rather than failing hard when
  one modality is unavailable.

#### Visualization

After fusion, `generate_classification_viz` annotates each source image with three
prediction lines:

```
Image:   <image-branch predicted class>
Sensors: <sensor-branch predicted class>
Overall: <fused predicted class> (<fused confidence>)
```

Annotated images are saved to `out/gas_detection/viz/`. These overlays make it
immediately visible in which samples the two modalities agree and in which they
diverge — divergence is often a signal worth investigating.


### Unit 2: Data and Storage — Extended Schema

The SQLite schema is extended relative to the base pipeline to capture per-modality
confidence values alongside the fused result. The database is written to
`out/gas_detection/sql_data/detections.db`.

| Column             | Type      | Description                                                  |
|--------------------|-----------|--------------------------------------------------------------|
| `id`               | INTEGER   | Auto-increment primary key                                   |
| `image_id`         | INTEGER   | Sequential index of the image in the processed batch         |
| `source`           | TEXT      | Image filename (e.g. `586_Perfume.png`)                      |
| `label`            | TEXT      | Fused predicted class: Mixture / NoGas / Perfume / Smoke     |
| `confidence`       | REAL      | Fused classification confidence (0.0–1.0)                    |
| `image_confidence` | REAL      | Image branch confidence for the predicted class              |
| `sensor_confidence`| REAL      | Sensor branch confidence for the predicted class             |
| `created_at`       | TIMESTAMP | Row insertion timestamp                                      |

Indexes are created on `label`, `confidence`, and `source` for efficient agent queries.

The addition of `image_confidence` and `sensor_confidence` columns is the key schema
difference from the unimodal pipeline. Downstream agents can query, for example:

- "Show all Smoke detections where sensor confidence exceeded 0.7 but image confidence
  was below 0.4" — samples where the sensor was the deciding factor
- "Identify NoGas samples with high fused confidence but low sensor confidence" —
  potential sensor drift events
- "Count Mixture detections where both modalities agreed (|image_conf − sensor_conf| < 0.1)"

These modality-level queries are what make the multimodal schema genuinely useful
rather than just informational.


### Unit 3: Agent Reasoning

Unchanged from the base pipeline. The four-agent hub-and-spoke architecture
(Policy → Analysis + Evidence → Ticketing), the Meta-Agent coordinator, and the
LangGraph execution model all operate identically on the gas detection SQLite
database. See [predictive_maintenance_pipeline.md — Unit 3: Agent Reasoning](predictive_maintenance_pipeline.md#unit-3-agent-reasoning----langraph-multi-agent-orchestration).

The only use-case-specific configuration is in:
- `config/gas_detection/config.yaml` — thresholds, model paths, agent settings
- `prompts/gas_detection.txt` — gas-domain system prompt, policy, analysis, and
  evidence instructions
- `config/gas_detection/policy_fallback.json` — rule-based fallback policy for
  gas classes


## Data Flow

```
Image Store                     Sensor CSV
(datasets/gas_detection/        (Gas_Sensors_Measurements.csv)
 images/val/)
      │                                │
      ▼                                ▼
YOLOv8 Classifier               Sensor MLP (CPU)
(OpenVINO, GPU)                 Z-score normalise
P(class | image)                P(class | sensors)
      │                                │
      └──────────────┬─────────────────┘
                     ▼
           Late Fusion (weighted avg)
         0.6 × P_image + 0.4 × P_sensor
                     │
                     ▼
              SQLite Database
  image_id · source · label · confidence
  image_confidence · sensor_confidence
                     │
                     ▼
         Meta-Agent (LangGraph Coordinator)
                     │
         ├──→ Policy Agent  → Filtering rules (JSON)
         ├──→ Analysis Agent → Statistics + Summary report
         ├──→ Evidence Agent → Compliance audit trail
         └──→ Ticketing Agent → HTML tickets
                     │
                     ▼
           Output Artifacts
  out/gas_detection/agent/policy.json
  out/gas_detection/agent/analysis_report.json
  out/gas_detection/agent/analysis_summary.txt
  out/gas_detection/agent/evidence.json
  out/gas_detection/agent/evidence_trail.txt
  out/gas_detection/viz/           ← per-image annotated classification overlays
```


## Configuration

The `config.json` root config selects the active use case:

```json
{
  "use-case-id": "gas_detection"
}
```

The use-case config at `config/gas_detection/config.yaml` contains the multimodal-
specific parameters in addition to the standard agent and LLM settings:

```yaml
modality: multi           # 'image', 'sensor', or 'multi'

inference:
  task: classify          # classification, not detection

sensor:
  model_path: models/ov_models/gas_detection/sensor_mlp/sensor_mlp.xml
  data_path: datasets/gas_detection/sensor_data/Gas_Sensors_Measurements.csv

fusion_weights:
  image: 0.6
  sensor: 0.4
```

Setting `modality: image` disables the sensor MLP and runs image-only classification.
Setting `modality: sensor` disables the image model and runs sensor-only classification.
Setting `modality: multi` (default) enables the full late-fusion pipeline.

All agent, LLM, and SQL query settings follow the same structure as the base pipeline.
See [predictive_maintenance_pipeline.md — Configuration](predictive_maintenance_pipeline.md#configuration)
for the complete configuration reference.


## Running the Pipeline

### Complete End-to-End Pipeline (Gas Detection)

```bash
# Set active use case
echo '{"use-case-id": "gas_detection"}' > config.json

python run_complete_pipeline.py --num-images 100 --device GPU
```

Runs image + sensor inference, fuses results, persists to SQLite, and executes all
agents.

### Inference Only

```bash
python run_inference_oep.py --num-images 100 --device GPU
```

Populates the SQLite database from both modalities without running agent reasoning.
Useful for validating fusion results before committing to a full agent run.

### Image-Only or Sensor-Only Mode

Temporarily override the modality in `config/gas_detection/config.yaml`:

```yaml
modality: image    # or: sensor
```

Then run inference as normal. This is useful for ablation — comparing unimodal
and multimodal performance on the same dataset.

### Agent Orchestration Only

```bash
python -m scripts.run_agent_orchestration --use-case gas_detection
```

Re-runs the agent pipeline against an existing SQLite database without repeating
inference. See [predictive_maintenance_pipeline.md — Agent Orchestration Only](predictive_maintenance_pipeline.md#agent-orchestration-only).

For the full setup, installation, video mode, interactive chat, and web application
instructions, see [predictive_maintenance_pipeline.md — Running the Pipeline](predictive_maintenance_pipeline.md#running-the-pipeline).


## Output Artifacts

```
out/
└── gas_detection/
    ├── sql_data/
    │   └── detections.db                    # SQLite: fused classifications
    ├── viz/
    │   └── <image_stem>.jpg                 # Annotated: Image / Sensors / Overall labels
    └── agent/
        ├── policy.json
        ├── analysis_report.json
        ├── analysis_summary.txt
        ├── evidence.json
        ├── evidence_trail.txt
        └── tickets/
            └── TICKET-F{image_id}.html
```

The `viz/` directory is unique to the multimodal pipeline and is not present in the
base unimodal output. Each annotated image shows the independent prediction of each
modality alongside the fused result, providing a human-readable record of agreement
and divergence between modalities.


## Architectural Notes on Multimodality

### Late Fusion vs. Early / Hybrid Fusion

This implementation uses **late fusion**: each modality is processed by its own
dedicated model to produce a class probability vector, and the vectors are combined
after inference. This design choice has practical advantages for edge deployment:

- **Independent model training** — the image classifier and sensor MLP can be trained,
  updated, and replaced independently without retraining each other
- **Graceful degradation** — if one modality is unavailable, the other still produces
  a usable probability vector for fusion (falling back to uniform for the missing branch)
- **Interpretability** — per-modality confidence values are preserved in the database,
  making it possible to attribute predictions to their source signals after the fact
- **Computational flexibility** — the image model runs on GPU while the sensor MLP
  runs on CPU, naturally distributing the workload across available accelerators

Early fusion (concatenating raw features before a shared model) or intermediate/hybrid
fusion (merging intermediate feature representations) would require a jointly trained
architecture and would lose the independent degradation and attribution properties above.

### Sensor Normalisation Stability

Z-score normalisation parameters are computed from the full CSV at inference time
rather than stored as fixed values. This means the normalisation adapts automatically
if the CSV is updated with new samples. The trade-off is that adding new samples can
slightly shift the normalisation of existing samples. For production deployments where
the sensor distribution is known and stable, pre-computing and caching the
normalisation statistics is recommended.

### Fusion Weight Tuning

The default weights `image=0.6, sensor=0.4` reflect a mild preference for the image
signal, which tends to be more stable across environmental conditions. These weights
are hyperparameters and can be tuned via cross-validation on a held-out split.
A weight of `0.5 / 0.5` gives equal trust to both modalities. Setting one weight to
`1.0` is equivalent to running that modality in isolation.


## Conclusion

The multimodal extension demonstrates that the three-unit architecture of the
Predictive Maintenance Pipeline extends naturally to sensor fusion without changes
to the agent reasoning layer. The fusion logic is entirely contained within Unit 1,
the schema extension to Unit 2 is minimal (two additional confidence columns), and
Unit 3 inherits the full benefit of richer data without any modifications.

For all aspects of the system not covered here — agent descriptions, the prompt
system, LLM backends, setup and installation, Architectural Principles, and
deployment guidance — refer to the base blueprint:
[predictive_maintenance_pipeline.md](predictive_maintenance_pipeline.md).

---

*This blueprint is a proof of concept and is not intended for production use. Other
names and brands may be claimed as the property of others.*
