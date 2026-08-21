#!/usr/bin/env python3
"""Create a scaffold for a new PACE use case."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

DEFAULT_HANDLERS = {
    ("detect", "image"): "dlstreamer_detect",
    ("detect", "multi"): "dlstreamer_detect",
    ("classify", "image"): "openvino_classify",
    ("classify", "sensor"): "sensor_flat",
    ("classify", "multi"): "sensor_flat",
}


def validate_name(name: str) -> str:
    """Return a safe use-case name or raise an argparse error."""
    if not VALID_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "Use-case name must start with a lowercase letter and contain only "
            "lowercase letters, numbers, and underscores."
        )
    return name


def default_handler(task: str, modality: str) -> str:
    """Return the default inference handler for the requested task/modality."""
    return DEFAULT_HANDLERS[(task, modality)]


def ensure_new_path(path: Path) -> None:
    """Raise when a scaffold target already exists."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {path}")


def write_text(path: Path, content: str) -> None:
    """Create a text file, refusing to overwrite existing files."""
    ensure_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    """Create a JSON file, refusing to overwrite existing files."""
    ensure_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_config_yaml(name: str, task: str, modality: str) -> str:
    """Return a placeholder config.yaml for a new use case."""
    handler = default_handler(task, modality)
    return f"""display_text: "{name.replace('_', ' ').title()}"
modality: {modality}  # 'image', 'sensor', or 'multi'

nc: 1
names:
  0: PlaceholderClass

inference:
  handler: {handler}
  task: {task}  # 'detect' or 'classify'
  model_path: models/ov_models/{name}/model.xml
  device: GPU
  confidence_threshold: 0.25
  imgsz: 640
  input_mode: images
  images_path: datasets/{name}/images/val

sqlite:
  db_path: out/{name}/sql_data/detections.db
  clear_on_run: true
  clear_outputs: true

agents:
  active: [policy, analysis, evidence]
  confidence_threshold: 0.5
  report_top_n: 5
  execution_mode: sequential
  shared_device: GPU

  policy:
    llm_mode: model
    model_id: models/ov_models/llms/Phi-4-mini-instruct-int4gq
    device: GPU

  analysis:
    llm_mode: model
    model_id: models/ov_models/llms/Phi-4-mini-instruct-int4gq
    device: GPU

  evidence:
    llm_mode: model
    model_id: models/ov_models/llms/Phi-4-mini-instruct-int4gq
    device: GPU

glue:
  mode: model
  model_id: models/ov_models/llms/Phi-4-mini-instruct-int4gq
  device: GPU
  max_steps: 4
  verbose: true
  enable_cache: false
  suppress_thinking: true

sql:
  model_id: models/ov_models/llms/sqlcoder-7b-2-int4cw
  device: GPU
  use_case: {name}

schema:
  table_name: detections
  columns:
    - name: id
      type: INTEGER PRIMARY KEY AUTOINCREMENT
      required: false
    - name: image_id
      type: INTEGER
      required: false
    - name: source
      type: TEXT
      required: true
    - name: label
      type: TEXT
      required: true
    - name: confidence
      type: REAL
      required: true
    - name: created_at
      type: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      required: false
  indexes:
    - label
    - confidence
    - source
"""


def build_prompt_text(name: str, task: str, modality: str) -> str:
    """Return placeholder prompt sections for a new use case."""
    title = name.replace("_", " ").title()
    return f"""# {title} - Prompt Configuration
# Replace placeholder text before running this use case.

[SYSTEM]
You are a predictive maintenance assistant for the {title} use case.
Task: {task}
Modality: {modality}

[POLICY]
Define filtering rules for this use case.
Return a policy JSON with min_conf_global and per_class_thresholds.

[ANALYSIS]
Generate a concise analysis report from the provided detection statistics.
Use actual values only and avoid placeholders in the final report.

[EVIDENCE]
Generate a concise audit trail showing policy decisions and traceability.
Output plain text only.

[SQL_SCHEMA]
Database Schema:
Table: detections
Columns:
  - id (INTEGER): Primary key
  - image_id (INTEGER): Image/sample identifier
  - source (TEXT): Source image or frame path
  - label (TEXT): Predicted class label
  - confidence (REAL): Prediction confidence
  - created_at (TIMESTAMP): Insert timestamp
"""


def create_usecase(name: str, task: str, modality: str, root: Path) -> list[Path]:
    """Create all scaffold files and directories for a new use case."""
    created: list[Path] = []

    config_dir = root / "config" / name
    dataset_train = root / "datasets" / name / "images" / "train"
    dataset_val = root / "datasets" / name / "images" / "val"
    out_dir = root / "out" / name
    prompts_path = root / "prompts" / f"{name}.txt"

    for directory in (config_dir, dataset_train, dataset_val, out_dir):
        ensure_new_path(directory)

    for directory in (config_dir, dataset_train, dataset_val, out_dir):
        directory.mkdir(parents=True, exist_ok=True)
        created.append(directory)

    write_text(config_dir / "config.yaml", build_config_yaml(name, task, modality))
    created.append(config_dir / "config.yaml")

    write_json(
        config_dir / "policy_fallback.json",
        {
            "min_conf_global": 0.5,
            "per_class_thresholds": {
                "PlaceholderClass": 0.5,
            },
        },
    )
    created.append(config_dir / "policy_fallback.json")

    write_json(
        config_dir / "predef_questions.json",
        {
            "analysis": [
                f"Summarize the {name} analysis.",
                "What are the most common classes detected?",
            ],
            "evidence": [
                "What policy was used for this run?",
                "How many detections were kept after filtering?",
            ],
            "sql": [
                "Count detections grouped by label.",
                "Show detections with confidence above 0.5.",
            ]
        },
    )
    created.append(config_dir / "predef_questions.json")

    write_text(prompts_path, build_prompt_text(name, task, modality))
    created.append(prompts_path)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new PACE use-case scaffold.")
    parser.add_argument("--name", required=True, type=validate_name, help="Use-case name, e.g. bridges")
    parser.add_argument("--task", required=True, choices=["detect", "classify"], help="Inference task")
    parser.add_argument(
        "--modality",
        required=True,
        choices=["image", "sensor", "multi"],
        help="Input modality",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    created = create_usecase(args.name, args.task, args.modality, root)

    print(f"Created use-case scaffold: {args.name}")
    for path in created:
        print(f"  - {path.relative_to(root)}")


if __name__ == "__main__":
    main()
