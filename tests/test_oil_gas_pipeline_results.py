"""Validation helpers for the Oil & Gas Pipeline use case."""

import math
import sqlite3
from pathlib import Path

import pandas as pd
import pytest


CSV_PATH = Path("datasets/oil_gas_pipeline/sensor_data/market_pipe_thickness_loss_dataset.csv")
DB_PATH = Path("out/oil_gas_pipeline/sql_data/detections.db")


def _load_oil_gas_comparison():
    if not CSV_PATH.exists():
        pytest.skip(f"Oil/gas dataset not found: {CSV_PATH}")
    if not DB_PATH.exists():
        pytest.skip(f"Oil/gas SQLite output not found: {DB_PATH}")

    df = pd.read_csv(CSV_PATH)

    conn = sqlite3.connect(DB_PATH)
    pred = pd.read_sql_query(
        """
        SELECT source, label, continuous_value
        FROM detections
        ORDER BY image_id
        """,
        conn,
    )
    conn.close()

    pred["row_idx"] = pred["source"].str.replace("row_", "", regex=False).astype(int)
    return pred.merge(
        df.reset_index().rename(columns={"index": "row_idx"}),
        on="row_idx",
    )


def compute_oil_gas_metrics():
    """Compare model outputs in SQLite against the source CSV labels and targets."""
    merged = _load_oil_gas_comparison()

    condition_correct = int((merged["label"] == merged["Condition"]).sum())
    condition_total = len(merged)
    condition_accuracy = condition_correct / condition_total

    errors = merged["continuous_value"] - merged["Thickness_Loss_mm"]
    mae = float(errors.abs().mean())
    rmse = float(math.sqrt((errors**2).mean()))

    return {
        "rows": condition_total,
        "condition_correct": condition_correct,
        "condition_accuracy": condition_accuracy,
        "thickness_loss_mae": mae,
        "thickness_loss_rmse": rmse,
    }


def test_oil_gas_sqlite_matches_dataset_size():
    merged = _load_oil_gas_comparison()
    assert len(merged) == 1000


def test_oil_gas_outputs_have_expected_columns():
    merged = _load_oil_gas_comparison()
    required_columns = {
        "source",
        "label",
        "continuous_value",
        "Condition",
        "Thickness_Loss_mm",
    }
    assert required_columns.issubset(set(merged.columns))
    assert merged["source"].notna().all()
    assert merged["label"].notna().all()
    assert merged["continuous_value"].notna().all()


def test_oil_gas_metrics_are_valid():
    metrics = compute_oil_gas_metrics()
    assert metrics["rows"] == 1000
    assert 0.0 <= metrics["condition_accuracy"] <= 1.0
    assert metrics["thickness_loss_mae"] >= 0.0
    assert metrics["thickness_loss_rmse"] >= 0.0


if __name__ == "__main__":
    metrics = compute_oil_gas_metrics()
    print("rows:", metrics["rows"])
    print("condition_correct:", metrics["condition_correct"])
    print("condition_accuracy:", metrics["condition_accuracy"])
    print("thickness_loss_mae:", metrics["thickness_loss_mae"])
    print("thickness_loss_rmse:", metrics["thickness_loss_rmse"])
