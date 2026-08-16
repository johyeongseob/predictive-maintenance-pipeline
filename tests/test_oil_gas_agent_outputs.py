import json
import sqlite3
from pathlib import Path

import pytest


CONFIG_PATH = Path("config/oil_gas_pipeline/config.yaml")
DB_PATH = Path("out/oil_gas_pipeline/sql_data/detections.db")
AGENT_DIR = Path("out/oil_gas_pipeline/agent")


def test_oil_gas_config_declares_corrosion_agent():
    if not CONFIG_PATH.exists():
        pytest.skip(f"Oil/gas config not found: {CONFIG_PATH}")

    import yaml

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    active_agents = config.get("agents", {}).get("active", [])
    schema_columns = {
        column["name"]
        for column in config.get("schema", {}).get("columns", [])
    }

    assert "corrosion" in active_agents
    assert "degradation_score" in schema_columns
    assert "continuous_value" in schema_columns
    assert "material_type" in schema_columns
    assert "max_pressure" in schema_columns
    assert "time_years" in schema_columns


def test_oil_gas_sqlite_contains_extended_columns():
    if not DB_PATH.exists():
        pytest.skip(f"Oil/gas SQLite output not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(detections)").fetchall()
        }
        row_count = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        sample = conn.execute(
            """
            SELECT source, label, degradation_score, continuous_value,
                   material_type, max_pressure, time_years
            FROM detections
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()

    assert row_count == 1000
    assert sample is not None
    assert {
        "degradation_score",
        "continuous_value",
        "material_type",
        "max_pressure",
        "time_years",
    }.issubset(columns)


def test_oil_gas_agent_artifacts_exist():
    required_files = [
        AGENT_DIR / "analysis_summary.txt",
        AGENT_DIR / "evidence_trail.txt",
        AGENT_DIR / "corrosion_summary.txt",
        AGENT_DIR / "corrosion_report.json",
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        pytest.skip(f"Oil/gas agent artifacts not found: {missing}")

    corrosion_summary = (AGENT_DIR / "corrosion_summary.txt").read_text(
        encoding="utf-8"
    )
    corrosion_report = json.loads(
        (AGENT_DIR / "corrosion_report.json").read_text(encoding="utf-8")
    )

    assert "corrosion" in corrosion_summary.lower() or "risk" in corrosion_summary.lower()
    assert "material" in corrosion_summary.lower()
    assert "maintenance" in corrosion_summary.lower()
    assert isinstance(corrosion_report, dict)
    assert corrosion_report
