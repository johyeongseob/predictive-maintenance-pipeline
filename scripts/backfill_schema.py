#!/usr/bin/env python3
"""
Backfill SQLite detection tables with columns from the configured schema.

This is intended for existing databases that were created before new nullable
columns, such as sensor_raw_json, were added to config/gas_detection/config.yaml.
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import yaml


DEFAULT_CONFIG_PATH = "config/gas_detection/config.yaml"
DEFAULT_DB_PATH = "out/gas_detection/sql_data/detections.db"


def validate_identifier(identifier: str) -> str:
    """Validate a SQLite table or column identifier."""
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    return identifier


def load_schema(config_path: Path) -> Dict[str, Any]:
    """Load schema configuration from a use-case YAML config."""
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    schema = config.get("schema")
    if not schema:
        raise ValueError(f"No schema section found in {config_path}")

    return schema


def get_existing_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Return existing column names for a SQLite table."""
    cursor = conn.execute(f"PRAGMA table_info({validate_identifier(table_name)})")
    return [row[1] for row in cursor.fetchall()]


def build_add_column_sql(table_name: str, column: Dict[str, Any]) -> str:
    """Build an ALTER TABLE ADD COLUMN statement for a schema column."""
    column_name = validate_identifier(column["name"])
    column_type = column["type"]
    table_name = validate_identifier(table_name)

    if "PRIMARY KEY" in column_type.upper() or "AUTOINCREMENT" in column_type.upper():
        raise ValueError(f"Cannot backfill primary key column: {column_name}")

    return f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"


def backfill_database(db_path: Path, schema: Dict[str, Any], dry_run: bool) -> int:
    """Backfill one SQLite database and return the number of added columns."""
    table_name = validate_identifier(schema.get("table_name", "detections"))

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        existing_columns = set(get_existing_columns(conn, table_name))
        if not existing_columns:
            raise ValueError(f"Table not found or has no columns: {table_name}")

        statements = []
        for column in schema.get("columns", []):
            column_name = column["name"]
            if column_name in existing_columns:
                continue
            statements.append(build_add_column_sql(table_name, column))

        if not statements:
            print(f"{db_path}: schema already up to date.")
            return 0

        for statement in statements:
            if dry_run:
                print(f"{db_path}: would run: {statement}")
            else:
                conn.execute(statement)
                print(f"{db_path}: ran: {statement}")

        if not dry_run:
            conn.commit()

        return len(statements)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SQLite detection DB columns from config schema."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to use-case config YAML. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--db",
        nargs="+",
        default=[DEFAULT_DB_PATH],
        help=f"One or more SQLite DB paths. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ALTER TABLE statements without modifying the database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(Path(args.config))

    total_added = 0
    for db in args.db:
        total_added += backfill_database(Path(db), schema, args.dry_run)

    mode = "would add" if args.dry_run else "added"
    print(f"Done: {mode} {total_added} column(s).")


if __name__ == "__main__":
    main()
