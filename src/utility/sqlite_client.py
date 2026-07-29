"""
SQLite client for storing and querying detections data.
Parallel to vdms_client.py but uses SQLite backend.
Supports dynamic schema loaded from use-case config.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
import logging

logger = logging.getLogger(__name__)

# Default schema (used when no config schema is provided)
DEFAULT_SCHEMA = {
    'table_name': 'detections',
    'columns': [
        {'name': 'id', 'type': 'INTEGER PRIMARY KEY AUTOINCREMENT', 'required': False},
        {'name': 'frame_id', 'type': 'INTEGER', 'required': True},
        {'name': 'label', 'type': 'TEXT', 'required': True},
        {'name': 'confidence', 'type': 'REAL', 'required': True},
        {'name': 'x', 'type': 'INTEGER', 'required': True},
        {'name': 'y', 'type': 'INTEGER', 'required': True},
        {'name': 'width', 'type': 'INTEGER', 'required': True},
        {'name': 'height', 'type': 'INTEGER', 'required': True},
        {'name': 'metadata', 'type': 'TEXT', 'required': False},
        {'name': 'created_at', 'type': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP', 'required': False},
    ],
    'indexes': ['frame_id', 'label', 'confidence']
}


def _validate_identifier(identifier: str) -> str:
    """
    Validate and sanitize SQL identifiers (table/column names).
    Raises ValueError if identifier is invalid.
    
    Args:
        identifier: SQL identifier to validate
        
    Returns:
        The validated identifier
        
    Raises:
        ValueError: If identifier is invalid
    """
    if not identifier or not isinstance(identifier, str):
        raise ValueError("Identifier must be a non-empty string")
    
    # Allow alphanumeric, underscore, and no leading digits
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        raise ValueError(f"Invalid identifier: {identifier}. Must be alphanumeric with underscores.")
    
    return identifier


class SQLiteClient:
    """SQLite client for detections database with dynamic schema support."""
    
    def __init__(self, db_path: str = "out/sql_data/detections.db",
                 schema: Optional[Dict[str, Any]] = None):
        """
        Initialize SQLite client.
        
        Args:
            db_path: Path to SQLite database file
            schema: Schema definition dict from use-case config.
                    If None, uses DEFAULT_SCHEMA.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema or DEFAULT_SCHEMA
        self.table_name = _validate_identifier(self.schema.get('table_name', 'detections'))
        self._insertable_columns = [
            col['name'] for col in self.schema['columns']
            if 'AUTOINCREMENT' not in col['type'] and 'DEFAULT' not in col['type']
        ]
        self.conn = None
        self._connect()
        self._create_schema()
    
    def _connect(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
    
    def _create_schema(self):
        """Create table from schema definition if it doesn't exist."""
        cursor = self.conn.cursor()
        
        # Validate column names
        columns_sql = ', '.join(
            f"{_validate_identifier(col['name'])} {col['type']}"
            + (" NOT NULL" if col.get('required') and 'PRIMARY KEY' not in col['type'] else "")
            for col in self.schema['columns']
        )
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} ({columns_sql})
        """)
        
        # Create indexes
        for idx_col in self.schema.get('indexes', []):
            idx_col = _validate_identifier(idx_col)
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{idx_col} ON {self.table_name}({idx_col})"
            )
        
        self.conn.commit()
    
    def insert_detection(self, **kwargs):
        """
        Insert a single detection using schema-defined columns.
        
        Args:
            **kwargs: Column name/value pairs matching the schema's insertable columns.
        """
        cols = [c for c in self._insertable_columns if c in kwargs]
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)
        values = tuple(
            json.dumps(kwargs[c]) if isinstance(kwargs[c], dict) else kwargs[c]
            for c in cols
        )
        
        cursor = self.conn.cursor()
        # col_names contains validated column names from schema
        cursor.execute(
            f"INSERT INTO {self.table_name} ({col_names}) VALUES ({placeholders})",  # nosec B608
            values
        )
        self.conn.commit()
    
    def insert_detections_batch(self, detections: List[Dict[str, Any]]):
        """
        Insert multiple detections in a batch.
        
        Args:
            detections: List of detection dictionaries with keys matching schema columns.
        """
        if not detections:
            return
        
        # Use columns present in the first detection dict
        cols = [c for c in self._insertable_columns if c in detections[0]]
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)
        
        data = [
            tuple(
                json.dumps(d[c]) if isinstance(d.get(c), dict) else d.get(c)
                for c in cols
            )
            for d in detections
        ]
        
        cursor = self.conn.cursor()
        # col_names contains validated column names from schema
        cursor.executemany(
            f"INSERT INTO {self.table_name} ({col_names}) VALUES ({placeholders})",  # nosec B608
            data
        )
        self.conn.commit()
    
    def clear_detections(self):
        """Clear all detections from the database."""
        cursor = self.conn.cursor()
        # table_name is validated during initialization
        cursor.execute(f"DELETE FROM {self.table_name}")  # nosec B608
        self.conn.commit()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters tuple
            
        Returns:
            List of result dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_schema(self) -> str:
        """
        Get database schema as string for LLM context.
        Dynamically generated from the schema definition.
        
        Returns:
            Schema description string
        """
        lines = [
            "Database Schema:",
            "",
            f"Table: {self.table_name}",
            "Columns:",
        ]
        for col in self.schema['columns']:
            required = " (NOT NULL)" if col.get('required') else ""
            lines.append(f"  - {col['name']} ({col['type']}){required}")
        
        # Include class names if available
        names = self.schema.get('class_names')
        if names:
            labels_str = ', '.join(f"'{v}'" for v in names.values())
            lines.append(f"\nAvailable labels: {labels_str}")
        
        if self.schema.get('indexes'):
            lines.append("\nIndexes:")
            for idx_col in self.schema['indexes']:
                lines.append(f"  - idx_{idx_col} on {idx_col}")
        
        return '\n'.join(lines) + '\n'

    def get_schema_with_sensor_columns(self) -> str:
        """
        Get database schema with sensor column documentation for SQLCoder.

        Returns:
            Schema description string including sensor_raw_json usage notes.
        """
        schema_text = self.get_schema()
        column_names = {col['name'] for col in self.schema['columns']}
        if 'sensor_raw_json' not in column_names:
            return schema_text
        sensor_lines = [
            "Sensor column notes:",
            "  - sensor_raw_json: TEXT column containing a JSON object with raw gas sensor readings.",
            "  - sensor_raw_json keys: MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135.",
            "  - Use SQLite json_extract(sensor_raw_json, '$.MQ2') to query a sensor value.",
            "  - If a question mentions MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, or MQ135, use json_extract(sensor_raw_json, '$.<sensor_key>').",
            "  - Do not map MQ sensor readings to confidence.",
            "  - Example: SELECT image_id, label, json_extract(sensor_raw_json, '$.MQ2') AS MQ2 FROM detections;",
        ]

        return schema_text + '\n'.join(sensor_lines) + '\n'
     

    def count_detections(self) -> int:
        """Get total count of detections."""
        cursor = self.conn.cursor()
        # table_name is validated during initialization
        cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name}")  # nosec B608
        return cursor.fetchone()[0]
    
    def clear_all(self):
        """Clear all detections from database."""
        cursor = self.conn.cursor()
        # table_name is validated during initialization
        cursor.execute(f"DELETE FROM {self.table_name}")  # nosec B608
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
