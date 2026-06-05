"""
Backend abstraction for database operations.
Provides unified interface for SQLite backend.
Uses schema from use-case config for table/column names.
"""

from typing import Dict, Any, List
from src.utility.sqlite_client import SQLiteClient


class DatabaseBackend:
    """Unified interface for database operations (SQLite)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database backend.
        
        Args:
            config: Configuration dictionary (use-case YAML)
        """
        self.config = config
        sqlite_cfg = config.get('sqlite', {})
        schema = config.get('schema')
        self.client = SQLiteClient(
            db_path=sqlite_cfg.get('db_path', 'out/sql_data/detections.db'),
            schema=schema
        )
        self.table_name = schema.get('table_name', 'detections') if schema else 'detections'
        # Determine column names from schema (for normalization)
        if schema:
            self._col_names = [col['name'] for col in schema.get('columns', [])]
        else:
            self._col_names = []
    
    def query_all(self, limit: int = None) -> List[Dict[str, Any]]:
        """Query all detections with optional limit."""
        query = f"SELECT * FROM {self.table_name}"
        if limit:
            query += f" LIMIT {limit}"
        results = self.client.execute_query(query)
        return self._normalize_sql_results(results)
    
    def query_by_confidence(self, min_conf: float, limit: int = None) -> List[Dict[str, Any]]:
        """Query detections above confidence threshold."""
        query = f"SELECT * FROM {self.table_name} WHERE confidence >= {min_conf}"
        if limit:
            query += f" LIMIT {limit}"
        results = self.client.execute_query(query)
        return self._normalize_sql_results(results)
    
    def insert_audit(self, audit_data: Dict[str, Any]):
        """Insert audit record (not implemented for SQL yet)."""
        pass
    
    def raw_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute raw SQL query."""
        if isinstance(query, str):
            return self.client.execute_query(query)
        else:
            raise ValueError("SQL backend requires string query")
    
    def get_client(self):
        """Get underlying client (for backward compatibility)."""
        return self.client
    
    def _normalize_sql_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return SQL results as-is (flat dicts). Schema-agnostic — no assumptions
        about which columns exist. Agents interpret fields based on config.
        """
        return results
