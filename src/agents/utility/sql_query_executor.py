"""
SQL query executor using natural language to SQL model.
Parallel to vdms_query_executor.py but uses SQLite + text-to-SQL model.
"""

from typing import Tuple
from src.utility.sqlite_client import SQLiteClient
from .openvino_llm import OpenVINOLLM


class SQLQueryExecutor:
    """Executes natural language queries against SQLite using text-to-SQL model."""
    
    def __init__(self, sqlite_client: SQLiteClient, llm: OpenVINOLLM):
        """
        Initialize SQL query executor.
        
        Args:
            sqlite_client: SQLite client instance
            llm: OpenVINO LLM instance (sqlcoder model)
        """
        self.db = sqlite_client
        self.llm = llm
        if hasattr(sqlite_client, "get_schema_with_sensor_columns"):
            self.schema = sqlite_client.get_schema_with_sensor_columns()
        else:
            self.schema = sqlite_client.get_schema()
    
    def _generate_sql(self, natural_query: str) -> str:
        """
        Generate SQL query from natural language using LLM.
        
        Args:
            natural_query: Natural language query string
            
        Returns:
            Generated SQL query string
        """
        prompt = f"""### Task
Generate a SQL query to answer the following question.

### Database Schema
{self.schema}

### Question
{natural_query}

### SQL Query
SELECT"""
        
        # Generate SQL with LLM
        response = self.llm.invoke(
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.1  # Low temperature for deterministic SQL
        )
        
        # Extract SQL query - model generates continuation after "SELECT"
        sql_continuation = response.strip()
        
        # Handle cases where model includes "SELECT" again or adds space
        if sql_continuation.upper().startswith("SELECT"):
            sql = sql_continuation
        else:
            # Add space if needed
            if sql_continuation and not sql_continuation[0].isspace():
                sql = "SELECT " + sql_continuation
            else:
                sql = "SELECT" + sql_continuation
        
        # Clean up SQL - remove markdown, extra text
        if "```" in sql:
            # Extract from code block
            parts = sql.split("```")
            for part in parts:
                if "SELECT" in part.upper():
                    sql = part.strip()
                    if sql.startswith("sql\n"):
                        sql = sql[4:]
                    break
        
        # Remove common suffixes
        for suffix in ["###", "Question:", "Answer:", "\n\n"]:
            if suffix in sql:
                sql = sql.split(suffix)[0].strip()
        
        # Clean up SQL
        sql = sql.strip()
        if not sql.endswith(";"):
            sql += ";"
        
        return sql
    
    def _execute_sql(self, sql: str) -> list:
        """
        Execute SQL query and return results.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query results as list of dictionaries
        """
        try:
            results = self.db.execute_query(sql)
            return results
        except Exception as e:
            raise RuntimeError(f"SQL execution error: {e}\nQuery: {sql}")
    
    def _format_results_naturally(self, natural_query: str, sql: str, 
                                   results: list) -> str:
        """
        Format query results in natural language.
        
        Args:
            natural_query: Original natural language query
            sql: Generated SQL query
            results: Query results
            
        Returns:
            Natural language formatted output
        """
        # Format results as context
        if not results:
            results_str = "No results found."
        elif len(results) <= 10:
            results_str = str(results)
        else:
            results_str = f"{str(results[:10])}... ({len(results)} total results)"
        
        prompt = f"""Format the following query results in clear, natural language.

Question: {natural_query}
SQL Query: {sql}
Results: {results_str}

Provide a concise, readable summary. Be direct and concise - no repetitive explanations.
Use bullet points, numbered lists, or tables as appropriate."""
        
        formatted = self.llm.invoke(
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.1
        )
        
        return formatted.strip()
    
    def execute_natural_language_query(self, natural_query: str, 
                                       format_output: bool = True) -> Tuple[str, str, list]:
        """
        Execute a natural language query end-to-end.
        
        Args:
            natural_query: Natural language query string
            format_output: Whether to format output naturally
            
        Returns:
            Tuple of (formatted_output, sql_query, raw_results)
        """
        # Generate SQL from natural language
        sql = self._generate_sql(natural_query)
        
        # Execute SQL
        results = self._execute_sql(sql)
        
        # Format output
        if format_output:
            formatted = self._format_results_naturally(natural_query, sql, results)
        else:
            formatted = str(results)
        
        return formatted, sql, results
