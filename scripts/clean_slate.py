#!/usr/bin/env python3
"""
Centralized cleanup utilities for PACE pipeline.
Provides functions to clean database and output directories.
"""

import shutil
from pathlib import Path
import sys

# Ensure project root is on sys.path so 'src' package can be imported
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def clean_database(db_path='out/sql_data/detections.db'):
    """
    Clear all detections from the SQLite database.
    
    Args:
        db_path: Path to the SQLite database file
    """
    try:
        import sqlite3
        
        db_file = Path(db_path)
        if db_file.exists():
            print(f"🗑️  Clearing database: {db_path}")
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            # Get existing table name(s) and clear them
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            for (table_name,) in tables:
                if not table_name.startswith('sqlite_'):
                    # Validate table name is alphanumeric (returned from sqlite_master, but validate for safety)
                    if table_name.replace('_', '').isalnum():
                        cursor.execute(f"DELETE FROM {table_name}")  # nosec B608
            conn.commit()
            conn.close()
            print(f"✓ Database cleared")
        else:
            print(f"ℹ️  Database not found: {db_path} (will be created)")
    except Exception as e:
        print(f"⚠️  Failed to clear database: {e}")


def clean_outputs(out_dir='out'):
    """
    Remove all files from the output directory.
    
    Args:
        out_dir: Path to the output directory
    """
    try:
        out_path = Path(out_dir)
        if out_path.exists():
            print(f"🧹 Clearing outputs directory: {out_dir}/")
            shutil.rmtree(out_path)
            out_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Output directory ready")
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Output directory created: {out_dir}/")
    except Exception as e:
        print(f"⚠️  Failed to clear outputs: {e}")


def clean_all(db_path=None, out_dir='out'):
    """
    Clean both database and output directory.
    
    Args:
        db_path: Path to the SQLite database file
        out_dir: Path to the output directory
    """
    print("\n" + "="*60)
    print("CLEANING WORKSPACE")
    print("="*60)
    clean_database(db_path)
    clean_outputs(out_dir)
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    import json
    import yaml
    
    parser = argparse.ArgumentParser(description='Clean PACE workspace')
    parser.add_argument('--db', default=None,
                       help='Database path (default: from config)')
    parser.add_argument('--out', default='out',
                       help='Output directory (default: out)')
    parser.add_argument('--db-only', action='store_true',
                       help='Clean database only')
    parser.add_argument('--out-only', action='store_true',
                       help='Clean outputs only')
    
    args = parser.parse_args()
    
    # Get database path from config if not provided
    db_path = args.db
    if db_path is None:
        with open("config.json", "r") as f:
            main_config = json.load(f)
        use_case_id = main_config.get("default-use-case", "pipeline_defects_detection")
        
        with open(f"config/{use_case_id}/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        
        db_path = cfg.get("sqlite", {}).get("db_path", f"out/sql_data/{use_case_id}.db")
    else:
        db_path = args.db
    
    if args.db_only:
        clean_database(db_path)
    elif args.out_only:
        clean_outputs(args.out)
    else:
        clean_all(db_path, args.out)
