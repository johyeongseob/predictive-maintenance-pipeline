#!/usr/bin/env python3
"""
Read and display sample data from SQLite database.
Outputs data in both formatted table and raw JSON format.
"""

import sys
from pathlib import Path

# Add project root to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import yaml
from src.utility.sqlite_client import SQLiteClient


def main():
    parser = argparse.ArgumentParser(description="Read sample data from SQLite")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite database path (default: from config)")
    parser.add_argument("--limit", type=int, default=10, help="Number of records to retrieve (default: 10)")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum confidence threshold (default: 0.0)")
    parser.add_argument("--label", type=str, default=None, help="Filter by label/class (e.g., Deformation, Obstacle)")
    parser.add_argument("--frame", type=int, default=None, help="Filter by frame_id")
    parser.add_argument("--raw", action="store_true", help="Display raw JSON response")
    
    args = parser.parse_args()
    
    # Get database path from config if not provided
    cfg = None
    if args.db_path is None:
        with open("config.json", "r") as f:
            main_config = json.load(f)
        use_case_id = main_config.get("default-use-case", "pipeline_defects_detection")
        
        with open(f"config/{use_case_id}/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        
        args.db_path = cfg.get("sqlite", {}).get("db_path", f"out/sql_data/{use_case_id}.db")
    
    # Connect to SQLite (with schema from config if available)
    schema = cfg.get('schema') if cfg else None
    client = SQLiteClient(db_path=args.db_path, schema=schema)
    
    # Detect schema columns
    col_info = client.execute_query("PRAGMA table_info(detections)")
    col_names = [c['name'] for c in col_info]
    has_frame_id = 'frame_id' in col_names
    
    # Build query
    query = "SELECT * FROM detections WHERE confidence >= ?"
    params = [args.min_conf]
    
    if args.label:
        query += " AND label = ?"
        params.append(args.label)
    
    if args.frame is not None and has_frame_id:
        query += " AND frame_id = ?"
        params.append(args.frame)
    
    query += f" LIMIT {args.limit}"
    
    # Execute query
    results = client.execute_query(query, tuple(params))
    
    # Get statistics
    stats_query = "SELECT COUNT(*) as total, AVG(confidence) as avg_conf FROM detections WHERE confidence >= ?"
    stats_params = [args.min_conf]
    if args.label:
        stats_query += " AND label = ?"
        stats_params.append(args.label)
    if args.frame is not None and has_frame_id:
        stats_query += " AND frame_id = ?"
        stats_params.append(args.frame)
    
    stats = client.execute_query(stats_query, tuple(stats_params))[0]
    
    # Display formatted output
    print("=" * 120)
    print(f"SQLite Data Sample - {args.db_path}")
    if args.label:
        print(f"Filtered by label: {args.label}")
    if args.frame is not None and has_frame_id:
        print(f"Filtered by frame: {args.frame}")
    print(f"Minimum confidence: {args.min_conf}")
    print(f"Total matching records: {stats['total']}")
    print(f"Average confidence: {stats['avg_conf']:.3f}" if stats['avg_conf'] else "No records")
    print(f"Showing: {len(results)} records")
    print("=" * 120)
    print()
    
    for i, row in enumerate(results, 1):
        if has_frame_id:
            frame_id = row['frame_id']
            label = row['label']
            confidence = row['confidence']
            x = row['x']
            y = row['y']
            width = row['width']
            height = row['height']
            print(f"{i:3d}. Frame: {frame_id:3d} | Label: {label:14s} | Conf: {confidence:.3f} | "
                  f"BBox: ({x}, {y}, {width}x{height})")
        else:
            source = row.get('source', row.get('image_id', ''))
            label = row['label']
            confidence = row['confidence']
            print(f"{i:3d}. Source: {str(source):20s} | Label: {label:14s} | Conf: {confidence:.3f}")
    
    print()
    print("=" * 120)
    
    # Display class distribution
    dist_query = "SELECT label, COUNT(*) as count FROM detections WHERE confidence >= ?"
    dist_params = [args.min_conf]
    if args.frame is not None and has_frame_id:
        dist_query += " AND frame_id = ?"
        dist_params.append(args.frame)
    dist_query += " GROUP BY label ORDER BY count DESC"
    
    distribution = client.execute_query(dist_query, tuple(dist_params))
    
    if distribution:
        print("\nClass Distribution:")
        print("=" * 120)
        for row in distribution:
            print(f"  {row['label']:14s}: {row['count']:3d} detections")
        print("=" * 120)
    
    # Display raw JSON if requested
    if args.raw:
        print("\nRaw JSON Response:")
        print("=" * 120)
        print(json.dumps(results, indent=2, default=str))
        print("=" * 120)
    
    client.close()


if __name__ == "__main__":
    main()
