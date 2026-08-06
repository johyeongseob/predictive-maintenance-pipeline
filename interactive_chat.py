#!/usr/bin/env python3
"""
Interactive chat interface for PACE pipeline analysis.
Provides a unified question prompt with automatic intent routing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import yaml
import re
from src.agents.utility.openvino_llm import OpenVINOLLM, RemoteLLM
from src.utility import load_prompts
from src.utility.chat_intent import classify_intent_keyword, resolve_active_chat_mode, classify_intent_with_llm
from src.utility.sqlite_client import SQLiteClient


class InteractiveChat:
    # Common instruction for all agent queries
    QA_INSTRUCTION = "Answer using only information from the context above. Do not add information not present in the context. Be concise."
    
    def __init__(self, config_path=None):
        self.llm = None
        self.sql_llm = None  # Dedicated SQL model (sqlcoder)
        self.config = None
        self.config_path = config_path
        self.use_case_id = None
        self.db_client = None
        self.analysis_summary = None
        self.evidence_data = None
        self.system_prompt = ""
        
    def load_config(self):
        """Load configuration and initialize LLM."""
        if self.config_path:
            config_path = Path(self.config_path)
            use_case_id = config_path.parent.name
        else:
            # Read default use case from config.json
            with open("config.json", "r") as f:
                main_config = json.load(f)
            use_case_id = main_config.get("default-use-case", "pipeline_defects_detection")
            config_path = Path("config") / use_case_id / "config.yaml"
        self.use_case_id = use_case_id
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        glue_cfg = self.config.get("glue", {})
        
        # Initialize LLM
        mode = glue_cfg.get("mode", "fallback")
        enable_cache = glue_cfg.get("enable_cache", False)
        
        if mode == "server":
            server_url = glue_cfg.get("server_url", "http://localhost:8000")
            print(f"🌐 Connecting to LLM server: {server_url}")
            try:
                self.llm = RemoteLLM(server_url=server_url, verbose=False, enable_cache=enable_cache)
                print("✅ Connected to remote LLM server")
            except Exception as e:
                print(f"❌ Failed to connect to server: {e}")
                return False
                
        elif mode == "model":
            model_id = glue_cfg.get("model_id", "models/ov_models/llms/phi-3.5-mini")
            device = glue_cfg.get("device", "CPU")
            suppress_thinking = glue_cfg.get("suppress_thinking", True)
            print(f"🚀 Loading OpenVINO LLM: {model_id} on {device}")
            try:
                self.llm = OpenVINOLLM(model_path=model_id, device=device, verbose=False, enable_cache=enable_cache, suppress_thinking=suppress_thinking)
                print("✅ LLM loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load LLM: {e}")
                return False
        else:
            print("❌ LLM mode not configured (set glue.mode to 'model' or 'server')")
            return False
        
        # Load system prompt
        prompt_file = f"prompts/{use_case_id}.txt"
        prompts = load_prompts(prompt_file)
        self.system_prompt = prompts.get("system", "")
        self.sql_schema = prompts.get("sql_schema", "")
        
        # Initialize SQL model (sqlcoder-7b-2)
        sql_cfg = self.config.get('sql', {})
        sql_model_id = sql_cfg.get('model_id')
        sql_device = sql_cfg.get('device', 'GPU')
        sql_suppress_thinking = sql_cfg.get('suppress_thinking', True)
        
        if not sql_model_id:
            print("⚠️  SQL model not configured in config, will use main LLM for SQL")
            self.sql_llm = None
        else:
            print(f"\n🗄️  Loading SQL model: {sql_model_id} on {sql_device}")
            try:
                self.sql_llm = OpenVINOLLM(model_path=sql_model_id, device=sql_device, verbose=False, enable_cache=False, suppress_thinking=sql_suppress_thinking)
                print("✅ SQL model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load SQL model ({e}), will use main LLM for SQL")
                self.sql_llm = None
        
        # Initialize database client
        sqlite_cfg = self.config.get('sqlite', {})
        schema = self.config.get('schema')
        db_path = sqlite_cfg.get('db_path', 'out/sql_data/detections.db')
        self.db_client = SQLiteClient(db_path, schema=schema)
        print(f"✅ Connected to database: {db_path}")
        
        return True
    
    def load_artifacts(self):
        """Load recent analysis artifacts."""
        try:
            # Determine use-case output dir
            use_case_id = self.use_case_id or "pipeline_defects_detection"
            out_dir = Path("out") / use_case_id / "agent"
            
            # Load analysis summary
            summary_path = out_dir / "analysis_summary.txt"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    self.analysis_summary = f.read()
            
            # Load evidence data
            evidence_path = out_dir / "evidence.json"
            if evidence_path.exists():
                with open(evidence_path, "r") as f:
                    self.evidence_data = json.load(f)
                    
            return True
        except Exception as e:
            print(f"⚠️ Could not load artifacts: {e}")
            return False
    
    def ask_question(self, prompt, show_thinking=True):
        """Ask LLM a question and return response."""
        if show_thinking:
            print("\n🤔 Thinking...", end="", flush=True)
        
        # Don't add extra instruction - the prompt already contains it
        try:
            response = self.llm.invoke(prompt, temperature=0.1, max_new_tokens=500)
            if show_thinking:
                print("\r" + " " * 20 + "\r", end="")  # Clear "Thinking..."
            
            return response.strip()
        except Exception as e:
            return f"❌ Error generating response: {e}"
    
    def generate_sql_query(self, natural_language_query: str) -> str:
        """Convert natural language query to SQL using sqlcoder-7b-2 model."""
        if self.sql_schema:
            schema_info = self.sql_schema
        elif self.db_client and hasattr(self.db_client, "get_schema_with_sensor_columns"):
            schema_info = self.db_client.get_schema_with_sensor_columns()
        elif self.db_client:
            schema_info = self.db_client.get_schema()
        else:
            schema_info = "Table: detections with columns: id, frame_id, label, confidence, x, y, width, height, created_at"
        
        prompt = f"""{schema_info}

Question: {natural_language_query}

Generate a SQLite query to answer this question. Return ONLY the SQL query without explanation.
Use SQLite syntax only. Do not use ILIKE; use LIKE or exact equality instead.

SELECT"""
        
        print("\n🔍 Generating SQL query...", end="", flush=True)
        
        # Use sqlcoder model if available, otherwise fallback to main LLM
        model_to_use = self.sql_llm if self.sql_llm else self.llm
        sql_query = model_to_use.invoke(prompt, temperature=0.0, max_new_tokens=200)
        
        print("\r" + " " * 30 + "\r", end="")  # Clear message
        
        # Clean up the SQL query
        sql_query = "SELECT " + sql_query.strip()  # Add back SELECT with space
        
        # Remove markdown code blocks if present
        if "```" in sql_query:
            lines = sql_query.split("\n")
            sql_query = "\n".join([l for l in lines if not l.strip().startswith("```")])
        
        # Extract just the SQL (stop at semicolon or explanation)
        if ";" in sql_query:
            sql_query = sql_query.split(";")[0] + ";"
        
        sql_query = sql_query.strip()
        sql_query = re.sub(r"\bILIKE\b", "LIKE", sql_query, flags=re.IGNORECASE)
        return sql_query
    
    def execute_and_format_query(self, sql_query: str) -> str:
        """Execute SQL query and format results."""
        try:
            results = self.db_client.execute_query(sql_query)
            
            if not results:
                return "No results found."
            
            # Format results as a table
            output = []
            # Build a summary line: N detections across M unique frames
            columns = list(results[0].keys())
            if 'frame_id' in columns:
                unique_frames = len(set(row['frame_id'] for row in results))
                output.append(f"\n📊 Query Results ({len(results)} detections across {unique_frames} unique frame{'s' if unique_frames != 1 else ''}):\n")
            else:
                output.append(f"\n📊 Query Results ({len(results)} rows):\n")
            output.append("-" * 70)
            
            # Get column names
            if results:
                columns = list(results[0].keys())
                # Header
                header = " | ".join(f"{col:15s}" for col in columns)
                output.append(header)
                output.append("-" * 70)
                
                # Data rows (limit to first 20)
                for i, row in enumerate(results[:20]):
                    row_str = " | ".join(f"{str(row[col]):15s}" for col in columns)
                    output.append(row_str)
                
                if len(results) > 20:
                    output.append(f"\n... and {len(results) - 20} more rows")
            
            output.append("-" * 70)
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Error executing query: {e}"

    def answer_question_by_mode(self, question: str, mode: str, show_thinking: bool = True) -> str:
        """Answer a question with the selected chat backend."""
        if mode == "analysis":
            prompt = f"Context:\n{self.analysis_summary}\n\nQuestion: {question}\n\n{self.QA_INSTRUCTION}"
            return self.ask_question(prompt, show_thinking=show_thinking)

        if mode == "evidence":
            evidence_trail = ""
            _ucid = self.use_case_id or "pipeline_defects_detection"
            trail_path = Path("out") / use_case_id / "agent" / "evidence_trail.txt"
            if trail_path.exists():
                with open(trail_path, "r") as f:
                    evidence_trail = f.read()

            prompt = f"Context:\n{evidence_trail}\n\nQuestion: {question}\n\n{self.QA_INSTRUCTION}"
            return self.ask_question(prompt, show_thinking=show_thinking)

        if mode == "sql":
            sql_query = self.generate_sql_query(question)
            results = self.execute_and_format_query(sql_query)
            return f"Generated SQL:\n{sql_query}\n\n{results}"

        return f"Unknown mode: {mode}"

    def run(self):
        """Main interaction loop."""
        print("\n" + "="*70)
        print("PACE INTERACTIVE CHAT")
        print("="*70)
        
        # Load configuration and LLM
        if not self.load_config():
            return
        
        # Load artifacts
        if not self.load_artifacts():
            print("\n⚠️ No analysis artifacts found. Run the pipeline first:")
            print("   python run_complete_pipeline.py --num-images 50 --device CPU")
            return
        
        print("\nType 'exit' or 'quit' to leave.\n")
        
        # Main interaction loop
        while True:
            custom_q = input("💬 Your question: ").strip()
            if custom_q.lower() in {"exit", "quit"}:
                  print("\n👋 Goodbye!")
                  break
            if not custom_q:
                continue

            detected_mode = classify_intent_keyword(custom_q)
            detected_mode = resolve_active_chat_mode(detected_mode, self.config)
            print(f"[Detected: {detected_mode}]")

            response = self.answer_question_by_mode(
                custom_q,
                detected_mode,
                show_thinking=True
            )
            print(f"\n{response}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for PACE pipeline analysis."
    )
    parser.add_argument(
        "--config",
        help="Path to a use-case config YAML, e.g. config/oil_gas_pipeline/config.yaml",
    )
    args = parser.parse_args()

    chat = InteractiveChat(config_path=args.config)
    chat.run()
