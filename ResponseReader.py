import json
from typing import Dict, Optional

class JsonlResponseReader:
    """Handles reading responses from JSONL files"""

    def __init__(self, jsonl_paths: Dict[str, str]):
        """Initialize with dictionary mapping model names to file paths"""
        self.paths = jsonl_paths
        self.responses: Dict[str, Dict[str, str]] = {}
        for model_name, path in jsonl_paths.items():
            self._load_jsonl(model_name, path)

    def _load_jsonl(self, model_name: str, path: str) -> None:
        """Load the JSONL file and store responses for a specific model"""
        try:
            self.responses[model_name] = {}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.responses[model_name][entry["question_id"]] = entry["text"]
            print(f"Loaded {len(self.responses[model_name])} responses from {path} for {model_name}")
        except FileNotFoundError:
            print(f"Error: Could not find JSONL file {path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {path}")
            raise
        except Exception as e:
            print(f"Error loading JSONL file {path}: {str(e)}")
            raise

    def get_response(self, model_name: str, question_id: str) -> Optional[str]:
        """Get model response for specific question ID"""
        if model_name not in self.responses:
            return None
        return self.responses[model_name].get(question_id)
