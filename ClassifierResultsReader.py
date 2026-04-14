import json
from typing import Dict, List
from base_models import ClassifierPrediction

class ClassifierResponseReader:
    """Handles reading and processing classifier model predictions from JSONL files"""

    def __init__(self, base_path: str = ""):
        """Initialize with base path for classifier files"""
        self.base_path = base_path
        self.predictions: Dict[str, Dict[str, ClassifierPrediction]] = {}

    def _get_file_path(self, model_name: str) -> str:
        """Get the full file path for a model"""
        return f"{self.base_path}{model_name}.jsonl"

    def _load_model_predictions(self, model_name: str) -> None:
        """Load predictions for a specific model"""
        file_path = self._get_file_path(model_name)
        try:
            self.predictions[model_name] = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        prediction = ClassifierPrediction(
                            model_name=model_name,
                            label=entry["label"],
                            confidence=entry["confidence"]
                        )
                        self.predictions[model_name][entry["question_id"]] = prediction
            print(f"Loaded {len(self.predictions[model_name])} predictions for {model_name}")
        except Exception as e:
            print(f"Error loading predictions for {model_name}: {str(e)}")
            raise

    def get_predictions(self, model_names: List[str], question_id: str) -> List[ClassifierPrediction]:
        results = []

        def extract_search_key(question: str) -> str:
            if "_" in question:
                question = question.split("_", 1)[1]
            if len(question) > 4:
                question = question[:-4]
            return question

        search_key = extract_search_key(question_id)

        for model_name in model_names:
            if model_name not in self.predictions:
                try:
                    self._load_model_predictions(model_name)
                except Exception as e:
                    print(f"Failed to load predictions for model {model_name}: {e}")
                    continue

            if prediction := self.predictions.get(model_name, {}).get(search_key):
                results.append(prediction)

        if not results:
            print(f"No predictions found for {question_id} ({search_key})")

        return results