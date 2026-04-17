# config.py
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))

class Config:
    # ── Root data directory (parent of WSI_Agents/) ───────────────────────
    DATA_ROOT = os.path.join(_HERE, "..")

    # ── Directories ───────────────────────────────────────────────────────
    OUTPUT_DIR      = os.path.join(DATA_ROOT, "output")
    QUESTIONS_DIR   = os.path.join(DATA_ROOT, "questions")
    MLLM_OUTPUT_DIR = os.path.join(DATA_ROOT, "MLLMs_output")
    CLASSIFIER_DIR  = os.path.join(DATA_ROOT, "classifier_outputs")
    KB_DIR          = os.path.join(DATA_ROOT, "medical_kb_structured")
    KB_BASE_DIR     = os.path.join(DATA_ROOT, "medical_knowledge_base")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── MLLM answer files (one file per model, shared across all tasks) ───
    # Each file contains answers for every question_id.
    # Edit the filenames here to point to your actual model output files.
    MLLM_PATHS = {
        "mllm_1": os.path.join(MLLM_OUTPUT_DIR, "mllm1-answers.jsonl"),
        "mllm_2": os.path.join(MLLM_OUTPUT_DIR, "mllm2-answers.jsonl"),
        "mllm_3": os.path.join(MLLM_OUTPUT_DIR, "mllm3-answers.jsonl"),
    }

    # ── Which MLLM keys each task reads ───────────────────────────────────
    # All four tasks currently use the same three models.
    # To use a different subset for a task, edit the list here.
    TASK_MLLM_KEYS = {
        "Morphology": ["mllm_1", "mllm_2", "mllm_3"],
        "Diagnosis":  ["mllm_1", "mllm_2", "mllm_3"],
        "Treatment":  ["mllm_1", "mllm_2", "mllm_3"],
        "Report":     ["mllm_1", "mllm_2", "mllm_3"],
    }

    # ── Per-task state (updated by configure_task) ────────────────────────
    _task_name             = ""
    QUESTIONS_PATH         = ""   # input: question list {"question_id", "prompt"}
    REFINED_RESPONSES_PATH = ""   # output {"question_id", "text"}

    # ── Task configuration ────────────────────────────────────────────────
    @classmethod
    def configure_task(cls, task_name: str, questions_file: str):
        """Switch Config to a specific task type.

        Args:
            task_name:      One of "Morphology", "Diagnosis", "Treatment", "Report"
            questions_file: JSONL filename in questions/ {"question_id", "prompt"}
        """
        if task_name not in cls.TASK_MLLM_KEYS:
            raise ValueError(f"Unknown task '{task_name}'. Valid: {list(cls.TASK_MLLM_KEYS)}")

        cls._task_name             = task_name
        cls.QUESTIONS_PATH         = os.path.join(cls.QUESTIONS_DIR, questions_file)
        cls.REFINED_RESPONSES_PATH = os.path.join(
            cls.OUTPUT_DIR, f"refined_responses_{task_name}.jsonl"
        )

    # ── Processed-ID helper ───────────────────────────────────────────────
    @classmethod
    def get_processed_ids(cls) -> set:
        """Return set of already-processed question_ids for the current task."""
        processed: set = set()
        if os.path.exists(cls.REFINED_RESPONSES_PATH):
            try:
                with open(cls.REFINED_RESPONSES_PATH, 'r', encoding='utf-8') as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            processed.add(json.loads(line)['question_id'])
            except Exception:
                pass
        return processed

    # ── Path update helpers ───────────────────────────────────────────────
    @classmethod
    def update_refined_responses_path(cls, new_path: str):
        cls.REFINED_RESPONSES_PATH = new_path
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

    @classmethod
    def update_questions_path(cls, new_path: str):
        cls.QUESTIONS_PATH = new_path

    @classmethod
    def update_mllm_path(cls, model_key: str, new_path: str):
        """Override a single MLLM slot at runtime.

        Example:
            Config.update_mllm_path("mllm_1", "/path/to/MLLMs_output/mllm1-answers.jsonl")
        """
        cls.MLLM_PATHS[model_key] = new_path
