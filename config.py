# config.py
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))

class Config:
    # ── Root data directory (parent of open_source/) ──────────────────────
    DATA_ROOT = os.path.join(_HERE, "..")

    # ── Directories ───────────────────────────────────────────────────────
    OUTPUT_DIR      = os.path.join(DATA_ROOT, "output")
    QUESTIONS_DIR   = os.path.join(DATA_ROOT, "questions")       # question-only JSONL files
    MLLM_OUTPUT_DIR = os.path.join(DATA_ROOT, "MLLMs_output")    # per-model answer JSONL files
    CLASSIFIER_DIR  = os.path.join(DATA_ROOT, "classifier_outputs")
    KB_DIR          = os.path.join(DATA_ROOT, "medical_kb_structured")
    KB_BASE_DIR     = os.path.join(DATA_ROOT, "medical_knowledge_base")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Shared MLLM answer files (same across all tasks) ──────────────────
    # Swap these paths to change which model is used as MLLM_1 / MLLM_2.
    MLLM1_ANSWERS_PATH = os.path.join(MLLM_OUTPUT_DIR, "mllm1-answers.jsonl")
    MLLM2_ANSWERS_PATH = os.path.join(MLLM_OUTPUT_DIR, "mllm2-answers.jsonl")

    # ── Per-task state (updated by configure_task) ────────────────────────
    _task_name             = ""
    QUESTIONS_PATH         = ""   # input: question list  {"question_id", "prompt"}
    MLLM3_ANSWERS_PATH     = ""   # input: MLLM_3 answers {"question_id", "text"}
    MLLM4_ANSWERS_PATH     = ""   # input: MLLM_4 answers (optional, defaults to MLLM_3)
    REFINED_RESPONSES_PATH = ""   # output {"question_id", "text"}

    MLLM_PATHS: dict = {}

    # ── Task configuration ────────────────────────────────────────────────
    @classmethod
    def configure_task(cls, task_name: str, questions_file: str, answers_file: str):
        """Switch Config to a specific task type.

        Questions and MLLM_3 answers are kept in separate files.
        MLLM_1 and MLLM_2 are shared across all tasks.

        Args:
            task_name:      Human-readable label, e.g. "Treatment"
            questions_file: JSONL filename in MLLMs_output/ containing questions
                            {"question_id", "prompt"}
            answers_file:   JSONL filename in MLLMs_output/ containing MLLM_3 answers
                            {"question_id", "text"}
        """
        cls._task_name         = task_name
        cls.QUESTIONS_PATH     = os.path.join(cls.QUESTIONS_DIR,   questions_file)
        cls.MLLM3_ANSWERS_PATH = os.path.join(cls.MLLM_OUTPUT_DIR, answers_file)
        cls.MLLM4_ANSWERS_PATH = cls.MLLM3_ANSWERS_PATH
        cls.MLLM_PATHS = {
            "mllm_1": cls.MLLM1_ANSWERS_PATH,
            "mllm_2": cls.MLLM2_ANSWERS_PATH,
            "mllm_3": cls.MLLM3_ANSWERS_PATH,
            "mllm_4": cls.MLLM4_ANSWERS_PATH,
        }
        cls.REFINED_RESPONSES_PATH = os.path.join(
            cls.OUTPUT_DIR, f"refined_responses_{task_name}.jsonl"
        )

    # ── Processed-ID helpers ──────────────────────────────────────────────
    @classmethod
    def get_processed_ids(cls) -> set:
        """Return set of already-processed question_ids for the current task.

        Checks both the new output file and legacy *_0219 / plain variants so
        that runs already completed under the old naming are not repeated.
        """
        name = cls._task_name
        candidates = [
            cls.REFINED_RESPONSES_PATH,
            os.path.join(cls.OUTPUT_DIR, f"refined_responses_{name}_0219.jsonl"),
            os.path.join(cls.OUTPUT_DIR, f"refined_responses_{name}_test.jsonl"),
        ]
        processed: set = set()
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
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
        """Update a single MLLM slot at runtime.

        Example:
            Config.update_mllm_path("mllm_1", "/path/to/new_model_answers.jsonl")
        """
        attr = {
            'mllm_1': 'MLLM1_ANSWERS_PATH',
            'mllm_2': 'MLLM2_ANSWERS_PATH',
            'mllm_3': 'MLLM3_ANSWERS_PATH',
            'mllm_4': 'MLLM4_ANSWERS_PATH',
        }.get(model_key)
        if attr:
            setattr(cls, attr, new_path)
        cls.MLLM_PATHS[model_key] = new_path
