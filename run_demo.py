"""
Demo runner — runs exactly 1 case for each of the four task types and writes
results to demo_output/ (does not touch the production output/ directory).

This is useful for verifying the full pipeline end-to-end after code changes.

Usage:
    # Demo all four task types
    python run_demo.py

    # Demo specific task types
    python run_demo.py Morphology Report
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from MedicalAnalysisSystem import MedicalAnalysisSystem, read_jsonl

# ── Task catalogue (same as run_experiments.py) ───────────────────────────────
ALL_TASKS = [
    ("Morphology", "Morphology-questions.jsonl"),
    ("Diagnosis",  "Diagnosis-questions.jsonl"),
    ("Treatment",  "Treatment-questions.jsonl"),
    ("Report",     "Report-questions.jsonl"),
]

TASK_MAP = {name: qf for name, qf in ALL_TASKS}

# ── Demo output directory (separate from production output/) ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_OUTPUT_DIR = os.path.join(_HERE, "..", "demo_output")
os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)


async def demo_task(task_name: str, questions_file: str) -> bool:
    """Run 1 case for a task type, writing to demo_output/.

    Returns True if a case was found and processed, False otherwise.
    """
    print(f"\n{'=' * 70}")
    print(f"  DEMO: {task_name}  ({questions_file})")
    print(f"{'=' * 70}")

    Config.configure_task(task_name, questions_file)
    demo_output_path = os.path.join(DEMO_OUTPUT_DIR, f"demo_{task_name}.jsonl")
    Config.REFINED_RESPONSES_PATH = demo_output_path

    if not os.path.exists(Config.QUESTIONS_PATH):
        print(f"[SKIP] Questions file not found: {Config.QUESTIONS_PATH}")
        return False

    # Load all cases (skip_processed=False so we always get at least 1 case)
    cases = read_jsonl(Config.QUESTIONS_PATH, skip_processed=False)
    if not cases:
        print(f"[SKIP] No cases found for {task_name}.")
        return False

    # Pick the first case
    case = cases[-1]
    print(f"Case ID  : {case['id']}")
    print(f"Question : {case['question'][:120]}...")
    print(f"Output   : {demo_output_path}")

    system = MedicalAnalysisSystem()
    if not await system.setup():
        print(f"[FAIL] System setup failed for {task_name}.")
        return False

    try:
        await system.analyze(case['id'], case['question'])
    except Exception as e:
        print(f"[FAIL] {task_name} demo error: {e}")
        return False

    # Print a snippet of the result
    if os.path.exists(demo_output_path):
        with open(demo_output_path) as f:
            for line in f:
                d = json.loads(line)
                if d['question_id'] == case['id']:
                    preview = d['text'][:200].replace('\n', ' ')
                    print(f"\nResult preview: {preview}...")
                    break

    return True


async def main():
    args = sys.argv[1:]

    if args:
        tasks_to_run = []
        for name in args:
            if name in TASK_MAP:
                tasks_to_run.append((name, TASK_MAP[name]))
            else:
                print(f"Unknown task '{name}'. Valid: {list(TASK_MAP)}")
                sys.exit(1)
    else:
        tasks_to_run = ALL_TASKS

    print(f"Running demo for {len(tasks_to_run)} task type(s). Output → demo_output/")

    results = {}
    for task_name, questions_file in tasks_to_run:
        try:
            ok = await demo_task(task_name, questions_file)
            results[task_name] = "OK" if ok else "SKIPPED"
        except Exception as e:
            print(f"\n[ERROR] {task_name}: {e}")
            results[task_name] = f"ERROR: {e}"

    print(f"\n{'=' * 70}")
    print("  DEMO SUMMARY")
    print(f"{'=' * 70}")
    for name, status in results.items():
        print(f"  {name:<16} {status}")


if __name__ == "__main__":
    asyncio.run(main())
