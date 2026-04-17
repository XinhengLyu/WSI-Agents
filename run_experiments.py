"""
Batch experiment runner.

Iterates through the four task types and runs MedicalAnalysisSystem for each one.
Already-processed cases are automatically skipped (resume-safe).

Usage:
    # Run all four task types
    python run_experiments.py

    # Run specific task types
    python run_experiments.py Morphology Treatment

    # Dry-run: show what would be run without executing
    python run_experiments.py --dry-run
"""

import asyncio
import sys
import os

# Make sure imports resolve from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MedicalAnalysisSystem import run_task

# ── Task catalogue ────────────────────────────────────────────────────────────
# Four task types, one per pipeline route.
# Update the JSONL filenames here to point to your answer files.
# (task_name, questions_file)
# questions_file → questions/  {"question_id", "prompt"}
# MLLM answer files are shared across all tasks — configure in config.py
ALL_TASKS = [
    ("Morphology", "Morphology-questions.jsonl"),
    ("Diagnosis",  "Diagnosis-questions.jsonl"),
    ("Treatment",  "Treatment-questions.jsonl"),
    ("Report",     "Report-questions.jsonl"),
]

TASK_MAP = {name: qf for name, qf in ALL_TASKS}


async def main():
    args = sys.argv[1:]

    # ── dry-run mode ─────────────────────────────────────────────────────────
    if "--dry-run" in args:
        from config import Config
        import json
        print("Dry-run mode — showing task list:\n")
        print("%-16s %8s %8s %8s  %s" % ("Task", "Total", "Done", "Left", "Output file"))
        print("-" * 72)
        for name, qf in ALL_TASKS:
            Config.configure_task(name, qf)
            total = sum(1 for _ in open(Config.QUESTIONS_PATH, encoding='utf-8')) if os.path.exists(Config.QUESTIONS_PATH) else 0
            done = len(Config.get_processed_ids())
            left = total - done
            out_name = os.path.basename(Config.REFINED_RESPONSES_PATH)
            print("%-16s %8d %8d %8d  %s" % (name, total, done, left, out_name))
        return

    # ── select tasks ─────────────────────────────────────────────────────────
    if args:
        tasks_to_run = []
        for name in args:
            if name in TASK_MAP:
                tasks_to_run.append((name, TASK_MAP[name]))
            else:
                print(f"Unknown task '{name}'. Valid names: {list(TASK_MAP)}")
                sys.exit(1)
    else:
        tasks_to_run = ALL_TASKS

    # ── run ───────────────────────────────────────────────────────────────────
    print(f"Running {len(tasks_to_run)} task type(s): {[t[0] for t in tasks_to_run]}\n")
    for task_name, questions_file in tasks_to_run:
        try:
            await run_task(task_name, questions_file)
        except Exception as e:
            print(f"\n[ERROR] Task '{task_name}' failed: {e}")
            print("Continuing with next task...\n")
            continue

    print("\nAll tasks finished.")


if __name__ == "__main__":
    asyncio.run(main())
