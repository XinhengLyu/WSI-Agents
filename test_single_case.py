"""
Quick smoke test: run exactly 1 unprocessed case from a given task type
to verify the full pipeline works end-to-end.

Usage:
    python test_single_case.py [task_name] [questions_file] [answers_file]

Defaults to Treatment task.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from MedicalAnalysisSystem import (
    MedicalAnalysisSystem, read_jsonl
)


async def main():
    task_name      = sys.argv[1] if len(sys.argv) > 1 else "Treatment"
    questions_file = sys.argv[2] if len(sys.argv) > 2 else "Treatment-questions.jsonl"
    answers_file   = sys.argv[3] if len(sys.argv) > 3 else "Treatment-mllm3-answers.jsonl"

    print(f"=== Smoke test: {task_name} ===")
    Config.configure_task(task_name, questions_file, answers_file)

    test_cases = read_jsonl(Config.QUESTIONS_PATH)
    if not test_cases:
        print("No unprocessed cases — task already complete.")
        return

    # Take only the first unprocessed case
    case = test_cases[-1]   # reversed() order matches production run
    print(f"Testing with case: {case['id']}")
    print(f"Output will be appended to: {Config.REFINED_RESPONSES_PATH}")

    system = MedicalAnalysisSystem()
    if not await system.setup():
        print("System setup failed.")
        return

    await system.analyze(case['id'], case['question'])
    print("\n=== Smoke test finished ===")

    # Show what was written
    if os.path.exists(Config.REFINED_RESPONSES_PATH):
        lines = open(Config.REFINED_RESPONSES_PATH).readlines()
        # Find the line for this case
        for line in lines:
            d = json.loads(line)
            if d['question_id'] == case['id']:
                print(f"Output saved: {d['text'][:150]}...")
                break


if __name__ == "__main__":
    asyncio.run(main())
