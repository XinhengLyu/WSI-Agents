import asyncio
import json
import os

from autogen_core import SingleThreadedAgentRuntime, TopicId
from InternalValidation import InternalConsistencyAgent
from ExternalValidation import ClassifierVerificationAgent, KnowledgeVerificationAgent
from IntegrationAgent import IntegrationAgent
from MLLM_agent import MorphologyAgent, DiagnosisAgent, TreatmentAgent, ReportAgent
from agent import TaskAllocationAgent, VerificationPanelAgent, AnswerSelectionAgent
from base_models import TaskAllocationRequest
from model_client import create_model_client
from knowledge_base import MedicalKnowledgeBuild, MedicalKnowledgeBase
from ResponseReader import JsonlResponseReader
from ClassifierResultsReader import ClassifierResponseReader
from config import Config


class MedicalAnalysisSystem:
    """Multi-agent medical image analysis system"""

    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize the medical analysis system"""
        self.model_client = create_model_client(model_name)
        self.model_clients = {
            "gpt-4o":   create_model_client("gpt-4o"),
            "Claude":   create_model_client("gpt-4o"),
            "Gemini":   create_model_client("gpt-4o"),
            "DeepSeek": create_model_client("gpt-4o"),
        }

        # Knowledge base — use absolute paths from Config
        knowledge_build = MedicalKnowledgeBuild(
            base_dir=Config.KB_BASE_DIR,
            persist_directory=Config.KB_DIR,
        )
        qa_chain = knowledge_build.create_qa_chain()
        self.knowledge_base = MedicalKnowledgeBase(qa_chain)

        # MLLM response reader — reloaded each time configure_task changes paths
        self.response_reader = JsonlResponseReader(Config.MLLM_PATHS)

        # Classifier reader — file names match what's on disk
        self.classifier_reader = ClassifierResponseReader(
            base_path=Config.CLASSIFIER_DIR + "/"
        )
        self.classifier_models = ["Conch", "MIZero", "TITAN"]

    async def setup(self):
        """Setup all agents"""
        try:
            self.runtime = SingleThreadedAgentRuntime()
            await self._register_agents()
            print("All agents registered successfully.")
            return True
        except Exception as e:
            print(f"Error during system setup: {str(e)}")
            return False

    async def analyze(self, question_id: str, question: str) -> None:
        """Run analysis workflow"""
        try:
            print(f"\n=== Starting Analysis for Question {question_id} ===")
            print(f"Question: {question}")

            self.runtime.start()

            request = TaskAllocationRequest(
                question_id=question_id,
                question=question,
            )

            await self.runtime.publish_message(
                request,
                TopicId("task_allocation", source="system")
            )

            await self.runtime.stop_when_idle()
            print("\nAnalysis workflow completed.")
            return None

        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            raise
        finally:
            try:
                await self.runtime.stop()
                print("Runtime stopped.")
                await self.runtime.close()
                print("Runtime closed.")
            except Exception as e:
                print(f"Warning: Error during runtime cleanup: {str(e)}")

    async def _register_agents(self):
        """Register all required agents"""
        await TaskAllocationAgent.register(
            self.runtime, "task_allocation",
            lambda: TaskAllocationAgent(self.model_client)
        )
        print("Task allocation agent registered.")

        await IntegrationAgent.register(
            self.runtime, "integration",
            lambda: IntegrationAgent(self.model_clients)
        )
        print("Integration agent registered.")

        await InternalConsistencyAgent.register(
            self.runtime, "consistency",
            lambda: InternalConsistencyAgent(self.model_client)
        )
        print("Consistency evaluation agent registered.")

        await KnowledgeVerificationAgent.register(
            self.runtime, "verification",
            lambda: KnowledgeVerificationAgent(self.model_client, self.knowledge_base)
        )
        print("Knowledge verification agent registered.")

        await MorphologyAgent.register(
            self.runtime, "morphology",
            lambda: MorphologyAgent(self.model_client, self.response_reader)
        )
        print("Morphology agent registered.")

        await DiagnosisAgent.register(
            self.runtime, "diagnosis",
            lambda: DiagnosisAgent(self.model_client, self.response_reader)
        )
        print("Diagnosis agent registered.")

        await ClassifierVerificationAgent.register(
            self.runtime, "classifier_verification",
            lambda: ClassifierVerificationAgent(
                self.model_client, self.classifier_reader, self.classifier_models
            )
        )
        print("Classifier verification agent registered.")

        await TreatmentAgent.register(
            self.runtime, "treatment",
            lambda: TreatmentAgent(self.model_client, self.response_reader)
        )
        print("Treatment agent registered.")

        await ReportAgent.register(
            self.runtime, "report",
            lambda: ReportAgent(self.model_client, self.response_reader)
        )
        print("Report agent registered.")

        await AnswerSelectionAgent.register(
            self.runtime, "answer_selection",
            lambda: AnswerSelectionAgent(self.model_client)
        )
        print("Answer selection agent registered.")

        await VerificationPanelAgent.register(
            self.runtime, "answer_verification",
            lambda: VerificationPanelAgent(self.model_clients, self.model_client)
        )
        print("Answer verification agent registered.")

    def _format_final_result(self, result: IntegrationResult) -> str:
        output = [
            "\n=== Final Analysis Report ===\n",
            "Model Reliability Scores:",
            f"MLLM_1: {result.mllm1_reliability.final_score:.2f}",
            f"MLLM_2: {result.mllm2_reliability.final_score:.2f}",
            f"MLLM_3: {result.mllm3_reliability.final_score:.2f}\n",
            "Final Conclusion:",
            result.final_conclusion,
            "\nSupporting Evidence:",
        ]
        for evidence in result.supporting_evidence:
            output.append(f"- {evidence}")
        return "\n".join(output)


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_jsonl(file_path: str, skip_processed: bool = True) -> list:
    """Read JSONL question file and return list of cases, skipping already-done ones."""
    test_cases = []
    processed_ids = Config.get_processed_ids() if skip_processed else set()
    if processed_ids:
        print(f"Found {len(processed_ids)} previously processed cases — skipping.")

    try:
        with open(file_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    test_case = json.loads(line.strip())
                    question_id = test_case["question_id"]
                    if skip_processed and question_id in processed_ids:
                        continue
                    test_cases.append({
                        "id": question_id,
                        "question": test_case["prompt"],
                    })
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line: {line.strip()} — {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return test_cases


async def run_task(task_name: str, questions_file: str, answers_file: str):
    """Run the full analysis pipeline for one task type."""
    print(f"\n{'=' * 70}")
    print(f"  TASK: {task_name}")
    print(f"  Questions : {questions_file}")
    print(f"  MLLM_3    : {answers_file}")
    print(f"{'=' * 70}")

    Config.configure_task(task_name, questions_file, answers_file)

    missing = [p for p in [Config.QUESTIONS_PATH] + list(Config.MLLM_PATHS.values())
               if not os.path.exists(p)]
    if missing:
        print(f"[SKIP] Missing files: {missing} — skipping task {task_name}.")
        return

    system = MedicalAnalysisSystem()
    if not await system.setup():
        print(f"System setup failed for task {task_name}.")
        return

    test_cases = read_jsonl(Config.QUESTIONS_PATH)
    if not test_cases:
        print(f"No remaining cases for task {task_name}.")
        return

    print(f"Running {len(test_cases)} cases for task {task_name} → {Config.REFINED_RESPONSES_PATH}")
    for case in reversed(test_cases):
        print(f"\nProcessing {case['id']}")
        try:
            await system.analyze(case['id'], case['question'])
        except Exception as e:
            print(f"Failed to process case {case['id']}: {str(e)}")
            continue


async def main():
    """Run a single task type. For batch runs use run_experiments.py instead."""
    import sys
    # CLI: python MedicalAnalysisSystem.py Treatment Treatment-questions.jsonl Treatment-mllm3-answers.jsonl
    if len(sys.argv) == 4:
        task_name, questions_file, answers_file = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        task_name      = "Treatment"
        questions_file = "Treatment-questions.jsonl"
        answers_file   = "Treatment-mllm3-answers.jsonl"

    await run_task(task_name, questions_file, answers_file)


if __name__ == "__main__":
    asyncio.run(main())
