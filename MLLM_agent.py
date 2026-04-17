import asyncio
from autogen_core import MessageContext, RoutedAgent, TopicId, message_handler, \
    type_subscription
from autogen_core.models import ChatCompletionClient, SystemMessage
from base_models import TaskAllocation, AnalysisTask
from ResponseReader import JsonlResponseReader
from config import Config


def _load_responses(
    response_reader: JsonlResponseReader,
    task_name: str,
    question_id: str,
) -> tuple[str, str, str]:
    """Read the three MLLM responses configured for this task.

    The keys to read are defined in Config.TASK_MLLM_KEYS[task_name].
    Returns (mllm1_response, mllm2_response, mllm3_response) mapped
    positionally to the first three keys in that list.
    """
    keys = Config.TASK_MLLM_KEYS[task_name]
    responses = [response_reader.get_response(k, question_id) for k in keys]
    missing = [keys[i] for i, r in enumerate(responses) if not r]
    if missing:
        raise ValueError(f"Missing responses for: {', '.join(missing)}")
    return responses[0], responses[1], responses[2]


@type_subscription(topic_type="morphology")
class MorphologyAgent(RoutedAgent):
    """Expert agent for morphological characteristic questions"""

    def __init__(self, model_client: ChatCompletionClient, response_reader: JsonlResponseReader):
        super().__init__(description="Analyzes morphological features of medical images")
        self._model_client = model_client
        self._response_reader = response_reader
        self._system_message = SystemMessage(content="")

    @message_handler
    async def handle_task_allocation(self, message: TaskAllocation, ctx: MessageContext) -> None:
        try:
            print(f"\n=== Morphological Analysis: {message.question_id} ===")
            r1, r2, r3 = _load_responses(self._response_reader, "Morphology", message.question_id)

            analysis_task = AnalysisTask(
                question_id=message.question_id,
                question=message.question,
                mllm1_response=r1,
                mllm2_response=r2,
                mllm3_response=r3,
            )

            await asyncio.gather(
                self.publish_message(analysis_task, TopicId("consistency",             source=message.question_id)),
                self.publish_message(analysis_task, TopicId("verification",            source=message.question_id)),
                self.publish_message(analysis_task, TopicId("classifier_verification", source=message.question_id)),
                self.publish_message(analysis_task, TopicId("integration",             source=message.question_id)),
            )

        except Exception as e:
            print(f"Error in MorphologyAgent: {str(e)}")
            raise


@type_subscription(topic_type="diagnosis")
class DiagnosisAgent(RoutedAgent):
    """Expert agent for diagnostic questions"""

    def __init__(self, model_client: ChatCompletionClient, response_reader: JsonlResponseReader):
        super().__init__("Analyzes diagnostic findings from medical images")
        self._model_client = model_client
        self._response_reader = response_reader
        self._system_message = SystemMessage(content="""You are a medical diagnosis expert.
            Analyze and synthesize diagnostic findings from multiple models.
            Focus on:
            1. Disease classification and identification
            2. Pathological conditions and abnormalities
            3. Clinical implications and significance
            4. Diagnostic confidence and supporting evidence

            Output format:
            {
                "diagnostic_findings": {
                    "primary_diagnosis": {
                        "condition": "identified condition",
                        "confidence": float between 0-1,
                        "evidence": ["supporting evidence"]
                    },
                    "differential_diagnoses": [
                        {
                            "condition": "alternative diagnosis",
                            "likelihood": float between 0-1,
                            "rationale": "reasoning"
                        }
                    ],
                    "additional_findings": ["other relevant observations"]
                },
                "synthesis": "comprehensive diagnostic assessment"
            }""")

    @message_handler
    async def handle_task_allocation(self, message: TaskAllocation, ctx: MessageContext) -> None:
        try:
            print(f"\n=== Diagnostic Analysis: {message.question_id} ===")
            r1, r2, r3 = _load_responses(self._response_reader, "Diagnosis", message.question_id)

            analysis_task = AnalysisTask(
                question_id=message.question_id,
                question=message.question,
                mllm1_response=r1,
                mllm2_response=r2,
                mllm3_response=r3,
            )

            await asyncio.gather(
                self.publish_message(analysis_task, TopicId("consistency",             source=message.question_id)),
                self.publish_message(analysis_task, TopicId("verification",            source=message.question_id)),
                self.publish_message(analysis_task, TopicId("classifier_verification", source=message.question_id)),
                self.publish_message(analysis_task, TopicId("integration",             source=message.question_id)),
            )

        except Exception as e:
            print(f"Error in DiagnosisAgent: {str(e)}")
            raise


@type_subscription(topic_type="treatment")
class TreatmentAgent(RoutedAgent):
    """Expert agent for treatment planning, TNM staging, prognosis, and subtype questions"""

    def __init__(self, model_client: ChatCompletionClient, response_reader: JsonlResponseReader):
        super().__init__(description="Analyzes treatment-related responses with full verification")
        self._model_client = model_client
        self._response_reader = response_reader

    @message_handler
    async def handle_task_allocation(self, message: TaskAllocation, ctx: MessageContext) -> None:
        try:
            print(f"\n=== Treatment Analysis: {message.question_id} ===")
            r1, r2, r3 = _load_responses(self._response_reader, "Treatment", message.question_id)

            analysis_task = AnalysisTask(
                question_id=message.question_id,
                question=message.question,
                mllm1_response=r1,
                mllm2_response=r2,
                mllm3_response=r3,
            )

            # Treatment does not use classifier verification (no tissue-type alignment needed)
            await asyncio.gather(
                self.publish_message(analysis_task, TopicId("consistency",  source=message.question_id)),
                self.publish_message(analysis_task, TopicId("verification", source=message.question_id)),
                self.publish_message(analysis_task, TopicId("integration",  source=message.question_id)),
            )

        except Exception as e:
            print(f"Error in TreatmentAgent: {str(e)}")
            raise


@type_subscription(topic_type="report")
class ReportAgent(RoutedAgent):
    """Expert agent for pathology report generation questions"""

    def __init__(self, model_client: ChatCompletionClient, response_reader: JsonlResponseReader):
        super().__init__(description="Generates pathology reports with full verification")
        self._model_client = model_client
        self._response_reader = response_reader

    @message_handler
    async def handle_task_allocation(self, message: TaskAllocation, ctx: MessageContext) -> None:
        try:
            print(f"\n=== Report Generation: {message.question_id} ===")
            r1, r2, r3 = _load_responses(self._response_reader, "Report", message.question_id)

            analysis_task = AnalysisTask(
                question_id=message.question_id,
                question=message.question,
                mllm1_response=r1,
                mllm2_response=r2,
                mllm3_response=r3,
            )

            await asyncio.gather(
                self.publish_message(analysis_task, TopicId("consistency",             source=message.question_id)),
                self.publish_message(analysis_task, TopicId("verification",            source=message.question_id)),
                self.publish_message(analysis_task, TopicId("classifier_verification", source=message.question_id)),
                self.publish_message(analysis_task, TopicId("integration",             source=message.question_id)),
            )

        except Exception as e:
            print(f"Error in ReportAgent: {str(e)}")
            raise
