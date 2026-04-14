import asyncio
from autogen_core import MessageContext, RoutedAgent, TopicId, message_handler, \
    type_subscription
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from base_models import  TaskAllocation,AnalysisTask
from ResponseReader import JsonlResponseReader

@type_subscription(topic_type="morphology")
class MorphologyAgent(RoutedAgent):
    """Agent for analyzing morphological characteristics"""

    def __init__(self, model_client: ChatCompletionClient, response_reader: JsonlResponseReader):
        # Call parent class constructor with the description
        super().__init__(description="Analyzes morphological features of medical images")
        self._model_client = model_client
        self._response_reader = response_reader
        self._system_message = SystemMessage(content="")

    @message_handler
    async def handle_task_allocation(self, message: TaskAllocation, ctx: MessageContext) -> None:
        try:
            print(f"\n=== Morphological Analysis: {message.question_id} ===")

            # Get responses from models
            mllm1_response = self._response_reader.get_response("mllm_1", message.question_id)
            mllm2_response = self._response_reader.get_response("mllm_2", message.question_id)
            mllm3_response = self._response_reader.get_response("mllm_3", message.question_id)

            if not all([mllm1_response, mllm2_response, mllm3_response]):
                print("Warning: Missing model responses")
                missing = []
                if not mllm1_response: missing.append("MLLM_1")
                if not mllm2_response: missing.append("MLLM_2")
                if not mllm3_response: missing.append("MLLM_3")
                print(f"Missing responses from: {', '.join(missing)}")
                raise ValueError("Missing required model responses")

            # Create analysis task
            analysis_task = AnalysisTask(
                question_id=message.question_id,
                question=message.question,
                mllm1_response=mllm1_response,
                mllm2_response=mllm2_response,
                mllm3_response=mllm3_response,

            )

            # Send to both agents in parallel
            await asyncio.gather(
                self.publish_message(
                    analysis_task,
                    TopicId("consistency", source=message.question_id)
                ),
                self.publish_message(
                    analysis_task,
                    TopicId("verification", source=message.question_id)
                ),
                self.publish_message(
                    analysis_task,
                    TopicId("integration", source=message.question_id)
                ),
                self.publish_message(
                    analysis_task,
                    TopicId("classifier_verification", source=message.question_id)
                )
            )

            print(f"Morphological analysis task sent for both consistency and knowledge verification")

        except Exception as e:
            print(f"Error in MorphologyAgent: {str(e)}")
            raise



@type_subscription(topic_type="treatment")
class TreatmentAgent(RoutedAgent):
    """Agent for treatment planning, TNM staging, prognosis, and subtype questions"""

    def __init__(self, model_client: ChatCompletionClient, response_reader: JsonlResponseReader):
        super().__init__(description="Analyzes treatment-related responses with full verification")
        self._model_client = model_client
        self._response_reader = response_reader

    @message_handler
    async def handle_task_allocation(self, message: TaskAllocation, ctx: MessageContext) -> None:
        try:
            print(f"\n=== Treatment Analysis: {message.question_id} ===")

            mllm1_response = self._response_reader.get_response("mllm_1", message.question_id)
            mllm2_response = self._response_reader.get_response("mllm_2", message.question_id)
            mllm3_response = self._response_reader.get_response("mllm_3", message.question_id)
            mllm4_response = self._response_reader.get_response("mllm_4", message.question_id)

            if not all([mllm1_response, mllm2_response, mllm3_response]):
                missing = [k for k, v in {"MLLM_1": mllm1_response, "MLLM_2": mllm2_response,
                                          "MLLM_3": mllm3_response}.items() if not v]
                print(f"Missing responses from: {', '.join(missing)}")
                raise ValueError("Missing required model responses")

            analysis_task = AnalysisTask(
                question_id=message.question_id,
                question=message.question,
                mllm1_response=mllm1_response,
                mllm2_response=mllm2_response,
                mllm3_response=mllm3_response,
                mllm4_response=mllm4_response,

            )

            await asyncio.gather(
                self.publish_message(analysis_task, TopicId("consistency",             source=message.question_id)),
                self.publish_message(analysis_task, TopicId("verification",            source=message.question_id)),
                self.publish_message(analysis_task, TopicId("classifier_verification", source=message.question_id)),
                self.publish_message(analysis_task, TopicId("integration",             source=message.question_id)),
            )

            print(f"Treatment analysis task sent to all verification agents and integration")

        except Exception as e:
            print(f"Error in TreatmentAgent: {str(e)}")
            raise


@type_subscription(topic_type="report")
class ReportAgent(RoutedAgent):
    """Agent for pathology report generation questions"""

    def __init__(self, model_client: ChatCompletionClient, response_reader: JsonlResponseReader):
        super().__init__(description="Generates pathology reports with full verification")
        self._model_client = model_client
        self._response_reader = response_reader

    @message_handler
    async def handle_task_allocation(self, message: TaskAllocation, ctx: MessageContext) -> None:
        try:
            print(f"\n=== Report Generation: {message.question_id} ===")

            mllm1_response = self._response_reader.get_response("mllm_1", message.question_id)
            mllm2_response = self._response_reader.get_response("mllm_2", message.question_id)
            mllm3_response = self._response_reader.get_response("mllm_3", message.question_id)

            if not all([mllm1_response, mllm2_response, mllm3_response]):
                missing = [k for k, v in {"MLLM_1": mllm1_response, "MLLM_2": mllm2_response,
                                          "MLLM_3": mllm3_response}.items() if not v]
                print(f"Missing responses from: {', '.join(missing)}")
                raise ValueError("Missing required model responses")

            analysis_task = AnalysisTask(
                question_id=message.question_id,
                question=message.question,
                mllm1_response=mllm1_response,
                mllm2_response=mllm2_response,
                mllm3_response=mllm3_response,

            )

            await asyncio.gather(
                self.publish_message(analysis_task, TopicId("consistency",             source=message.question_id)),
                self.publish_message(analysis_task, TopicId("verification",            source=message.question_id)),
                self.publish_message(analysis_task, TopicId("classifier_verification", source=message.question_id)),
                self.publish_message(analysis_task, TopicId("integration",             source=message.question_id)),
            )

            print(f"Report generation task sent to all verification agents and integration")

        except Exception as e:
            print(f"Error in ReportAgent: {str(e)}")
            raise



@type_subscription(topic_type="diagnosis")
class DiagnosisAgent(RoutedAgent):
    """Agent for analyzing diagnostic aspects"""

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

            # Get responses from models
            mllm1_response = self._response_reader.get_response("mllm_1", message.question_id)
            mllm2_response = self._response_reader.get_response("mllm_2", message.question_id)
            mllm3_response = self._response_reader.get_response("mllm_3", message.question_id)

            if not all([mllm1_response, mllm2_response, mllm3_response]):
                print("Warning: Missing model responses")
                missing = []
                if not mllm1_response: missing.append("MLLM_1")
                if not mllm2_response: missing.append("MLLM_2")
                if not mllm3_response: missing.append("MLLM_3")
                print(f"Missing responses from: {', '.join(missing)}")
                raise ValueError("Missing required model responses")

            analysis_task = AnalysisTask(
                question_id=message.question_id,
                question=message.question,

                mllm1_response=mllm1_response,
                mllm2_response=mllm2_response,
                mllm3_response=mllm3_response,
            )

            # Fan out to all verification agents + integration (same as MorphologyAgent)
            await asyncio.gather(
                self.publish_message(
                    analysis_task,
                    TopicId("consistency", source=message.question_id)
                ),
                self.publish_message(
                    analysis_task,
                    TopicId("verification", source=message.question_id)
                ),
                self.publish_message(
                    analysis_task,
                    TopicId("classifier_verification", source=message.question_id)
                ),
                self.publish_message(
                    analysis_task,
                    TopicId("integration", source=message.question_id)
                ),
            )

            print(f"Diagnostic analysis task sent to all verification agents and integration")

        except Exception as e:
            print(f"Error in DiagnosisAgent: {str(e)}")
            raise