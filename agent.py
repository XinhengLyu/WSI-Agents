from autogen_core import MessageContext, RoutedAgent, TopicId, message_handler, type_subscription
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from base_models import TaskAllocationRequest, TaskAllocation
import json


@type_subscription(topic_type="task_allocation")
class TaskAllocationAgent(RoutedAgent):
    """Agent responsible for analyzing questions and allocating to specialized agents"""

    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Analyzes questions and allocates tasks to specialized agents")
        self._model_client = model_client
        self._system_message = SystemMessage(content="""You are a medical task allocation expert.
            Analyze questions about medical images and determine the most appropriate agent:

            1. Morphology Agent: For questions about morphological characteristics, tissue structure,
               cellular patterns, architectural organization, and visual features.
               Example keywords: structure, pattern, appearance, organization, features

            2. Diagnosis Agent: For questions about diagnosis, disease classification,
               pathological conditions, and clinical implications.
               Example keywords: diagnosis, disease, condition, finding

            3. Treatment Agent: For questions about staging, treatment planning,
               clinical management, and prognosis.
               Example keywords: TNM, staging, treatment, management, prognosis,
               molecular markers, immunohistochemistry, biomarkers, therapeutic options, subtype

            4. Report Agent: For pathology report generation tasks that require producing
               a structured or free-text pathology report from WSI findings.
               Example keywords: report, generate report, write report, pathology report, summarize findings

            Your output MUST be a JSON with this format:
            {
                "agent_type": "morphology" or "diagnosis" or "treatment" or "report",
                "confidence": float between 0-1,
                "reasoning": "detailed explanation of choice"
            }""")

    @message_handler
    async def handle_allocation_request(self, message: TaskAllocationRequest, ctx: MessageContext) -> None:
        try:
            allocation_prompt = f"""
            Please analyze the following question and determine which agent should handle it:
            Morphology Agent, Diagnosis Agent, Treatment Agent, or Report Agent.

            Question ID: {message.question_id}
            Question: {message.question}

            Consider the question type carefully and provide detailed reasoning for your choice.
            Focus on keywords and implied requirements in the question.
            """

            response = await self._model_client.create(
                messages=[
                    self._system_message,
                    UserMessage(content=allocation_prompt, source=self.id.type)
                ],
                json_output=True,
                cancellation_token=ctx.cancellation_token
            )

            result = json.loads(response.content)
            # Validate agent_type; default to morphology if unrecognised
            if result.get('agent_type') not in ("morphology", "diagnosis", "treatment", "report"):
                result['agent_type'] = "morphology"
            print(f"\n=== Task Allocation Decision ===")
            print(f"Question: {message.question}")
            print(f"Allocated to: {result['agent_type']} agent")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Reasoning: {result['reasoning']}")

            # Create and publish allocation
            allocation = TaskAllocation(
                agent_type=result["agent_type"],
                question_id=message.question_id,
                question=message.question,
            )

            await self.publish_message(
                allocation,
                TopicId(result["agent_type"], source=self.id.key)
            )

        except Exception as e:
            print(f"Error in TaskAllocationAgent: {str(e)}")
            raise
