from autogen_core import MessageContext, RoutedAgent, TopicId, message_handler,type_subscription
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from base_models import *
import json
import asyncio
from typing import Dict

from config import Config


def parse_model_response(response, model_name):
    """
    Parse a model response, handling both JSON and plain-text formats.

    Args:
        response: model response object with a `.content` attribute
        model_name: reviewer model name (e.g. Claude, DeepSeek, GPT-4o, Gemini)

    Returns:
        dict with keys: verdict, reasoning, suggestion
    """
    print(f"Raw response from {model_name}: {response}")

    if not hasattr(response, "content"):
        print(f"{model_name} returned an unexpected format. No `.content` attribute found.")
        return {
            "verdict": "NAN",
            "reasoning": "Model response format invalid.",
            "suggestion": "Please check model output."
        }

    try:
        review_content = json.loads(response.content)
        return {
            "verdict": review_content.get("verdict", "reject"),
            "reasoning": review_content.get("reasoning", "No reasoning provided."),
            "suggestion": review_content.get("suggestion", "")
        }

    except json.JSONDecodeError:
        content = response.content.strip()
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx + 1]
            review_content = json.loads(json_str)
            verdict = review_content.get("verdict", "").lower()
            if verdict in ["approve", "accept", "yes", "agreed"]:
                verdict = "approve"
            else:
                verdict = "reject"
            return {
                "verdict": verdict,
                "reasoning": review_content.get("reasoning", "No reasoning provided."),
                "suggestion": review_content.get("suggestion", "")
            }

        # No JSON found — default to reject
        return {
            "verdict": "reject",
            "reasoning": content,
            "suggestion": ""
        }
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
            Please analyze the following question and determine whether it should be handled by the
            Morphology Agent or Diagnosis Agent:

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

@type_subscription(topic_type="answer_selection")
class AnswerSelectionAgent(RoutedAgent):
    """Agent for selecting the best treatment answer"""

    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Selects optimal treatment answer")
        self._model_client = model_client
        self._system_message = SystemMessage(content="""You are a pathology expert responsible for 
            selecting the most appropriate treatment answer.""")

    @message_handler
    async def handle_treatment_responses(self, message: AnalysisTask, ctx: MessageContext) -> None:
        try:
            print(f"\nProcessing answer selection for question {message.question_id}")

            selection_prompt = f"""
            choose the most appropriate response from these three responses.
            If none are appropriate, you can provide a new response.

            Question: {message.question}

            MLLM_1 Response:
            {message.mllm1_response}

            MLLM_2 Response:
            {message.mllm2_response}

            MLLM_3 Response:
            {message.mllm3_response if message.mllm3_response else message.mllm4_response}

            You MUST provide your response in this JSON format:
            {{
                "response": "selected response text or your generated response",
                "reasoning": "brief explanation of your selection or generation"
            }}
            """

            response = await self._model_client.create(
                messages=[
                    self._system_message,
                    UserMessage(content=selection_prompt, source=self.id.type)
                ],
                json_output=True
            )

            result = json.loads(response.content)

            selection_result = TreatmentAnswerSelection(
                question_id=message.question_id,
                question=message.question,
                selected_answer=result,
                original_responses={
                    "mllm1": message.mllm1_response,
                    "mllm2": message.mllm2_response,
                    "mllm3": message.mllm3_response if message.mllm3_response else message.mllm4_response
                }
            )


            await self.publish_message(
                selection_result,
                TopicId("answer_verification", source=message.question_id)
            )

        except Exception as e:
            print(f"Error in answer selection: {str(e)}")
            raise

@type_subscription(topic_type="answer_verification")
class VerificationPanelAgent(RoutedAgent):
    """Verification panel for treatment answers"""

    def __init__(self, model_clients: Dict[str, ChatCompletionClient], model_client:ChatCompletionClient):
        super().__init__("Verifies treatment answer selections")
        self._model_clients = model_clients
        self._model_client = model_client
        self._system_message = SystemMessage(content="""You are a pathology expert reviewer.
            Verify the selected treatment answer for accuracy and completeness.""")

    @message_handler
    async def handle_selection_result(self, message: TreatmentAnswerSelection, ctx: MessageContext) -> None:
        try:
            question_id = message.question_id
            max_revisions = 2
            current_answer = message.selected_answer
            revision_count = 0

            while revision_count <= max_revisions:
                print(f"\nVerification round {revision_count + 1} for question {question_id}")

                verification_prompt = f"""
                Review this Subtype answer.

                Question: {message.question}
                Selected Answer: {current_answer.get('response')}

                Reference Responses:
                MLLM_1: {message.original_responses['mllm1']}
                MLLM_2: {message.original_responses['mllm2']}
                MLLM_3: {message.original_responses['mllm3']}

                You MUST provide your review in this JSON format:
                {{
                    "verdict": "approve" or "reject",
                    "reasoning": "explanation for your decision",
                    "suggestion": "specific improvement suggestion if rejected, leave empty if approved,no more than 50 words"
                }}
                """

                # Gather reviews from all model clients
                reviews = await asyncio.gather(*[
                    self._get_review(client, verification_prompt, model_name)
                    for model_name, client in self._model_clients.items()
                ])

                approved_count = sum(1 for review in reviews if review.verdict == "approve")

                # If enough approvals or max revisions reached, finalize and exit
                if approved_count >= 2 or revision_count >= max_revisions:
                    await self._save_and_publish_result(
                        question_id=question_id,
                        final_answer=current_answer['response'],
                        reasoning=current_answer['reasoning'],
                        approved=(approved_count >= 2)
                    )
                    break

                # Otherwise, revise the answer
                revision_prompt = f"""
                Revise the Subtype answer based on reviewer feedback.
                IMPORTANT: Maintain the same writing style as the reference responses.

                Question: {message.question}
                Current Answer: {current_answer['response']}

                Reference Responses for Style:
                MLLM_1: {message.original_responses['mllm1']}
                MLLM_2: {message.original_responses['mllm2']}
                MLLM_3: {message.original_responses['mllm3']}

                Reviewer Feedback:
                {json.dumps([{
                    'reviewer': r.reviewer_id,
                    'reasoning': r.reasoning,
                    'suggestion': r.suggestion
                } for r in reviews if r.verdict == 'reject'], indent=2)}

                You MUST provide your revised answer in this JSON format:
                {{
                    "response": "revised answer matching reference style, no more than 50 words.",
                    "reasoning": "brief explanation of your revisions"
                }}
                """

                revision_response = await self._model_client.create(
                    messages=[
                        SystemMessage(
                            content="You are a pathology expert. Revise the Subtype answer while maintaining the reference style."),
                        UserMessage(content=revision_prompt, source=self.id.type)
                    ],
                    json_output=True
                )

                current_answer = json.loads(revision_response.content)
                revision_count += 1

                print(f"Completed revision {revision_count}. Approval count: {approved_count}")

        except Exception as e:
            print(f"Error in verification panel: {str(e)}")
            raise

    async def _get_review(
            self,
            model_client: ChatCompletionClient,
            prompt: str,
            reviewer_id: str
    ) -> VerificationReview:
        """Get review from individual model"""
        response = await model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=prompt, source=self.id.type)
            ],
            json_output=True
        )

        # review_content = json.loads(response.content)
        review_content = parse_model_response(response, reviewer_id)
        return VerificationReview(
            reviewer_id=reviewer_id,
            verdict=review_content.get("verdict"),
            reasoning=review_content.get("reasoning"),
            suggestion=review_content.get("suggestion", "")
        )

    async def _save_and_publish_result(
            self,
            question_id: str,
            final_answer: str,
            reasoning: str,
            approved: bool
    ):
        """Save to file and publish final result"""
        # Save to JSONL file
        data_to_save = {
            'question_id': question_id,
            'text': final_answer
        }

        file_path = Config.REFINED_RESPONSES_PATH
        # file_path = 'output/refined_responses_Prognosis_test.jsonl'
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(data_to_save, ensure_ascii=False) + "\n")

        # Create and publish final result
        final_result = VerificationPanelResult(
            content=json.dumps({
                'response': final_answer,
                'reasoning': reasoning,
                'approved': approved
            }),
            final_conclusion=final_answer
        )

        await self.publish_message(
            final_result,
            TopicId("final_result", source=question_id)
        )
