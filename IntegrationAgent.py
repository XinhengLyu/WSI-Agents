import asyncio
import json
from typing import Dict, Tuple
from agent import parse_model_response
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from ScoreCalculator import ScoreCalculator
from base_models import VerificationResult, ConsistencyResult, ClassifierVerificationResult, ModelReliability, \
    AnalysisTask, IntegrationResult
from autogen_core import MessageContext, RoutedAgent, TopicId, message_handler, type_subscription
from config import Config


@type_subscription(topic_type="integration")
class IntegrationAgent(RoutedAgent):
    """Agent responsible for merging and integrating analysis results"""

    def __init__(self, model_clients: Dict[str, ChatCompletionClient]):
        super().__init__("Integrates and merges model outputs")

        self._model_clients = model_clients
        self._model_client = model_clients["gpt-4o"]
        self._calculator = ScoreCalculator()
        self._pending_results = {}  # Store pending results by session

        self._system_message = SystemMessage(content="""You are a medical content integration expert.
            Your task is to refine the highest-scored model output based on consistency and knowledge verification results.

            Process:
            1. Examine the original response with the highest combined reliability score
            2. Review consistency analysis findings for any semantic or logical conflicts
            3. Check knowledge verification results for any missing or incorrect features
            4. Create a refined version that:
               - Maintains the original structure and format
               - Removes any incorrect statements
               - Modifies conflicting content based on verification
               - Adds verified missing information
               - Ensures all statements are consistent

            You MUST provide your output in this JSON format:
            {
                "merged_analysis": {
                    "original_response": "the complete original response from best-scoring model",
                    "refined_response": "the refined and corrected response"
                }
            }

            Guidelines for refinement:
            - Keep the same writing style and format as the original
            - Only make changes supported by consistency and verification findings
            - Ensure all statements are internally consistent
            - Remove any contradictory content
            - Add missing verified features seamlessly
            - Exclude diagnostic procedures, clinical symptoms, and medical workup information
            	""")

    def _select_best_response(
            self,
            model_reliabilities: Dict[str, ModelReliability]
    ) -> Tuple[ModelReliability, str, float]:
        """Select the response with highest combined reliability score"""

        best_model = max(model_reliabilities.items(), key=lambda x: x[1].final_score)
        return (
            best_model[1],  # ModelReliability object
            best_model[0],  # model name (mllm1/mllm2/mllm3)
            best_model[1].final_score  # final score
        )

    def _create_integration_prompt(
            self,
            verification_result: VerificationResult,
            consistency_result: ConsistencyResult,
            classifier_result: ClassifierVerificationResult,
            best_model: str,
            original_response: str,
    ) -> str:
        """Create prompt for refining the best response based on analysis findings"""

        # Retrieve analysis data for the best-scoring model
        verification = getattr(verification_result, f"{best_model}_validation")
        consistency  = getattr(consistency_result,  f"{best_model}_analysis")

        # Build analysis issues section
        analysis_issues = f"""
        - Identified Issues: {', '.join(consistency.issues_found)}
        - Missing Features (from knowledge verification): {', '.join(verification.unmatched_rules)}
        """

        if classifier_result is not None:
            classifier_verification = getattr(classifier_result, f"{best_model}_result")
            high_confidence_predictions = [
                pred for pred in classifier_verification.classifier_predictions
                if pred.confidence > 0.7
            ]
            classifier_summary = "\n".join([
                f"- {pred.model_name}: {pred.label} (confidence: {pred.confidence:.2f})"
                for pred in high_confidence_predictions
            ]) if high_confidence_predictions else "No high-confidence predictions available."
            analysis_issues += f"""
        - Multi-modal Consistency Score: {classifier_verification.multi_modal_consistency:.2f}
        - Classifier Internal Consistency: {classifier_verification.classifier_internal_consistency:.2f}
            """
            classifier_section = f"""
        Classifier Model Results (High Confidence Predictions):
        {classifier_summary}
            """
            classifier_guidelines = (
                "1. Ensure the diagnostic conclusions align with the high-confidence classifier predictions.\n"
                "        2. If there's a conflict between the original diagnosis and classifier predictions, prioritize classifier results.\n"
                "        3. Keep the morphological descriptions that support the classifier-aligned diagnosis.\n"
                "        4. Fix any other identified issues while maintaining the same writing style.\n"
                "        5. Do not add any analysis process phrases, confidence statements, or classifier references\n"
                "        6. Exclude diagnostic procedures, clinical symptoms, and medical workup information"
            )
        else:
            classifier_section = ""
            classifier_guidelines = (
                "1. Fix identified consistency and knowledge issues while maintaining the same writing style.\n"
                "        2. Do not add any analysis process phrases or confidence statements.\n"
                "        3. Exclude diagnostic procedures, clinical symptoms, and medical workup information."
            )

        prompt = f"""
        Original response to modify:

        {original_response}
        {classifier_section}
        Analysis found these issues to address:
        {analysis_issues}

        Please modify the original response to fix these issues. Keep the same writing style and any correct content unchanged.

        Please modify the original response following these guidelines:
        {classifier_guidelines}

        Provide your response in this JSON format:
        {{
            "merged_analysis": {{
                "original_response": "the original response",
                "refined_response": "the modified response"
            }}
        }}
        """

        return prompt

    def _create_enhancement_prompt(
            self,
            refined_response: str,
            model_responses: dict,
            best_model: str
    ) -> str:
        """Create prompt for enhancing with non-conflicting information from other models"""

        # Get responses from other models
        other_responses = {
            model: response for model, response in model_responses.items()
            if model != best_model and response
        }

        if not other_responses:
            return ""

        prompt = f"""
        Current refined response:
        {refined_response}

        Consider these additional model responses:
        {chr(10).join(f'=== {model.upper()} Response ==={response}' for model, response in other_responses.items())}

        Please enhance the current response by incorporating any non-conflicting observations or details from the other responses.
        Only add information that:
        1. Does not contradict the current response
        2. Provides additional valuable details(Exclude diagnostic procedures, clinical symptoms, and medical workup information)
        3. Maintains diagnostic consistency

        Provide your response in this JSON format:
        {{
            "merged_analysis": {{
                "original_response": "the current refined response",
                "refined_response": "the enhanced response with additional non-conflicting details"
            }}
        }}
        """

        return prompt

    async def _try_process_results(self, session_id: str) -> None:
        """Process results and generate enhanced integration"""
        results = self._pending_results.get(session_id, {})

        has_classifier = 'classifier' in results
        required = {'consistency', 'verification', 'analysis_task'}
        if not required.issubset(results.keys()):
            print(f"Waiting for: {required - results.keys()}")
            return
        # Classifier is optional: tasks that skip it (e.g. Treatment) proceed without it
        task_name = Config._task_name
        needs_classifier = task_name != "Treatment"
        if needs_classifier and not has_classifier:
            print(f"Waiting for classifier result for session {session_id}")
            return

        try:
            print(f"\nProcessing complete set of results for session {session_id}")

            consistency_result  = results['consistency']
            verification_result = results['verification']
            classifier_result   = results.get('classifier')   # None for Treatment
            analysis_task       = results['analysis_task']

            def _classifier_score(model_result):
                return model_result.classifier_verification_score if classifier_result else None

            # Compute reliability scores for each model
            model_reliabilities = {
                "mllm1": ModelReliability(
                    knowledge_score=verification_result.mllm1_validation.score,
                    consistency_score=consistency_result.mllm1_analysis.consistency_score,
                    classifier_score=_classifier_score(classifier_result.mllm1_result) if classifier_result else 0.0,
                    final_score=self._calculator.calculate_final_score(
                        verification_result.mllm1_validation.score,
                        consistency_result.mllm1_analysis.consistency_score,
                        _classifier_score(classifier_result.mllm1_result) if classifier_result else None,
                    ),
                    assessment_details="MLLM_1 analysis reliability assessment"
                ),
                "mllm2": ModelReliability(
                    knowledge_score=verification_result.mllm2_validation.score,
                    consistency_score=consistency_result.mllm2_analysis.consistency_score,
                    classifier_score=_classifier_score(classifier_result.mllm2_result) if classifier_result else 0.0,
                    final_score=self._calculator.calculate_final_score(
                        verification_result.mllm2_validation.score,
                        consistency_result.mllm2_analysis.consistency_score,
                        _classifier_score(classifier_result.mllm2_result) if classifier_result else None,
                    ),
                    assessment_details="MLLM_2 analysis reliability assessment"
                ),
                "mllm3": ModelReliability(
                    knowledge_score=verification_result.mllm3_validation.score,
                    consistency_score=consistency_result.mllm3_analysis.consistency_score,
                    classifier_score=_classifier_score(classifier_result.mllm3_result) if classifier_result else 0.0,
                    final_score=self._calculator.calculate_final_score(
                        verification_result.mllm3_validation.score,
                        consistency_result.mllm3_analysis.consistency_score,
                        _classifier_score(classifier_result.mllm3_result) if classifier_result else None,
                    ),
                    assessment_details="MLLM_3 analysis reliability assessment"
                ),
            }

            # Select best-scoring model
            best_model_reliability, best_model, best_score = self._select_best_response(model_reliabilities)
            original_response = getattr(analysis_task, f"{best_model}_response")

            print(f"\nSelected {best_model.upper()} response as base (score: {best_score:.2f})")
            integration_prompt = self._create_integration_prompt(
                verification_result,
                consistency_result,
                classifier_result,   # may be None for Treatment
                best_model,
                original_response,
            )

            print("\nGenerating refined analysis...")
            response = await self._model_client.create(
                messages=[
                    self._system_message,
                    UserMessage(content=integration_prompt, source=self.id.type)
                ],
                json_output=True
            )
            result = json.loads(response.content)
            refined_response = result['merged_analysis']['refined_response']

            # Enhance with non-conflicting details from other models
            print("\nEnhancing refined analysis with additional non-conflicting details...")
            enhanced_response = await self._enhance_with_verification(
                refined_response,
                {
                    'mllm1': analysis_task.mllm1_response,
                    'mllm2': analysis_task.mllm2_response,
                    'mllm3': analysis_task.mllm3_response
                },
                best_model,
                self._model_clients
            )



            data_to_save = {
                'question_id': session_id,
                "text": enhanced_response
            }

            with open(Config.REFINED_RESPONSES_PATH, 'a', encoding='utf-8') as file:
                file.write(json.dumps(data_to_save, ensure_ascii=False) + "\n")

            final_result = IntegrationResult(
                content=json.dumps({
                    'original_response': refined_response,
                    'refined_response': enhanced_response
                }),
                mllm1_reliability=model_reliabilities["mllm1"],
                mllm2_reliability=model_reliabilities["mllm2"],
                mllm3_reliability=model_reliabilities["mllm3"],
                final_conclusion=enhanced_response,
                supporting_evidence=[]
            )

            print(f"\nPublishing final result for session {session_id}")
            await self.publish_message(
                final_result,
                TopicId("final_result", source=session_id)
            )

            del self._pending_results[session_id]
            print(f"Completed processing for session {session_id}")
        except Exception as e:
            print(f"Error processing results for session {session_id}: {str(e)}")
            raise

    @message_handler
    async def handle_consistency_result(self, message: ConsistencyResult, ctx: MessageContext) -> None:
        """Handle incoming consistency analysis result"""
        session_id = ctx.topic_id.source
        print(f"\nReceived consistency result for session: {session_id}")

        if session_id not in self._pending_results:
            self._pending_results[session_id] = {}

        self._pending_results[session_id]['consistency'] = message
        print(f"Stored consistency result for session {session_id}")
        print(f"Pending results for session {session_id}: {list(self._pending_results[session_id].keys())}")

        await self._try_process_results(session_id)

    @message_handler
    async def handle_verification_result(self, message: VerificationResult, ctx: MessageContext) -> None:
        """Handle incoming verification result"""
        session_id = ctx.topic_id.source
        print(f"\nReceived verification result for session: {session_id}")

        if session_id not in self._pending_results:
            self._pending_results[session_id] = {}

        self._pending_results[session_id]['verification'] = message

        print(f"Stored verification result for session {session_id}")
        print(f"Pending results for session {session_id}: {list(self._pending_results[session_id].keys())}")

        await self._try_process_results(session_id)

    @message_handler
    async def handle_analysis_task(self, message: AnalysisTask, ctx: MessageContext) -> None:
        """Handle incoming analysis task"""
        session_id = ctx.topic_id.source
        print(f"\nReceived analysis task for session: {session_id}")

        if session_id not in self._pending_results:
            self._pending_results[session_id] = {}

        self._pending_results[session_id]['analysis_task'] = message
        print(f"Stored analysis task for session {session_id}")
        print(f"Pending results for session {session_id}: {list(self._pending_results[session_id].keys())}")

        await self._try_process_results(session_id)

    @message_handler
    async def handle_classifier_verification(self, message: ClassifierVerificationResult, ctx: MessageContext) -> None:
        """Handle incoming classifier verification result"""
        session_id = ctx.topic_id.source
        print(f"\nReceived classifier verification result for session: {session_id}")

        if session_id not in self._pending_results:
            self._pending_results[session_id] = {}

        self._pending_results[session_id]['classifier'] = message
        print(f"Stored classifier verification result for session {session_id}")
        print(f"Pending results for session {session_id}: {list(self._pending_results[session_id].keys())}")

        await self._try_process_results(session_id)

    async def _get_enhancement_review(
            self,
            model_client: ChatCompletionClient,
            prompt: str,
            reviewer_id: str
    ) -> Dict:
        """Get review for enhancement from individual model"""
        review_prompt = f"""
        Review this enhanced response.
        {prompt}

        IMPORTANT REQUIREMENTS:
        - Your reasoning must be between 0-200 words

        You MUST provide your review in this JSON format:
        {{
            "verdict": "approve" or "reject",
            "reasoning": "explanation for your decision",
            "suggestion": "specific improvement suggestion if rejected"
        }}
        """

        response = await model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=review_prompt, source=self.id.type)
            ],
            json_output=True
        )

        return parse_model_response(response, reviewer_id)

    async def _enhance_with_verification(
            self,
            refined_response: str,
            model_responses: dict,
            best_model: str,
            model_clients: Dict[str, ChatCompletionClient],
            max_revisions: int = 3
    ) -> str:
        """Enhance response with verification from multiple models"""
        current_response = refined_response
        revision_count = 0

        while revision_count < max_revisions:
            # Create enhancement prompt
            enhancement_prompt = self._create_enhancement_prompt(
                current_response,
                model_responses,
                best_model
            )

            if not enhancement_prompt:  # No other responses to enhance with
                return current_response

            # Get initial enhancement
            enhanced_response_result = await model_clients["gpt-4o"].create(
                messages=[
                    self._system_message,
                    UserMessage(content=enhancement_prompt, source=self.id.type)
                ],
                json_output=True
            )
            enhanced_result = json.loads(enhanced_response_result.content)
            enhanced_response = enhanced_result['merged_analysis']['refined_response']

            # Get reviews from other models
            reviews = await asyncio.gather(*[
                self._get_enhancement_review(
                    client,
                    enhancement_prompt + f"\nProposed Enhancement:\n{enhanced_response}",
                    reviewer_id
                )
                for reviewer_id, client in list(model_clients.items())[1:]  # Skip primary model
            ])

            # Count approvals
            approved_count = sum(1 for review in reviews if review["verdict"] == "approve")

            if approved_count >= 3 or revision_count >= max_revisions - 1:
                return enhanced_response

            # Need revision - collect feedback
            revision_prompt = f"""
            Revise the enhanced response based on reviewer feedback.
            Current Response: {enhanced_response}

            Reviewer Feedback:
            {json.dumps([{
                'reviewer': idx,
                'reasoning': review['reasoning'],
                'suggestion': review['suggestion']
            } for idx, review in enumerate(reviews) if review['verdict'] == 'reject'], indent=2)}

            Original Response Style Reference:
            {current_response}

            You MUST provide your revised response in this JSON format:
            {{
                "merged_analysis": {{
                    "original_response": "the current response",
                    "refined_response": "the revised response maintaining original style"
                }}
            }}
            """

            revision_response = await model_clients["gpt-4o"].create(
                messages=[
                    self._system_message,
                    UserMessage(content=revision_prompt, source=self.id.type)
                ],
                json_output=True
            )

            revised_result = json.loads(revision_response.content)
            current_response = revised_result['merged_analysis']['refined_response']
            revision_count += 1

        return current_response