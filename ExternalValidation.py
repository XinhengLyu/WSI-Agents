import json

from ScoreCalculator import *
from knowledge_base import MedicalKnowledgeBase
from autogen_core import MessageContext, RoutedAgent, TopicId, message_handler, \
    type_subscription
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from base_models import *
from ClassifierResultsReader import ClassifierResponseReader,ClassifierPrediction


@type_subscription(topic_type="classifier_verification")
class ClassifierVerificationAgent(RoutedAgent):
    """Agent for verifying consistency with classifier models"""

    def __init__(self, model_client: ChatCompletionClient, classifier_reader: ClassifierResponseReader,
                 classifier_models: List[str]):
        super().__init__("Verifies consistency with classifier models")
        self._model_client = model_client
        self._classifier_reader = classifier_reader
        self._classifier_models = classifier_models
        self._system_message = SystemMessage(content="""You are a medical diagnosis extraction expert.
            Extract the main diagnostic finding from each model's response.

            You MUST provide your output in JSON format:
            {
                "diagnosis": "the main diagnostic finding",
                "confidence": float between 0-1
            }""")

    async def _extract_diagnosis(self, model_name: str, response: str) -> Dict:
        """Extract diagnostic finding from model response"""
        prompt = f"""
        Extract the main diagnostic finding from this {model_name} analysis:

        {response}

        Focus on the primary diagnosis or main pathological condition.
        Provide confidence based on the certainty of language used.
        """
        response = await self._model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=prompt, source=self.id.type)
            ],
            json_output=True
        )

        return json.loads(response.content)

    async def compare_diagnoses(self, diag1: str, diag2: str) -> float:
        prompt = f"""Compare these two medical diagnoses:
        Diagnosis 1: {diag1}
        Diagnosis 2: {diag2}

        Rules for comparison:
        1. Score 1.0 if:
           - Exactly the same condition (including synonyms/abbreviations)
           - One is a more specific form of the other
           - One includes the other as a primary component
        2. Score 0.0 if:
           - Different conditions
           - Unrelated pathologies

        Response format: {{"similarity": 0.0 or 1.0}}"""

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            response = await self._model_client.create(
                messages=[
                    SystemMessage(
                        content="""You are a medical terminology expert specialized in comparing diagnoses.
                            You will receive two diagnoses and must determine their similarity.
                            You MUST respond with ONLY a JSON object containing a single 'similarity' field.
                            The similarity score should be:
                            - 1.0 if the diagnoses are identical or one directly includes the other
                            - 0.0 if they are completely different conditions
                            - Values between 0.0 and 1.0 are not allowed."""
                    ),
                    UserMessage(content=prompt, source=self.id.type)
                ],
                json_output=True
            )
            result = json.loads(response.content)
            if "similarity" in result:
                return result["similarity"]
            print("Failed to extract similarity. Retrying...")
            retry_count += 1
        return 0.5


    async def _calculate_verification_scores(
            self,
            diagnosis: Dict,
            classifier_predictions: List[ClassifierPrediction]
    ) -> ClassifierVerification:
        """Calculate MCS, CIS, and CVS verification scores"""
        if not classifier_predictions:
            return ClassifierVerification(
                multi_modal_consistency=0.0,
                classifier_internal_consistency=0.0,
                classifier_verification_score=0.0,
                classifier_predictions=[]
            )

        # Calculate CIS
        prediction_counts = {}
        total_classifiers = len(classifier_predictions)

        for pred in classifier_predictions:
            label = pred.label.lower()
            prediction_counts[label] = prediction_counts.get(label, 0) + 1

        max_agreed = max(prediction_counts.values())
        cis = max_agreed / total_classifiers


        mcs_values = []
        for pred in classifier_predictions:
            similarity = await self.compare_diagnoses(diagnosis["diagnosis"], pred.label)
            mcs_value = similarity * pred.confidence
            mcs_values.append(mcs_value)

        mcs = sum(mcs_values) / len(mcs_values) if mcs_values else 0.0


        cvs = max(mcs * cis, 0.01)

        return ClassifierVerification(
            multi_modal_consistency=mcs,
            classifier_internal_consistency=cis,
            classifier_verification_score=cvs,
            classifier_predictions=classifier_predictions
        )

    @message_handler
    async def handle_analysis_task(self, message: AnalysisTask, ctx: MessageContext) -> None:
        try:
            print(f"\n{'-' * 80}\nClassifierVerificationAgent Starting Analysis...")

            # Fetch classifier predictions
            classifier_predictions = self._classifier_reader.get_predictions(
                self._classifier_models,
                message.question_id
            )

            if not classifier_predictions:
                print(f"Warning: No classifier predictions found for case {message.question_id}")

            # Extract diagnosis from each MLLM response
            mllm1_diagnosis = await self._extract_diagnosis("MLLM_1", message.mllm1_response)
            mllm2_diagnosis = await self._extract_diagnosis("MLLM_2", message.mllm2_response)
            mllm3_diagnosis = await self._extract_diagnosis(
                "MLLM_3",
                message.mllm3_response if message.mllm3_response else message.mllm4_response
            )

            # Calculate verification scores
            mllm1_verification = await self._calculate_verification_scores(mllm1_diagnosis, classifier_predictions)
            mllm2_verification = await self._calculate_verification_scores(mllm2_diagnosis, classifier_predictions)
            mllm3_verification = await self._calculate_verification_scores(mllm3_diagnosis, classifier_predictions)

            result = ClassifierVerificationResult(
                mllm1_result=mllm1_verification,
                mllm2_result=mllm2_verification,
                mllm3_result=mllm3_verification,
                content=json.dumps({
                    "classifier_predictions": [
                        {
                            "model_name": pred.model_name,
                            "label": pred.label,
                            "confidence": pred.confidence
                        } for pred in classifier_predictions
                    ],
                    "verifications": {
                        "mllm1": {
                            "mcs": mllm1_verification.multi_modal_consistency,
                            "cis": mllm1_verification.classifier_internal_consistency,
                            "cvs": mllm1_verification.classifier_verification_score
                        },
                        "mllm2": {
                            "mcs": mllm2_verification.multi_modal_consistency,
                            "cis": mllm2_verification.classifier_internal_consistency,
                            "cvs": mllm2_verification.classifier_verification_score
                        },
                        "mllm3": {
                            "mcs": mllm3_verification.multi_modal_consistency,
                            "cis": mllm3_verification.classifier_internal_consistency,
                            "cvs": mllm3_verification.classifier_verification_score
                        }
                    }
                }),
                original_task=message
            )

            self._print_verification_summary(result)

            await self.publish_message(
                result,
                TopicId("integration", source=message.question_id)
            )

        except Exception as e:
            print(f"Error in ClassifierVerificationAgent: {str(e)}")
            raise

    def _print_verification_summary(self, result: ClassifierVerificationResult) -> None:
        """Print classifier verification summary"""
        print("\n=== Classifier Verification Summary ===")
        print("\nClassifier Predictions:")
        for pred in result.mllm1_result.classifier_predictions:
            print(f"- {pred.model_name}: {pred.label} (confidence: {pred.confidence:.2f})")

        for model_name, verification in [
            ("MLLM_1", result.mllm1_result),
            ("MLLM_2", result.mllm2_result),
            ("MLLM_3", result.mllm3_result)
        ]:
            print(f"\n{model_name}:")
            print(f"Multi-Modal Consistency (MCS): {verification.multi_modal_consistency:.2f}")
            print(f"Classifier Internal Consistency (CIS): {verification.classifier_internal_consistency:.2f}")
            print(f"Classifier Verification Score (CVS): {verification.classifier_verification_score:.2f}")

@type_subscription(topic_type="verification")
class KnowledgeVerificationAgent(RoutedAgent):
    """Agent responsible for verifying findings against medical knowledge base"""

    def __init__(self, model_client: ChatCompletionClient, knowledge_base: MedicalKnowledgeBase):
        super().__init__("Verifies findings against medical knowledge")
        self._model_client = model_client
        self._knowledge_base = knowledge_base
        self._calculator = ScoreCalculator()

        # System message for diagnosis extraction
        self._diagnosis_extraction_message = SystemMessage(content="""You are a medical diagnosis extraction expert.
            Extract the core diagnostic finding or key pathological condition from the given text.

            You MUST provide your output in this JSON format:
            {
                "diagnosis": "the main diagnostic finding",
                "confidence": float between 0-1,
                "evidence": "key evidence and observations supporting this diagnosis",
                "additional_findings": ["list of secondary or supporting findings"]
            }""")

        # System message for knowledge verification
        self._verification_message = SystemMessage(content="""You are a medical knowledge verification expert.
            Compare the model findings with the knowledge base reference information.

            For each model's findings, analyze:
            1. Alignment with standard diagnostic criteria
            2. Completeness of feature identification
            3. Accuracy of described manifestations

            You MUST provide your output in this JSON format:
            {
                "mllm1_validation": {
                    "matched_rules": ["list of matched diagnostic criteria"],
                    "unmatched_rules": ["list of expected but missing criteria"],
                    "incorrect_findings": ["list of findings that contradict knowledge base"],
                    "validation_details": "detailed verification explanation"
                },
                "mllm2_validation": {
                    // Same structure as mllm1_validation
                },
                "mllm3_validation": {
                    // Same structure as mllm1_validation
                }
            }""")

    async def _extract_diagnosis(self, model_name: str, findings: str) -> Dict:
        """Extract diagnostic finding from model findings"""
        try:
            extraction_prompt = f"""
            Extract the main diagnostic finding from this {model_name} analysis:

            {findings}

            Focus on:
            1. Primary diagnosis or main pathological condition
            2. Key supporting evidence
            3. Level of diagnostic confidence
            4. Any additional relevant findings

            Remember to provide your response in the specified JSON format.
            """

            response = await self._model_client.create(
                messages=[
                    self._diagnosis_extraction_message,
                    UserMessage(content=extraction_prompt, source=self.id.type)
                ],
                json_output=True,
                cancellation_token=None
            )

            result = json.loads(response.content)
            print(f"\nExtracted {model_name} diagnosis: {result['diagnosis']}")
            return result

        except Exception as e:
            print(f"Error extracting diagnosis from {model_name}: {str(e)}")
            raise

    @message_handler
    async def handle_analysis_task(self, message: AnalysisTask, ctx: MessageContext) -> None:
        try:
            print(f"\n{'-' * 80}\nKnowledgeVerificationAgent Starting Verification...")

            # Extract diagnoses from each model's findings
            mllm1_diagnosis = await self._extract_diagnosis(
                "MLLM_1",
                message.mllm1_response
            )
            mllm2_diagnosis = await self._extract_diagnosis(
                "MLLM_2",
                message.mllm2_response
            )
            mllm3_diagnosis = await self._extract_diagnosis(
                "MLLM_3",
                message.mllm3_response if message.mllm3_response else message.mllm4_response
            )

            # Query knowledge base for each diagnosis
            mllm1_kb_info = self._knowledge_base.get_diagnosis_info(mllm1_diagnosis['diagnosis'])
            mllm2_kb_info = self._knowledge_base.get_diagnosis_info(mllm2_diagnosis['diagnosis'])
            mllm3_kb_info = self._knowledge_base.get_diagnosis_info(mllm3_diagnosis['diagnosis'])

            # Create verification prompt
            verification_prompt = f"""
            Verify these diagnostic findings against medical knowledge:

            === MLLM_1 Analysis ===
            Diagnosis: {mllm1_diagnosis['diagnosis']}
            Evidence: {mllm1_diagnosis['evidence']}
            Confidence: {mllm1_diagnosis['confidence']}
            Additional Findings: {', '.join(mllm1_diagnosis['additional_findings'])}

            Knowledge Base Reference:
            {mllm1_kb_info}

            === MLLM_2 Analysis ===
            Diagnosis: {mllm2_diagnosis['diagnosis']}
            Evidence: {mllm2_diagnosis['evidence']}
            Confidence: {mllm2_diagnosis['confidence']}
            Additional Findings: {', '.join(mllm2_diagnosis['additional_findings'])}

            Knowledge Base Reference:
            {mllm2_kb_info}

            === MLLM_3 Analysis ===
            Diagnosis: {mllm3_diagnosis['diagnosis']}
            Evidence: {mllm3_diagnosis['evidence']}
            Confidence: {mllm3_diagnosis['confidence']}
            Additional Findings: {', '.join(mllm3_diagnosis['additional_findings'])}

            Knowledge Base Reference:
            {mllm3_kb_info}

            For each model:
            1. List diagnostic criteria that match with knowledge base
            2. List expected criteria that are missing
            3. Identify any findings that contradict knowledge base
            4. Provide detailed verification explanation

            Remember to provide your analysis in the specified JSON format.
            """

            # Get verification result
            response = await self._model_client.create(
                messages=[
                    self._verification_message,
                    UserMessage(content=verification_prompt, source=self.id.type)
                ],
                json_output=True,
                cancellation_token=ctx.cancellation_token
            )

            result = json.loads(response.content)
            verification_result = self._process_verification(result)

            # Print verification summary
            self._print_verification_summary(verification_result)
            session_id = message.question_id
            # Send to integration agent
            await self.publish_message(
                verification_result,
                TopicId("integration", source=session_id)
            )

        except Exception as e:
            print(f"Error in KnowledgeVerificationAgent: {str(e)}")
            raise

    def _process_verification(self, result: Dict) -> VerificationResult:
        """Process verification results into structured format"""

        def process_validation(validation: Dict) -> KnowledgeValidation:
            matched_rules = validation.get("matched_rules", [])
            unmatched_rules = validation.get("unmatched_rules", [])
            incorrect_findings = validation.get("incorrect_findings", [])

            # Calculate score considering matched, unmatched, and incorrect findings
            total_criteria = len(matched_rules) + len(unmatched_rules)
            deductions = len(incorrect_findings)

            if total_criteria == 0:
                score = 0.0
            else:
                # Score based on matched criteria minus deductions for incorrect findings
                base_score = len(matched_rules) / total_criteria
                score = max(0.0, min(1.0, base_score - (deductions * 0.1)))  # Deduct 0.1 for each incorrect finding

            return KnowledgeValidation(
                score=score,
                matched_rules=matched_rules,
                unmatched_rules=unmatched_rules,
                validation_details=validation.get("validation_details", "")
            )

        return VerificationResult(
            content=json.dumps(result),
            mllm1_validation=process_validation(result["mllm1_validation"]),
            mllm2_validation=process_validation(result["mllm2_validation"]),
            mllm3_validation=process_validation(result["mllm3_validation"])
        )

    def _print_verification_summary(self, result: VerificationResult) -> None:
        """Print verification summary"""
        print("\n=== Knowledge Verification Summary ===")

        for model_type in ['mllm1', 'mllm2', 'mllm3']:
            validation = getattr(result, f"{model_type}_validation")
            print(f"\n{model_type.upper()} Validation:")
            print(f"Knowledge Score: {validation.score:.2f}")
            print(f"Matched Criteria: {len(validation.matched_rules)}")
            print(f"Unmatched Criteria: {len(validation.unmatched_rules)}")
            print("Validation Details:", validation.validation_details)
