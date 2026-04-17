import json
from typing import Dict
from autogen_core import MessageContext, RoutedAgent, TopicId, message_handler, \
    type_subscription
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from base_models import AnalysisTask, ModelAnalysis, ConsistencyResult
from ScoreCalculator import ScoreCalculator


@type_subscription(topic_type="consistency")
class InternalConsistencyAgent(RoutedAgent):
    """Agent responsible for evaluating internal content and reasoning consistency"""

    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Evaluates internal content and reasoning consistency")
        self._model_client = model_client
        self._calculator = ScoreCalculator()
        self._system_message = SystemMessage(content="""You are a medical consistency evaluation expert.
                    Analyze the internal consistency of each model's response independently.

                    For each response, evaluate and score two main aspects:

                    1. Content Consistency Score (CCS):
                       Evaluate content consistency based on these criteria:
                       - Descriptive coherence: The consistency and uniformity of descriptions throughout the response. 
                         Look for contradictions in feature descriptions, inconsistent terminology, or conflicting 
                         statements about the same structures.

                       - Morphological correlation: The alignment and compatibility between described morphological 
                         features. Assess whether the described cellular patterns, architectural arrangements, and 
                         tissue characteristics form a coherent and compatible set of observations.

                       - Question-answer relevance: The direct alignment between the response content and the 
                         original question. Evaluate whether all parts of the response contribute to answering 
                         the specific question asked.

                    2. Reasoning Consistency Score (RCS):
                       Evaluate reasoning consistency based on these criteria:
                       - Logical coherence: The soundness and consistency of the reasoning process. Look for 
                         clear connections between observations and interpretations, proper use of medical 
                         knowledge, and absence of logical fallacies or jumps in reasoning.

                       - Conclusion-evidence alignment: The support between stated conclusions and presented 
                         evidence. Verify that all conclusions are properly supported by specific observations 
                         or established medical knowledge.

                       - Pathology-feature correlation: The relationship between described pathological features
                         and their interpretations. Check that the pathological features described are consistent
                         with and support any diagnostic conclusions.

                    Calculate each score (0-1) based on the adherence to these criteria. A perfect score indicates
                    complete consistency and alignment across all aspects. Deduct points for each inconsistency
                    or misalignment found.

                    You MUST provide your analysis in JSON format with this structure:
                    {
                        "mllm1_analysis": {
                            "content_score": float (0-1),
                            "reasoning_score": float (0-1),
                            "issues_found": ["list of main issues"]
                        },
                        "mllm2_analysis": {
                            // Same structure as mllm1_analysis
                        },
                        "mllm3_analysis": {
                            // Same structure as mllm1_analysis
                        }
                    }""")

    @message_handler
    async def handle_analysis_task(self, message: AnalysisTask, ctx: MessageContext) -> None:
        try:
            print(f"\n{'-' * 80}\nInternalConsistencyAgent Starting Analysis...")

            analysis_prompt = f"""
            Analyze each model's response independently and calculate two consistency scores.

            ANALYSIS CONTEXT:
            Question: {message.question}

            RESPONSES TO ANALYZE:
            === MLLM_1 Response ===
            {message.mllm1_response}

            === MLLM_2 Response ===
            {message.mllm2_response}

            === MLLM_3 Response ===
            {message.mllm3_response}

            EVALUATION CRITERIA:

            Content Consistency Score (CCS):
            Evaluate each response's internal content consistency based on:
            - Uniformity and coherence of descriptions
            - Compatibility between morphological features
            - Relevance and completeness in addressing the question

            Reasoning Consistency Score (RCS):
            - Logical connections between observations and conclusions
            - Evidence supporting stated conclusions
            - Correlation between pathological features and diagnoses

            For each response, provide:
            1. A CCS score (0-1)
            2. An RCS score (0-1)
            3. List of significant issues found"""

            # Get model response
            response = await self._model_client.create(
                messages=[
                    self._system_message,
                    UserMessage(content=analysis_prompt, source=self.id.type)
                ],
                json_output=True,
                cancellation_token=ctx.cancellation_token
            )

            result = json.loads(response.content)
            consistency_result = self._process_consistency_analysis(result)
            self._print_analysis_summary(consistency_result)

            await self.publish_message(
                consistency_result,
                TopicId("integration", source=message.question_id)
            )

        except Exception as e:
            print(f"Error in InternalConsistencyAgent: {str(e)}")
            raise

    def _process_consistency_analysis(self, result: Dict) -> ConsistencyResult:
        """Process analysis results into structured consistency result"""

        def process_model_analysis(analysis: Dict) -> ModelAnalysis:
            """Convert model analysis dictionary into a ModelAnalysis object"""

            # Ensure the input has required keys
            required_keys = {"content_score", "reasoning_score", "issues_found"}
            if not all(key in analysis for key in required_keys):
                raise ValueError(f"Missing required fields in analysis data: {analysis}")

            return ModelAnalysis(
                content_score=float(analysis["content_score"]),
                reasoning_score=float(analysis["reasoning_score"]),
                consistency_score=(float(analysis["content_score"]) + float(analysis["reasoning_score"])) / 2,
                issues_found=analysis["issues_found"] if isinstance(analysis["issues_found"], list) else []
            )

        return ConsistencyResult(
            content=json.dumps(result),
            mllm1_analysis=process_model_analysis(result["mllm1_analysis"]),
            mllm2_analysis=process_model_analysis(result["mllm2_analysis"]),
            mllm3_analysis=process_model_analysis(result["mllm3_analysis"])
        )

    def _print_analysis_summary(self, result: ConsistencyResult) -> None:
        """Print analysis summary"""
        print("\n=== Internal Consistency Analysis Summary ===")

        for model in ['mllm1', 'mllm2', 'mllm3']:
            analysis = getattr(result, f"{model}_analysis")
            print(f"\n{model.upper()} Analysis:")
            print(f"Content Consistency Score (CCS): {analysis.content_score:.2f}")
            print(f"Reasoning Consistency Score (RCS): {analysis.reasoning_score:.2f}")
            print(f"Overall Consistency Score: {analysis.consistency_score:.2f}")

            if analysis.issues_found:
                print("\nIdentified Issues:")
                for issue in analysis.issues_found:
                    print(f"- {issue}")