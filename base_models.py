from typing import List, Optional, Dict
from pydantic import BaseModel

class TaskAllocationRequest(BaseModel):
    """Request for task allocation"""
    question_id: str
    question: str

class TaskAllocation(BaseModel):
    """Task allocation result"""
    agent_type: str
    question_id: str
    question: str

class AnalysisTask(BaseModel):
    """Analysis task carrying pre-stored MLLM responses"""
    question_id: str
    question: str
    mllm1_response: str
    mllm2_response: str
    mllm3_response: Optional[str] = None
    mllm4_response: Optional[str] = None


class ModelAnalysis(BaseModel):
    """Analysis results for a single model response"""
    content_score: float
    reasoning_score: float
    consistency_score: float
    issues_found: List[str]

class ConsistencyResult(BaseModel):
    """ICV agent result containing consistency analysis for all models"""
    content: str
    mllm1_analysis: ModelAnalysis
    mllm2_analysis: ModelAnalysis
    mllm3_analysis: ModelAnalysis

class KnowledgeValidation(BaseModel):
    """Knowledge verification result for a single model"""
    score: float
    matched_rules: List[str]
    unmatched_rules: List[str]
    validation_details: str

class VerificationResult(BaseModel):
    """EKV/Fact agent result containing knowledge verification scores for all models"""
    content: str
    mllm1_validation: KnowledgeValidation
    mllm2_validation: KnowledgeValidation
    mllm3_validation: KnowledgeValidation

class ModelReliability(BaseModel):
    """Reliability assessment for a single model"""
    knowledge_score: float
    consistency_score: float
    classifier_score: float
    final_score: float
    assessment_details: str

class IntegrationResult(BaseModel):
    """Final integrated analysis result"""
    content: str
    mllm1_reliability: ModelReliability
    mllm2_reliability: ModelReliability
    mllm3_reliability: ModelReliability
    final_conclusion: str
    supporting_evidence: List[str]


class ClassifierPrediction(BaseModel):
    """Prediction from a single classifier model"""
    model_name: str
    label: str
    confidence: float

class ClassifierVerification(BaseModel):
    """Classifier verification scores for a single model"""
    multi_modal_consistency: float  # MCS
    classifier_internal_consistency: float  # CIS
    classifier_verification_score: float  # CVS
    classifier_predictions:List[ClassifierPrediction]


class ClassifierVerificationResult(BaseModel):
    """EKV/Consensus agent result containing classifier verification for all models"""
    mllm1_result: ClassifierVerification
    mllm2_result: ClassifierVerification
    mllm3_result: ClassifierVerification
    content: str  # JSON string with full analysis
    original_task: AnalysisTask


class TreatmentAnswerSelection(BaseModel):
    """Selected treatment answer"""
    question_id: str
    question: str
    selected_answer: Dict
    original_responses: Dict[str, str]

class VerificationReview(BaseModel):
    """Review from a verification agent"""
    reviewer_id: str
    verdict: str
    reasoning: str
    suggestion: str = ""

class VerificationPanelResult(BaseModel):
    content: str
    final_conclusion: str
