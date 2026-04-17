class ScoreCalculator:
    """Utility class for calculating various scores"""

    @staticmethod
    def calculate_semantic_score(total_relations: int, conflicts: int) -> float:
        """Calculate semantic consistency score"""
        if total_relations == 0:
            return 0.0
        return max(0.0, min(1.0, (total_relations - conflicts) / total_relations))

    @staticmethod
    def calculate_logical_score(total_relations: int, conflicts: int) -> float:
        """Calculate logical consistency score"""
        if total_relations == 0:
            return 0.0
        return max(0.0, min(1.0, (total_relations - conflicts) / total_relations))

    @staticmethod
    def calculate_consistency_score(semantic_score: float, logical_score: float) -> float:
        """Calculate overall consistency score"""
        return (semantic_score + logical_score) / 2

    @staticmethod
    def calculate_knowledge_score(matched_rules: int, total_rules: int) -> float:
        """Calculate knowledge validation score"""
        if total_rules == 0:
            return 0.0
        return max(0.0, min(1.0, matched_rules / total_rules))

    @staticmethod
    def calculate_final_score(
            knowledge_score: float,
            consistency_score: float,
            classifier_score: float = None
    ) -> float:
        """Calculate final reliability score.

        If classifier_score is None (e.g. Treatment task where classifier
        verification is skipped), only ICV and EKV/Fact scores are used,
        re-weighted to sum to 1.0: W_k=0.6, W_l=0.4.
        """
        if classifier_score is None:
            return max(0.0, min(1.0,
                0.6 * knowledge_score +
                0.4 * consistency_score
            ))
        W1 = 0.3  # knowledge score
        W2 = 0.2  # consistency score
        W3 = 0.5  # classifier verification score
        return max(0.0, min(1.0,
            W1 * knowledge_score +
            W2 * consistency_score +
            W3 * classifier_score
        ))