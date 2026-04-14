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
            classifier_score: float
    ) -> float:
        """Calculate final score including classifier verification"""
        W1 = 0.3  # Weight for knowledge score
        W2 = 0.2  # Weight for consistency score
        W3 = 0.5  # Weight for classifier verification score
        return max(0.0, min(1.0,
            W1 * knowledge_score +
            W2 * consistency_score +
            W3 * classifier_score
        ))