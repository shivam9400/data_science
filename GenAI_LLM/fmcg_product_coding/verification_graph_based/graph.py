"""
Verification graph orchestration and aggregation
"""

from nodes import (
    BrandCategoryNode,
    DescriptionNLINode,
    ImageCategoryNode,
    WebEvidenceNode
)


class VerificationGraph:
    def __init__(self):
        self.nodes = [
            BrandCategoryNode(),
            DescriptionNLINode(),
            ImageCategoryNode(),
            WebEvidenceNode()
        ]

        # Importance weights (policy-driven, not learned)
        self.weights = {
            "BrandCategory": 0.30,
            "DescriptionNLI": 0.25,
            "ImageCategory": 0.25,
            "WebEvidence": 0.20
        }

    def run(self, product):
        node_results = []
        #support_score = 0.0
        weighted_sum = 0.0
        total_weight = 0.0

        for node in self.nodes:
            result = node.run(product)
            if result is None:
                print(f"[WARN] {node.__class__.__name__} returned None. Skipping.")
                continue

            node_results.append(result)

            weight = self.weights.get(result.name, 0)
            total_weight += weight
            if result.passed:
                weighted_sum += weight * result.confidence

            #contribution = weight * result.confidence * (1 if result.passed else 0)
            #support_score += contribution
        
        support_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        decision = self._decision_rule(support_score)

        return {
            "support_score": round(support_score, 3),
            "decision": decision,
            "node_results": node_results
        }

    @staticmethod
    def _decision_rule(score):
        if score >= 0.75:
            return "STRONGLY_SUPPORTED"
        elif score >= 0.50:
            return "WEAKLY_SUPPORTED"
        else:
            return "UNSUPPORTED"