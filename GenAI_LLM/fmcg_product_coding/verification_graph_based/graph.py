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
        support_score = 0.0

        for node in self.nodes:
            result = node.run(product)
            node_results.append(result)

            weight = self.weights.get(result.name, 0)
            contr
