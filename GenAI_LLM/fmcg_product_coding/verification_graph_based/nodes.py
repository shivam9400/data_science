"""
Verification nodes for Agentless Verification Graph
"""

import random
from dataclasses import dataclass

# -----------------------------
# Common Node Output
# -----------------------------

@dataclass
class NodeResult:
    name: str
    passed: bool
    confidence: float
    reason: str


# -----------------------------
# Node 1: Brand–Category Compatibility
# -----------------------------

class BrandCategoryNode:
    def __init__(self):
        self.brand_category_map = {
            "Parle": ["Biscuits", "Snacks"],
            "Pampers": ["Baby Care"],
            "Dove": ["Personal Care"]
        }

    def run(self, product):
        brand = product["predicted"]["brand"]
        category = product["predicted"]["category"]

        allowed_categories = self.brand_category_map.get(brand, [])

        if category in allowed_categories:
            return NodeResult(
                name="BrandCategory",
                passed=True,
                confidence=0.95,
                reason=f"{brand} is commonly associated with {category}"
            )
        else:
            return NodeResult(
                name="BrandCategory",
                passed=False,
                confidence=0.95,
                reason=f"{brand} is not associated with {category}"
            )


# -----------------------------
# Node 2: Description–Category Semantic Check (Mock NLI)
# -----------------------------

class DescriptionNLINode:
    def __init__(self):
        self.category_keywords = {
            "Biscuits": ["biscuit", "cookies", "glucose"],
            "Shampoo": ["shampoo", "hair", "cleanser"],
            "Baby Care": ["baby", "infant", "diaper"]
        }

    def run(self, product):
        description = product["description"].lower()
        category = product["predicted"]["category"]

        keywords = self.category_keywords.get(category, [])
        matched = [kw for kw in keywords if kw in description]

        if matched:
            return NodeResult(
                name="DescriptionNLI",
                passed=True,
                confidence=0.85,
                reason=f"Description mentions keywords: {matched}"
            )
        else:
            return NodeResult(
                name="DescriptionNLI",
                passed=False,
                confidence=0.60,
                reason="Description does not semantically support category"
            )


# -----------------------------
# Node 3: Image–Category Compatibility (Mock CLIP)
# -----------------------------

class ImageCategoryNode:
    def run(self, product):
        category = product["predicted"]["category"]

        # Simulated CLIP probability
        if category == "Biscuits":
            clip_score = random.uniform(0.75, 0.95)
        else:
            clip_score = random.uniform(0.10, 0.40)

        passed = clip_score >= 0.7


# -----------------------------
# Node 4: Web Evidence Consistency
# -----------------------------

class WebEvidenceNode:
    def run(self, product):
        web_text = product.get("web_text", "").lower()
        category = product["predicted"]["category"].lower()

        if not web_text:
            return NodeResult(
                name="WebEvidence",
                passed=False,
                confidence=0.4,
                reason="No web evidence available"
            )

        if category in web_text:
            return NodeResult(
                name="WebEvidence",
                passed=True,
                confidence=0.80,
                reason="Category explicitly mentioned in web content"
            )
        else:
            return NodeResult(
                name="WebEvidence",
                passed=False,
                confidence=0.55,
                reason="Web content does not support category"
            )