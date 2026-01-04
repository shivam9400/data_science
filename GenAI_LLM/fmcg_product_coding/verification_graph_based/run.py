"""
Run the Agentless Verification Graph end-to-end
"""

from data import get_sample_product
from graph import VerificationGraph


def main():
    product = get_sample_product()
    graph = VerificationGraph()

    result = graph.run(product)

    print("\nFINAL VERIFICATION RESULT")
    print("=" * 30)
    print(f"Support Score : {result['support_score']}")
    print(f"Decision      : {result['decision']}\n")