import numpy as np

class VectorMetrics:
    """
    A class to demonstrate and calculate key vector similarity metrics.
    """
    def __init__(self, vector_a: np.ndarray, vector_b: np.ndarray, vector_c: np.ndarray):
        """Initializes with three example vectors."""
        self.A = vector_a
        self.B = vector_b
        self.C = vector_c
        print("--- Input Vectors ---")
        print(f"Vector A (Query): {self.A} | Magnitude: {self.get_magnitude(self.A):.2f}")
        print(f"Vector B (Similar): {self.B} | Magnitude: {self.get_magnitude(self.B):.2f}")
        print(f"Vector C (Long/Rich): {self.C} | Magnitude: {self.get_magnitude(self.C):.2f}\n")

    @staticmethod
    def get_magnitude(vec: np.ndarray) -> float:
        """Calculates the Euclidean Norm (Magnitude) of a vector."""
        return np.linalg.norm(vec)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the Cosine Similarity (Direction only).
        Range: -1 (opposite) to 1 (identical direction).
        """
        dot_product = np.dot(vec1, vec2)
        magnitude_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        # Avoid division by zero if magnitude is zero
        if magnitude_product == 0:
            return 0.0
        return dot_product / magnitude_product

    def dot_product(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the Dot Product (Direction and Magnitude).
        No fixed range. Higher is more similar.
        """
        return np.dot(vec1, vec2)

    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the Euclidean Distance (Straight-line distance).
        Range: 0 (identical points) to infinity. Lower is more similar.
        """
        # np.linalg.norm(vec1 - vec2) calculates the magnitude of the difference vector
        return np.linalg.norm(vec1 - vec2)

# --- Example Vectors ---
# Note: C is simply 5 times A, meaning it points in the exact same direction but is 5x longer.
V_A = np.array([1, 2, 3])          # Short Query Vector (Magnitude: 3.74)
V_B = np.array([1, 2, 3])          # Identical to A (Magnitude: 3.74)
V_C = np.array([5, 10, 15])        # Same direction as A, 5x the magnitude (Magnitude: 18.71)
V_D = np.array([-1, 3, -1])        # Different/Orthogonal direction (Magnitude: 3.32)

# Instantiate the class
metrics = VectorMetrics(V_A, V_B, V_C)

print("--- 1. Comparison of A (Query) and B (Identical) ---")
print(f"Cosine Similarity (A, B): {metrics.cosine_similarity(V_A, V_B):.4f} -> Perfect direction match.")
print(f"Dot Product (A, B):       {metrics.dot_product(V_A, V_B):.4f} -> High, influenced by medium magnitude.")
print(f"Euclidean Distance (A, B):{metrics.euclidean_distance(V_A, V_B):.4f} -> Zero distance (perfect overlap).\n")

print("--- 2. Comparison of A (Query) and C (Long Document, Same Topic) ---")
print(f"Cosine Similarity (A, C): {metrics.cosine_similarity(V_A, V_C):.4f} -> **Perfect 1.0** (Direction match, length ignored).")
print(f"Dot Product (A, C):       {metrics.dot_product(V_A, V_C):.4f} -> **Very High** (Direction *and* large magnitude).")
print(f"Euclidean Distance (A, C):{metrics.euclidean_distance(V_A, V_C):.4f} -> **Very High Distance** (High number because the points are far apart).\n")

print("--- 3. Comparison of A (Query) and D (Different Topic) ---")
print(f"Cosine Similarity (A, D): {metrics.cosine_similarity(V_A, V_D):.4f} -> Low or negative direction match.")
print(f"Dot Product (A, D):       {metrics.dot_product(V_A, V_D):.4f} -> Low, indicating low similarity.")
print(f"Euclidean Distance (A, D):{metrics.euclidean_distance(V_A, V_D):.4f} -> Medium distance.\n")