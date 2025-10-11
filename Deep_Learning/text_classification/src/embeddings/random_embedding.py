import numpy as np

class RandomEmbedding:
    """Create a random embedding matrix for a vocabulary."""
    def __init__(self, vocab_size, embedding_dim, seed=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seed = seed

    def build_matrix(self):
        rng = np.random.RandomState(self.seed)
        return rng.normal(size=(self.vocab_size, self.embedding_dim)).astype('float32')
