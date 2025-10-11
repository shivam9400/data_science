from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, GlobalMaxPooling1D


class ModelFactory:
    @staticmethod
    def create(architecture: str, vocab_size: int, max_len: int, embedding_dim: int, embedding_matrix=None):
        """Create a model by name. Supported: 'cnn_lstm', 'simple_lstm', 'cnn_globalpool'."""
        if architecture == 'cnn_lstm':
            model = Sequential()
            if embedding_matrix is not None:
                model.add(Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    input_length=max_len,
                                    weights=[embedding_matrix],
                                    trainable=False))
            else:
                model.add(Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    input_length=max_len))
            model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(64))
            model.add(Dense(1, activation='sigmoid'))
            return model

        if architecture == 'simple_lstm':
            model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
                LSTM(64),
                Dense(1, activation='sigmoid')
            ])
            return model

        if architecture == 'cnn_globalpool':
            model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
                Conv1D(filters=128, kernel_size=5, activation='relu'),
                GlobalMaxPooling1D(),
                Dense(1, activation='sigmoid')
            ])
            return model

        raise ValueError(f'Unknown architecture: {architecture}')
