import argparse
from pathlib import Path
import numpy as np
from tensorflow.keras.optimizers import Adam
from tokenizers.keras_tokenizer import KerasTokenizerWrapper
from models.model_factory import ModelFactory
from embeddings.random_embedding import RandomEmbedding
from utils.io import save_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.load_dataset import load_dataset

def main(args):
    # load training sentences and their labels [0, 1]
    sentences, labels = load_dataset()

    tokenizer = KerasTokenizerWrapper(num_words=args.vocab_size)
    tokenizer.fit(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    X = pad_sequences(sequences, 
                      maxlen=args.max_len, 
                      padding='post')

    embedding = None
    if args.embedding == 'random':
        emb = RandomEmbedding(vocab_size=args.vocab_size, embedding_dim=args.embedding_dim)
        embedding = emb.build_matrix()

    model = ModelFactory.create(args.architecture, 
                                vocab_size=args.vocab_size, 
                                max_len=args.max_len,
                                embedding_dim=args.embedding_dim, 
                                embedding_matrix=embedding)
    
    model.compile(optimizer=Adam(learning_rate=args.lr), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    model.fit(X, labels, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / 'model.h5')
    tokenizer.save(out / 'tokenizer.json')
    save_json({'vocab_size': args.vocab_size, 'max_len': args.max_len, 'embedding_dim': args.embedding_dim,
               'architecture': args.architecture}, out / 'config.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--embedding', type=str, default='random')
    parser.add_argument('--architecture', type=str, default='cnn_lstm')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='artifacts')
    args = parser.parse_args()
    main(args)
