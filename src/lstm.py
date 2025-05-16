import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
import os
import random

def load_and_preprocess_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    words = text.split()
    return words

def build_vocabulary_mappings(words):
    vocabulary = sorted(list(set(words)))
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
    idx_to_word = {idx: word for idx, word in enumerate(vocabulary)}
    return vocabulary, word_to_idx, idx_to_word

def prepare_training_data(words, word_to_idx, seq_length=10, step=1):
    sequences = []
    next_words = []

    for i in range(0, len(words) - seq_length, step):
        sequences.append(words[i:i + seq_length])
        next_words.append(words[i + seq_length])

    X = np.zeros((len(sequences), seq_length, len(word_to_idx)), dtype=np.bool)
    y = np.zeros((len(sequences), len(word_to_idx)), dtype=np.bool)

    for i, seq in enumerate(sequences):
        for t, word in enumerate(seq):
            X[i, t, word_to_idx[word]] = 1
        y[i, word_to_idx[next_words[i]]] = 1

    return X, y

def build_model(vocab_size, lstm_units=128, learning_rate=0.01):
    model = Sequential([
        LSTM(lstm_units, input_shape=(max_length, vocab_size)),
        Dense(vocab_size),
        Activation('softmax')
    ])
    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def sample_word(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

def generate_text(model, seed_words, word_to_idx, idx_to_word, length=1500, diversity=0.2):
    generated_text = seed_words.copy()
    current_sequence = seed_words.copy()

    for _ in range(length):
        x = np.zeros((1, max_length, len(word_to_idx)))
        for t, word in enumerate(current_sequence):
            x[0, t, word_to_idx[word]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_word_idx = sample_word(predictions, diversity)
        next_word = idx_to_word[next_word_idx]

        generated_text.append(next_word)
        current_sequence = current_sequence[1:] + [next_word]

    return ' '.join(generated_text)

if __name__ == "__main__":
    max_length = 10
    lstm_units = 128
    learning_rate = 0.01
    epochs = 50
    batch_size = 128
    gen_length = 1500
    diversity = 0.2

    words = load_and_preprocess_text('src/input.txt')
    vocabulary, word_to_idx, idx_to_word = build_vocabulary_mappings(words)

    X, y = prepare_training_data(words, word_to_idx, max_length)

    model = build_model(len(vocabulary), lstm_units, learning_rate)
    model.fit(X, y, batch_size=batch_size, epochs=epochs)

    seed_idx = random.randint(0, len(words) - max_length - 1)
    seed_words = words[seed_idx:seed_idx + max_length]
    generated_text = generate_text(model, seed_words, word_to_idx, idx_to_word, gen_length, diversity)
    os.makedirs('../result', exist_ok=True)
    with open('../result/gen.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print(generated_text)