from keras.layers import Dense, Embedding, LSTM, TimeDistributed
from keras.models import Sequential, load_model
import h5py
import numpy as np
import pandas as pd
import string
import sys

SEQ_LENGTH = 50

def process_twitter(path):
    data = pd.read_csv(path, delimiter='|',
            names=['handle', 'timestamp', 'date', 'text'])
    return data.text.tolist()

def map_index_to_char():
    index_to_char = [chr(i) for i in range(128)]
    return index_to_char

def map_char_to_index():
    index_to_char = map_index_to_char()
    char_to_index = {token: i for i, token in enumerate(index_to_char)}
    return char_to_index

def process_sentences_as_char(data):
    num_vocab = 128 # ASCII characters
    printable = set(string.printable)
    filtered_data = map(lambda x: filter(lambda y: y in printable, x), data)
    return filtered_data

def lstm_model(input_shape):
    model = Sequential()
    model.add(Dense(200, activation='tanh',
                    input_shape=input_shape))
    model.add(LSTM(200, return_sequences=True))
                   # input_shape=input_shape))
    model.add(LSTM(200, return_sequences=True))
    model.add(TimeDistributed(Dense(128, activation='softmax')))
    return model

def preprocess_batch(data_batch):
    X, Y = [], []
    char_to_index = map_char_to_index()
    for data_point in data_batch:
        x, y = np.zeros((SEQ_LENGTH, 128)), np.zeros((SEQ_LENGTH, 128))
        start = np.random.randint(0, len(data_point) - SEQ_LENGTH)
        x_str = data_point[start:start + SEQ_LENGTH]
        y_str = data_point[start + 1:start + SEQ_LENGTH + 1]
        x[np.arange(SEQ_LENGTH), map(lambda c: char_to_index[c], x_str)] = 1
        y[np.arange(SEQ_LENGTH), map(lambda c: char_to_index[c], y_str)] = 1
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

def train_rnn_model(model, model_path, data, num_iters, batch_size):
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Train by getting the model to regress on next character.
    for i in range(num_iters):
        idx = np.random.choice(np.arange(len(data)), batch_size, replace=False)
        data_batch = [data[j] for j in idx]
        X, Y = preprocess_batch(data_batch)
        loss = model.train_on_batch(X, Y) 
        if i % 10 == 0:
            print(i, loss)
    model.save(model_path)

# "Greedy" sampling at each step.
def generate_strings(model_path, num_strings, str_len):
    model = load_model(model_path)
    index_to_char = map_index_to_char()
    generated_strings = []
    for i in range(num_strings):
        vals = [np.zeros(128)]
        vals[0][np.random.choice(np.arange(128))] = 1
        for j in range(str_len - 1):
            input_seq = np.array(vals)
            input_seq = input_seq.reshape(1, len(vals), 128)
            probs = model.predict(input_seq)[0]
            next_idx = np.random.choice(np.arange(128),
                                        p=probs[-1].flatten())
            vals.append(np.zeros(128))
            vals[-1][next_idx] = 1
            print(len(vals), str_len)
        decoded_string = map(lambda x: index_to_char[np.argmax(x)], vals)
        generated_strings.append(''.join(decoded_string))
        # print(generated_strings[-1])
    return generated_strings

def generate_strings_beam(model_path, num_strings, str_len, beam_width=30):
    model = load_model(model_path)
    index_to_char = map_index_to_char()
    generated_strings = []
    for i in range(num_strings):
        vals = [[np.zeros(128)] for _ in range(beam_width)]
        # We'd rather add logs rather than multiply small floats.
        state_probs = [np.log(1 / 128.)] * beam_width
        for k in range(beam_width):
            vals[k][0][np.random.choice(np.arange(128))] = 1
        for j in range(str_len - 1):
            next_step_probs = np.zeros((beam_width, 128))
            # Expand each node.
            for k in range(beam_width):
                input_seq = np.array(vals[k]).reshape(1, len(vals[k]), 128)
                probs = model.predict(input_seq)[0][-1]
                next_step_probs[k] = np.log(probs)
            combined_probs = (next_step_probs.T + state_probs).T
            # Get indices corresponding to top-beam_width probs.
            top_k = np.argpartition(-combined_probs.flatten(),
                                    beam_width)[:beam_width]
            top_k_probs = combined_probs.flatten()[top_k]
            ys, xs = np.where(np.isin(combined_probs, top_k_probs))
            ys, xs = ys[:beam_width], xs[:beam_width]
            new_vals, new_state_probs = [], []
            for state_idx, char_idx in zip(ys, xs):
                next_char = np.zeros(128)
                next_char[char_idx] = 1
                new_vals.append(vals[state_idx] + [next_char])
                new_state_probs.append(combined_probs[state_idx, char_idx])
            # Update vals and state_probs.
            vals = new_vals
            state_probs = new_state_probs
            print(len(vals[0]), str_len)
            # for k in range(beam_width):
            #     decoded_string = map(lambda x: index_to_char[np.argmax(x)], vals[k])
            #     print(''.join(decoded_string))
        for k in range(beam_width):
            decoded_string = map(lambda x: index_to_char[np.argmax(x)], vals[k])
            generated_strings.append(''.join(decoded_string))
    return generated_strings


if __name__ == '__main__':
    csv_path = "realDonaldTrump_tweets.csv"
    dataset_path = "tweet_data.h5"
    model_path = "trained_lstm.h5"
    data = process_twitter(csv_path)
    data = process_sentences_as_char(data)
    data = filter(lambda x: len(x) > SEQ_LENGTH, data)
    if False:
        input_shape = (None, 128)
        model = lstm_model(input_shape)
        print(model.predict(preprocess_batch(data[:5])[0]).shape)
        train_rnn_model(model, model_path, data, 10000, 32)
    # generated_strings = generate_strings(model_path, 5, 300)
    generated_strings = generate_strings_beam(model_path, 1, 300)
    for sample_str in generated_strings:
        print(sample_str)
