from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
import h5py
import numpy as np
import pandas as pd
import string
import sys

# Set to a bit greater than 280 due to rendering of some special characters.
max_data_length = 350
sentence_start_token = "S_START"
sentence_end_token = "S_END"

def process_twitter(path):
    data = pd.read_csv(path, delimiter='|',
            names=['handle', 'timestamp', 'date', 'text'])
    return data.text.tolist()

def process_sentences_as_char(data):
    num_vocab = 128 + 2 # ASCII characters and SENTENCE_START/END
    printable = set(string.printable)
    index_to_word = [chr(i) for i in range(128)] + \
                    [sentence_start_token, sentence_end_token]
    word_to_index = {token: i for i, token in enumerate(index_to_word)}
    filtered_data = map(lambda x: filter(lambda y: y in printable, x), data)
    return index_to_word, word_to_index, filtered_data

def create_structured_data(path, dataset_path):
    data = process_twitter(csv_path)
    index_to_char, char_to_index, data = process_sentences_as_char(data)
    hdf5_file = h5py.File(dataset_path, mode='w')
    hdf5_file.create_dataset(
        'data', (len(data), max_data_length + 2, 128 + 2), 'u1')
    start_index = char_to_index[sentence_start_token]
    end_index = char_to_index[sentence_end_token]
    for i, string in enumerate(data):
        data_point = np.zeros((max_data_length + 2, 128 + 2), dtype=np.uint8)
        data_point[0, start_index] = 1
        char_indices = map(lambda x: char_to_index[x], string)
        data_point[np.arange(1, len(string) + 1), char_indices] = 1
        # Pad the end of the data with more sentence end tokens.
        data_point[len(string) + 1:, end_index] = 1
        # Add to data file.
        hdf5_file['data'][i] = data_point
    # Test that tweets look reasonable.
    # for i in range(len(data)):
    #     data_point = hdf5_file['data'][i]
    #     vals = [index_to_char[np.argmax(v)] for v in data_point]
    #     print(''.join(vals))

def lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(200, return_sequences=True,
                   input_shape=input_shape))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dense(128 + 2, activation='softmax'))
    return model

def train_rnn_model(model, data, num_iters, batch_size):
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Train by getting the model to regress on next character.
    for i in range(num_iters):
        idx = np.random.choice(np.arange(len(data)), batch_size, replace=False)
        data_batch = data[sorted(idx)]
        loss = model.train_on_batch(data_batch, data_batch) 
        if i % 10 == 0:
            print(i, loss)

if __name__ == '__main__':
    csv_path = "realDonaldTrump_tweets.csv"
    dataset_path = "tweet_data.h5"
    if False:
        create_structured_data(csv_path, dataset_path)
    data = h5py.File(dataset_path, mode='r')['data']
    model = lstm_model(data.shape[1:])
    # print(model.predict(data[:2]).shape)
    train_rnn_model(model, data, 1000, 32)
