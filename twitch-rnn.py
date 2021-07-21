import tensorflow as tf
from tensorflow._api.v2 import random
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time
import random

import psycopg2
from psycopg2 import sql

CHANNEL = 'rlly'
VOCAB = [chr(0), *(chr(x) for x in range(32, 127))]
MAX_MESSAGE_LENGTH = 500
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 2048
EPOCHS = 20
TRAIN = False
DATASET_INFO = False
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'training_checkpoints')
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'cp-0020.ckpt')
#CHECKPOINT_FILE = None
DB_NAME = ':)'
DB_PORT = ':)'
DB_HOST = ':)'
DB_USER = ':)'
DB_PASS = ':)'

def main():
    ids_from_chars, chars_from_ids = setup_vocab()
    
    model = NeuralRNN(vocab_size=ids_from_chars.vocabulary_size(), embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CHECKPOINT_DIR, 'cp-{epoch:04d}.ckpt'), save_weights_only=True, verbose=1)

    if CHECKPOINT_FILE:
        model.load_weights(CHECKPOINT_FILE)

    if TRAIN:
        dataset = dataset_from_messages(load_messages(CHANNEL), ids_from_chars)
        if DATASET_INFO:
            dataset_info(model, dataset, loss)
        train(model, dataset, checkpoint_callback)

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    start = time.time()
    states = None
    next_char = tf.constant([chr(random.randrange(65, 91))])
    result = [next_char]

    for n in range(500):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
    print('\nRun time:', end - start)

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                            return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

class NeuralRNN(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

def dataset_from_messages(messages, ids_from_chars):
    tensors = []
    for message in messages:
        ids = ids_from_chars(tf.strings.unicode_split(message, 'UTF-8'))
        ids = tf.pad(ids,[[0,MAX_MESSAGE_LENGTH-len(ids)]], 'CONSTANT', constant_values=ids_from_chars(chr(0)).numpy())
        tensors.append(ids)
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.map(split_input_target)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def dataset_info(model, dataset, loss):
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        example_batch_loss = loss(target_example_batch, example_batch_predictions)
        mean_loss = example_batch_loss.numpy().mean()
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("Mean loss:        ", mean_loss)
        print(tf.exp(mean_loss).numpy())

def train(model: NeuralRNN, dataset, checkpoint_callback):
    return model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

def load_messages(channel):
    connection = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASS}")
    cursor = connection.cursor()
    cmd = sql.SQL('SELECT {} FROM {}.{} WHERE {} = {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal(channel))
    cursor.execute(cmd)
    messages = [x[0] for x in cursor.fetchall() if is_friendly(x[0])]
    print(len(messages))
    connection.commit()
    cursor.close()
    connection.close()
    return messages

def is_friendly(message, vocab=VOCAB):
    for c in message:
        if not c in vocab:
            return False
    return True

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def setup_vocab(vocab=VOCAB):
    ids_from_chars = preprocessing.StringLookup(vocabulary=vocab, mask_token=None)
    chars_from_ids = preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    return ids_from_chars, chars_from_ids

def text_from_ids(chars_from_ids, ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

if __name__ == '__main__':
    main()