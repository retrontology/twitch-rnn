import tensorflow as tf

import os
import random

from create_dataset import *


EMBEDDING_DIM = 256
RNN_UNITS = 2048
EPOCHS = 100
TRAIN = True
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'training_checkpoints')
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, f'{os.path.basename(SAVE_FILE)}.cpkt')
#CHECKPOINT_FILE = None

def main():
    ids_from_chars, chars_from_ids = setup_vocab()
    
    model = NeuralRNN(vocab_size=ids_from_chars.vocabulary_size(), embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CHECKPOINT_DIR, f'{os.path.basename(SAVE_FILE)}.cpkt'), save_weights_only=True, verbose=1, monitor='accuracy', save_best_only=True)

    if CHECKPOINT_FILE and os.path.exists(CHECKPOINT_FILE):
        model.load_weights(CHECKPOINT_FILE)

    if TRAIN:
        #model.fit(sql_channel_dataset_generator(ids_from_chars, channel=CHANNEL, batch_size=BATCH_SIZE), steps_per_epoch=int(get_channel_rows(CHANNEL)/BATCH_SIZE), epochs=EPOCHS, callbacks=[checkpoint_callback])
        #model.fit(sql_dataset_generator(ids_from_chars, batch_size=BATCH_SIZE), steps_per_epoch=int(get_rows()/BATCH_SIZE), epochs=EPOCHS, callbacks=[checkpoint_callback])
        dataset, count = read_channel_dataset_from_file()
        model.fit(dataset, steps_per_epoch=int(count/BATCH_SIZE), epochs=EPOCHS, callbacks=[checkpoint_callback])

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    print(f'{generate_message(one_step_model, "@MajorEcho")}') #chr(random.randrange(65, 91))
    

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

def checkpoint_print_callback():
    pass

def generate_message(one_step_model, seed):
    states = None
    next_char = tf.constant([seed])
    result = [next_char]

    for n in range(500):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        if next_char[0].numpy() == b'\x00':
            break
        else:
            result.append(next_char)

    result = tf.strings.join(result)
    return result[0].numpy().decode('utf-8')

if __name__ == '__main__':
    main()