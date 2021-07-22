from posixpath import split
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np

import psycopg2
from psycopg2 import sql
from tensorflow.python.ops.gen_batch_ops import batch

VOCAB = [chr(0), *(chr(x) for x in range(32, 127))]
CHANNEL = 'rlly'
MAX_MESSAGE_LENGTH = 500
BATCH_SIZE = 128
DATASET_INFO = False

DB_NAME = ':)'
DB_PORT = ':)'
DB_HOST = ':)'
DB_USER = ':)'
DB_PASS = ':)'

def main():
    ids_from_chars, chars_from_ids = setup_vocab()

def dataset_from_messages(messages, ids_from_chars):
    tensors = []
    for message in messages:
        ids = ids_from_chars(tf.strings.unicode_split(message, 'UTF-8'))
        ids = tf.cast(ids, tf.uint8)
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

def get_channel_rows(channel=CHANNEL):
    connection = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASS}")
    cursor = connection.cursor()
    cmd = sql.SQL('SELECT COUNT({}) FROM {}.{} WHERE {} = {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal(channel))
    cursor.execute(cmd)
    rows = cursor.fetchall()[0][0]
    connection.commit()
    cursor.close()
    connection.close()
    return rows

def get_rows():
    connection = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASS}")
    cursor = connection.cursor()
    cmd = sql.SQL('SELECT COUNT({}) FROM {}.{} WHERE {} != {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal('xQcOW'))
    cursor.execute(cmd)
    rows = cursor.fetchall()[0][0]
    connection.commit()
    cursor.close()
    connection.close()
    return rows

def get_all_channel_messages_from_sql(channel=CHANNEL):
    connection = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASS}")
    cursor = connection.cursor()
    cmd = sql.SQL('SELECT {} FROM {}.{} WHERE {} = {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal(channel))
    cursor.execute(cmd)
    messages = [x[0] for x in cursor.fetchall() if is_friendly(x[0])]
    print(f'Number of messages: {len(messages)}')
    connection.commit()
    cursor.close()
    connection.close()
    return messages

def get_select_channel_messages_from_sql(channel=CHANNEL, batch_size=BATCH_SIZE, offset=0):
    connection = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASS}")
    cursor = connection.cursor()
    cmd = sql.SQL('SELECT {} FROM {}.{} WHERE {} = {} LIMIT {} OFFSET {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal(channel), sql.Literal(batch_size), sql.Literal(offset))
    cursor.execute(cmd)
    messages = [x[0] for x in cursor.fetchall() if is_friendly(x[0])]
    connection.commit()
    cursor.close()
    connection.close()
    return messages

def get_select_messages_from_sql(batch_size=BATCH_SIZE, offset=0):
    connection = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASS}")
    cursor = connection.cursor()
    cmd = sql.SQL('SELECT {} FROM {}.{} WHERE {} != {} LIMIT {} OFFSET {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal('xQcOW'), sql.Literal(batch_size), sql.Literal(offset))
    cursor.execute(cmd)
    messages = [x[0] for x in cursor.fetchall() if is_friendly(x[0])]
    connection.commit()
    cursor.close()
    connection.close()
    return messages

def is_friendly(message, vocab=VOCAB):
    for c in message:
        if not c in vocab:
            return False
    return True

def setup_vocab(vocab=VOCAB):
    ids_from_chars = preprocessing.StringLookup(vocabulary=vocab, mask_token=None)
    chars_from_ids = preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    return ids_from_chars, chars_from_ids

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def sql_channel_dataset_generator(ids_from_chars, channel=CHANNEL, batch_size=BATCH_SIZE):
    offset = 0
    limit = get_channel_rows(channel)
    while offset + batch_size < limit:
        messages = get_select_channel_messages_from_sql(channel, batch_size, offset)
        original = []
        target = []
        tensors = []
        for message in messages:
            ids = ids_from_chars(tf.strings.unicode_split(message, 'UTF-8'))
            ids = tf.cast(ids, tf.uint8)
            ids = tf.pad(ids,[[0,MAX_MESSAGE_LENGTH-len(ids)]], 'CONSTANT', constant_values=ids_from_chars(chr(0)).numpy())
            tensors.append(ids)
            x, y = split_input_target(ids)
            original.append(x)
            target.append(y)
        offset += batch_size
        print(np.array(original), np.array(target))
        yield (np.array(original), np.array(target))

def sql_dataset_generator(ids_from_chars, batch_size=BATCH_SIZE):
    offset = 0
    limit = get_rows()
    while offset + batch_size < limit:
        messages = get_select_messages_from_sql(batch_size, offset)
        original = []
        target = []
        for message in messages:
            ids = ids_from_chars(tf.strings.unicode_split(message, 'UTF-8'))
            ids = tf.cast(ids, tf.uint8)
            ids = tf.pad(ids,[[0,MAX_MESSAGE_LENGTH-len(ids)]], 'CONSTANT', constant_values=ids_from_chars(chr(0)).numpy())
            x, y = split_input_target(ids)
            original.append(x)
            target.append(y)
        offset += batch_size
        yield (np.array(original), np.array(target))
        

if __name__ == "__main__":
    main()