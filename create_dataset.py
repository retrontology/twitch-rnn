import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
from glob import glob

import psycopg2
from psycopg2 import sql

import emoji

from multiprocessing.pool import ThreadPool

VOCAB = [chr(0), *(chr(x) for x in range(32, 127))]
IDS_FROM_CHARS = preprocessing.StringLookup(vocabulary=VOCAB, mask_token=None)
CHARS_FROM_IDS = preprocessing.StringLookup(vocabulary=IDS_FROM_CHARS.get_vocabulary(), invert=True, mask_token=None)

CHANNEL = 'rlly'
MAX_MESSAGE_LENGTH = 500
BATCH_SIZE = 128
DATASET_INFO = False
#SAVE_FILE = os.path.join(os.path.dirname(__file__), f'{CHANNEL}')
SAVE_FILE = 'allchat'
SQL_PAGE_SIZE = 1024

DB_NAME = ':)'
DB_PORT = ':)'
DB_HOST = ':)'
DB_USER = ':)'
DB_PASS = ':)'

def main():
    #write_channel_messages_to_file(IDS_FROM_CHARS, CHANNEL, SAVE_FILE)
    write_all_messages_to_file()

def dataset_from_messages(messages):
    tensors = []
    for message in messages:
        ids = IDS_FROM_CHARS(tf.strings.unicode_split(message, 'UTF-8'))
        ids = tf.cast(ids, tf.uint8)
        ids = tf.pad(ids,[[0,MAX_MESSAGE_LENGTH-len(ids)]], 'CONSTANT', constant_values=IDS_FROM_CHARS(chr(0)).numpy())
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

def decode_fn(record_bytes):
    data = tf.io.parse_single_example(record_bytes, {"x": tf.io.FixedLenFeature([MAX_MESSAGE_LENGTH-1], dtype=tf.int64), "y": tf.io.FixedLenFeature([MAX_MESSAGE_LENGTH-1], dtype=tf.int64)})
    return (tf.cast(data['x'], tf.uint8), tf.cast(data['y'], tf.uint8))

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
    cmd = sql.SQL('SELECT COUNT({}) FROM {}.{} WHERE {} != {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal('xqcow'))
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
    messages = [emoji.demojize(x[0]) for x in cursor.fetchall() if is_friendly(x[0])]
    connection.commit()
    cursor.close()
    connection.close()
    return messages

def get_all_messages_from_sql():
    connection = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASS}")
    cursor = connection.cursor()
    cmd = sql.SQL('SELECT {} FROM {}.{} WHERE {} != {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal('xqcow'))
    cursor.execute(cmd)
    messages = [emoji.demojize(x[0]) for x in cursor.fetchall() if is_friendly(x[0])]
    connection.commit()
    cursor.close()
    connection.close()
    return messages

def get_select_channel_messages_from_sql(channel=CHANNEL, batch_size=BATCH_SIZE, offset=0):
    connection = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASS}")
    cursor = connection.cursor()
    cmd = sql.SQL('SELECT {} FROM {}.{} WHERE {} = {} LIMIT {} OFFSET {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal(channel), sql.Literal(batch_size), sql.Literal(offset))
    cursor.execute(cmd)
    messages = [emoji.demojize(x[0]) for x in cursor.fetchall() if is_friendly(x[0])]
    connection.commit()
    cursor.close()
    connection.close()
    return messages

def get_select_messages_from_sql(batch_size=SQL_PAGE_SIZE, offset=0):
    connection = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASS}")
    cursor = connection.cursor()
    cmd = sql.SQL('SELECT {} FROM {}.{} WHERE {} != {} LIMIT {} OFFSET {};').format(sql.Identifier('content'), sql.Identifier(f'twitchlogger'), sql.Identifier(f'chat'), sql.Identifier('channel'), sql.Literal('xqcow'), sql.Literal(batch_size), sql.Literal(offset))
    cursor.execute(cmd)
    messages = [emoji.demojize(x[0]) for x in cursor.fetchall() if is_friendly(x[0])]
    connection.commit()
    cursor.close()
    connection.close()
    return messages

def is_friendly(message, vocab=VOCAB):
    for c in set(message):
        if not c in vocab:
            return False
    return True

def read_dataset_from_file(path=SAVE_FILE, batch_size=BATCH_SIZE):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        exit('Save directory not found!')
    dir = path
    prefix = os.path.basename(path)
    files = glob(os.path.join(dir, f'{prefix}-*.tfrecord'))
    return tf.data.TFRecordDataset(files).map(decode_fn).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).repeat(), len(files)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def sql_channel_dataset_generator(channel=CHANNEL, batch_size=BATCH_SIZE):
    offset = 0
    limit = get_channel_rows(channel)
    while offset + batch_size < limit:
        messages = get_select_channel_messages_from_sql(channel, batch_size, offset)
        original = []
        target = []
        for message in messages:
            ids = IDS_FROM_CHARS(tf.strings.unicode_split(message, 'UTF-8'))
            ids = tf.cast(ids, tf.uint8)
            ids = tf.pad(ids,[[0,MAX_MESSAGE_LENGTH-len(ids)]], 'CONSTANT', constant_values=IDS_FROM_CHARS(chr(0)).numpy())
            x, y = split_input_target(ids)
            original.append(x)
            target.append(y)
        offset += batch_size
        yield (np.array(original), np.array(target))

def sql_dataset_generator(batch_size=BATCH_SIZE):
    offset = 0
    limit = get_rows()
    while offset + batch_size < limit:
        messages = get_select_messages_from_sql(batch_size, offset)
        original = []
        target = []
        for message in messages:
            ids = IDS_FROM_CHARS(tf.strings.unicode_split(message, 'UTF-8'))
            ids = tf.cast(ids, tf.uint8)
            ids = tf.pad(ids,[[0,MAX_MESSAGE_LENGTH-len(ids)]], 'CONSTANT', constant_values=IDS_FROM_CHARS(chr(0)).numpy())
            x, y = split_input_target(ids)
            original.append(x)
            target.append(y)
        offset += batch_size
        yield (np.array(original), np.array(target))

def write_channel_messages_to_file(channel=CHANNEL, path=SAVE_FILE):
    messages = get_all_channel_messages_from_sql(channel)
    path = os.path.abspath(path)
    if os.path.exists(path):
        exit('Save file already exists!')
    else:
        os.mkdir(path)
    zero_pad = len(str(len(messages)))
    dir = path
    prefix = os.path.basename(path)
    progress = tf.keras.utils.Progbar(len(messages), unit_name='message')
    index = 0
    for message in messages:
        if is_friendly(message):
            path = os.path.join(dir, f'{prefix}-{str(index).zfill(zero_pad)}.tfrecord')
            file_writer = tf.io.TFRecordWriter(path) 
            ids = IDS_FROM_CHARS(tf.strings.unicode_split(message, 'UTF-8'))
            ids = tf.pad(ids,[[0,MAX_MESSAGE_LENGTH-len(ids)]], 'CONSTANT', constant_values=IDS_FROM_CHARS(chr(0)).numpy())
            x, y = split_input_target(ids)
            example = tf.train.Example(features=tf.train.Features(feature={
            'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x.numpy())), 
            'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.numpy()))}))
            file_writer.write(example.SerializeToString())
            file_writer.close()
            index += 1
        progress.add(1)
    file_writer.close()

def write_all_messages_to_file(path=SAVE_FILE, sql_page_size=SQL_PAGE_SIZE, num_proc=8):
    path = os.path.abspath(path)
    if os.path.exists(path):
        exit('Save file already exists!')
    else:
        os.mkdir(path)
    sql_rows = get_rows()
    zero_pad = len(str(sql_rows))
    dir = path
    prefix = os.path.basename(path)
    progress = tf.keras.utils.Progbar(None, unit_name='message')
    file_index = 0
    sql_offset = 0
    next_messages = get_select_messages_from_sql(sql_page_size, sql_offset)

    def progress_callback():
        progress.add(1)

    while sql_offset + sql_page_size < sql_rows:
        messages = next_messages
        pool = ThreadPool(8)
        for message in messages:
            if is_friendly(message):
                pool.apply(func=write_message_to_file, args=(message, dir, prefix, file_index, zero_pad, progress_callback))
                file_index += 1
        sql_offset += sql_page_size
        next_messages = get_select_messages_from_sql(sql_page_size, sql_offset)
        pool.close()
        pool.join()
    progress.update(progress._last_update, finalize=True)

def write_message_to_file(message, dir, prefix, index, zero_pad, callback):
    path = os.path.join(dir, f'{prefix}-{str(index).zfill(zero_pad)}.tfrecord')
    file_writer = tf.io.TFRecordWriter(path)
    ids = IDS_FROM_CHARS(tf.strings.unicode_split(message, 'UTF-8'))
    ids = tf.pad(ids,[[0,MAX_MESSAGE_LENGTH-len(ids)]], 'CONSTANT', constant_values=IDS_FROM_CHARS(chr(0)).numpy())
    x, y = split_input_target(ids)
    example = tf.train.Example(features=tf.train.Features(feature={
    'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x.numpy())), 
    'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.numpy()))}))
    file_writer.write(example.SerializeToString())
    file_writer.close()
    if callback:
        callback()
    
if __name__ == "__main__":
    main()