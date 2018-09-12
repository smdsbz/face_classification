# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

import sqlite3
import pickle

# import io
import base64
# import matplotlib.image as mpimage

# import numpy as np
import pandas as pd

import os


''' Configuration Variables '''

BREAKPOINT_PATH = './bkpt'
OUTPUT_DIR      = './output'
ORIG_DATA_PATH  = '../FaceImageCroppedWithoutAlignment.tsv'
DATABASE_PATH   = './encodings.db'

CHUNK_SIZE      = int(1e3)


''' Session Control '''

breakpoint = 0

def load_breakpoint(path=BREAKPOINT_PATH):
    global breakpoint
    if os.path.exists(path):
        with open(path, 'r') as f:
            breakpoint = int(f.read())
    else:
        print('Breakpoint file not found in \'{}\', resetting progress...'
              .format(path))
        breakpoint = 0
    return breakpoint


def save_breakpoint(path=BREAKPOINT_PATH):
    if os.path.exists(path):
        with open(path, 'r') as f:
            old_breakpoint = int(f.read())
        if old_breakpoint > breakpoint:
            print('A later breakpoint is found: {} later than {}!'
                  .format(old_breakpoint, breakpoint))
            print('New breakpoint not saved!')
            return
    with open(path, 'w') as f:
        f.write(str(breakpoint - 1))
    print('Breakpoint at {} saved!'.format(breakpoint))


''' Helper Functions - Image '''

def save_record_to_class_folder(
        rec, klass, format='jpg',
        root_path=OUTPUT_DIR):
    '''
    save record to corresponding classified folder

    Args:
        `rec`: `pandas`-read record
        `klass`: class index of the record
        `format`: format of input image (will be used as output format too)
        `root_path`: default output root directory
    '''
    target_dir = os.path.join(root_path, str(klass))
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    target_path = os.path.join(target_dir, '{}.{}'.format(rec.name, format))
    with open(target_path, 'wb') as f:
        f.write(base64.b64decode(rec[6]))


''' Helper Functions - Database '''

def get_class_of_record(db_cursor, rec):
    '''
    retrieve class index of record from database

    Args:
        `db_cursor`: database cursor
        `rec`: `pandas`-read record

    Return:
        class index of type `int` or `None` if not found
    '''
    db_cursor.execute('''
        SELECT belong FROM encodings
        WHERE
            id = ?
        ''', (rec.name,))
    klass = db_cursor.fetchone()
    if klass is None:   # `id` not found
        print('Record with id {} not found in database!'
              .format(rec.name))
        return None
    klass = klass[0]    # NOTE: could be `int` or `None``
    return klass


def check_if_class_regular(db_cursor, klass, valve_freq=10):
    '''
    checks if a class has sufficient members

    Args:
        `db_cursor`: database cursor
        `klass`: the index of the class to be checked
        `valve_freq`: minimum number of members to be considered *regular*

    Return:
        `True` if regular, else returns `False`
    '''
    db_cursor.execute('''
        SELECT count FROM seen_classes
        WHERE
            belong = ?
        ''', (klass,))
    count = db_cursor.fetchone()
    if count is None:     # class entry deleted (for it's not frequent)
        return False
    count = count[0]
    if count < valve_freq:
        return False
    return True


''' Processes '''

def process_record(db_cursor, rec):
    global breakpoint
    print('Processing {}...'.format(rec.name))
    klass = get_class_of_record(db_cursor, rec)
    if klass is not None \
            and check_if_class_regular(db_cursor, klass):
        save_record_to_class_folder(rec, klass)
    else:
        print('Face in record {} has count more than 1, '.format(rec.name)
              + 'or of a rare class. Skipping...')
    breakpoint += 1


def process_chunk(db_connection, chunk):
    print('Starting with chunk {} - {}...'
          .format(chunk.iloc[0].name, chunk.iloc[-1].name))
    if breakpoint > chunk.iloc[-1].name:
        print('Skipping this chunk for it\'s already processed!')
        return
    db_cursor = db_connection.cursor()
    for idx, rec in chunk.iterrows():
        process_record(db_cursor, rec)
    save_breakpoint()


''' Main '''

if __name__ == '__main__':

    load_breakpoint()

    orig_reader = pd.read_csv(ORIG_DATA_PATH,
                              sep='\t', header=None,
                              chunksize=CHUNK_SIZE)

    with sqlite3.connect(DATABASE_PATH) as db_connection:
        for chunk in orig_reader:
            process_chunk(db_connection, chunk)
            print('Loading next chunk...')

    print('All done!')
