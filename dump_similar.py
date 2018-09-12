# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

from face_recognition import compare_faces

import sqlite3
import pickle
import base64

import pandas as pd

import os


''' Configuration Variables '''

BASE_CLASS          = 0

BREAKPOINT_PATH     = './bkpt_{}_similar'.format(BASE_CLASS)
OUTPUT_DIR          = './output-{}-similar'.format(BASE_CLASS)

ORIGINAL_DATA_PATH  = '../FaceImageCroppedWithoutAlignment.tsv'
CHUNK_SIZE          = int(1e3)

DATABASE_PATH       = './encodings.db'


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
            print('Breakpoint not saved!')
            return
    with open(path, 'w') as f:
        f.write(str(breakpoint - 1))
    print('Breakpoint at {} saved!'.format(breakpoint))


''' Helper Functions - Database '''

def retrieve_encodings_by_id(db_cursor, idx):
    db_cursor.execute(
        '''
        SELECT encoding FROM encodings
        WHERE
            id = ?
        ''',
        (idx,)
    )
    encodings = db_cursor.fetchone()
    if encodings is None:
        return None
    encodings = encodings[0]
    encodings = pickle.loads(encodings)
    return encodings


''' Helper Functions - File I/O '''

def save_record_to_folder(
        rec,
        format='jpg',
        path=OUTPUT_DIR):
    '''
    save record to corresponding classified folder

    Args:
        `rec`: `pandas`-read record
        `format`: format of input image (will be used as output format too)
        `path`: output directory
    '''
    target_path = os.path.join(path, '{}.{}'.format(rec.name, format))
    with open(target_path, 'wb') as f:
        f.write(base64.b64decode(rec[6]))


''' Process '''

def process_record(db_cursor, target_encoding, rec):
    global breakpoint
    print('Processing {}...'.format(rec.name))
    other_encoding = retrieve_encodings_by_id(db_cursor, rec.name)
    if other_encoding is None or len(other_encoding) != 1:
        pass
    else:
        if compare_faces(target_encoding, other_encoding[0],
                         tolerance=0.54)[0]:
            save_record_to_folder(rec)
    breakpoint = rec.name


def process_chunk(db_connection, target_encoding, chunk):
    print('Starting with chunk {} - {}...'
          .format(chunk.iloc[0].name, chunk.iloc[1].name))
    if breakpoint > chunk.iloc[-1].name:
        print('Skipping this chunk for it\'s already been processed...')
        return
    db_cursor = db_connection.cursor()
    for idx, rec in chunk.iterrows():
        process_record(db_cursor, target_encoding, rec)
    save_breakpoint()


def main():
    load_breakpoint()
    orig_reader = pd.read_csv(ORIGINAL_DATA_PATH,
                              sep='\t', header=None,
                              chunksize=CHUNK_SIZE)
    with sqlite3.connect(DATABASE_PATH) as db_connection:
        db_cursor = db_connection.cursor()
        target_encoding = retrieve_encodings_by_id(db_cursor, BASE_CLASS)
        if target_encoding is None or len(target_encoding) != 1:
            raise ValueError('Bad choice on base class!')
        for chunk in orig_reader:
            process_chunk(db_connection, target_encoding, chunk)
            print('Loading next chunk...')
    print('All done!')


''' Main '''

if __name__ == '__main__':
    main()

