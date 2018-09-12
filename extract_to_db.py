'''
The original data is approximately 150GB in size, making it impossible to
simply load into memory with `pd.read_csv`. The main purpose of this script
is to dump base64-encoded images in original data to a folder filled with
`.pkl`s containing their face encodings in `np.ndarray`.
'''

__author__ = 'Xiaoguang Zhu'

import io
import base64

import matplotlib.image as mpimage

import numpy as np
import pandas as pd
from PIL import Image

import os
import pickle
import sqlite3

import argparse

import face_recognition as fr


''' Session Control Utilities '''

CHUNK_SIZE = int(1e4)
PROCESSES_COUNT = 7

SOURCE_DIR = os.path.join(os.pardir, 'FaceImageCroppedWithoutAlignment.tsv')
DB_PATH = os.path.join(os.curdir, 'encodings.db')
BREAKPOINT_PATH = os.path.join(os.curdir, 'bkpt')

range_start = 0
range_end = -1


def load_breakpoint(path=BREAKPOINT_PATH):
    global range_start
    try:
        with open(path, 'r') as f:
            range_start = int(f.read())
        print('Loaded breakpoint {}.'.format(range_start))
    except FileNotFoundError:
        print('Previous breakpoint not found, starting from stratch...')
        range_start = 0


def save_breakpoint(path=BREAKPOINT_PATH):
    global range_start
    try:
        with open(path, 'r') as f:
            old_breakpoint = int(f.read())
        if range_start > old_breakpoint:
            with open(path, 'w') as f:
                f.write(str(range_start - 1))   # `-1` for safe
        else:
            print('Current not saved for a newer version is detected!')
    except FileNotFoundError:
        print('Previous breakpoint not found, creating a new one...')
        with open(path, 'w') as f:
            f.write(str(range_start - 1))   # `-1` for safe


''' Helper Functions - Data Format '''

get_rec_uid = lambda rec: rec.name
get_rec_format = lambda rec: rec[2].split('.')[-1]


''' Helper Functions - Image Processing '''

def convert_rec_to_ndarray(rec, format='jpg'):
    '''
    read base64 image from record as `np.ndarray`

    :param rec: raw record from `FaceImageCroppedWithOutAlignment.tsv`
    :param format: base64 image format
    :return: `np.ndarray` of shape `[width, height, channels]`
    '''
    img = mpimage.imread(io.BytesIO(base64.b64decode(rec[6])), format=format)
    img.setflags(write=True)
    return img


''' Record Process '''

def process_record(db_cursor, idx, rec):
    '''
    process a record in original data

    :param db_cursor: database cursor
    :param idx: unique index for current record processing
    :param rec: the record under processing
    '''
    # get **unique** id of current image, must match that in `orig_reader`
    # NOTE: `rec[1]` is not unique!
    # HACK: using `iloc` in `orig_idx`, this requires that orders in
    #       original data does **NOT** change
    rec_uid = idx
    # skip if already in database
    SQL = '''
        SELECT id FROM encodings WHERE id = ?
    '''
    db_cursor.execute(SQL, (rec_uid,))
    if db_cursor.fetchall():
        print('Current record already in database, skipping...')
    else:
        # NOTE: believe me, they're all `jpg` images
        imgarr = convert_rec_to_ndarray(rec, format='jpg')
        encodings = fr.face_encodings(imgarr)
        SQL = '''
            INSERT INTO encodings (id, encoding) VALUES (?, ?)
        '''
        db_cursor.execute(SQL, (rec_uid,
                                sqlite3.Binary(pickle.dumps(encodings)))
        )
        global range_start
        range_start = rec_uid


# def process_records(db_cursor, idxes, recs):
#     '''
#     process a bunch of records, using process-parallel while calculateing
#     face encodings

#     :param db_cursor: database cursor
#     :param idxes: unique indexes of records
#     :param records: pool of records to be processed
#     '''
#     rec_uids = idxes
#     # decide records to calculate
#     if_exist_mask = []
#     for rec_uid in rec_uids:
#         SQL = '''
#             SELECT id FROM encodings WHERE id = ?
#         '''
#         db_cursor.execute(SQL, (rec_uid,))
#         if db_cursor.fetchall():
#             if_exist_mask.append(True)
#             print('Record {} already in database, skipping...'
#                   .format(rec_uid))
#         else:
#             if_exist_mask.append(False)
#     # create job pool
#     job_rec_uids = [
#         rec_uids[e] for e in range(len(recs)) if not if_exist_mask[e]
#     ]
#     job_records_pool = [
#         recs[e] for e in range(len(recs)) if not if_exist_mask[e]
#     ]
#     # do jobs in parallel
#     with Pool(processes=PROCESSES_COUNT) as p:
#         pkls = p.map(
#             _dump_rec_to_pkl,
#             job_records_pool
#         )
#     # register changes in database cursor
#     for job_rec_uid, pkl in zip(job_rec_uids, pkls):
#         SQL = '''
#             INSERT INTO encodings (id, encoding) VALUES (?, ?)
#         '''
#         db_cursor.execute(SQL, (job_rec_uid, sqlite3.Binary(pkl)))
#     print('Process for {} finished'.format(job_rec_uids))
#     global range_start
#     range_start = max(idxes)


''' Chunk Process '''

def process_chunk(chunk, db_connection, nop=False):
    '''
    process a chunk of data

    :param chunk: chunk of data
    :param db_connection: database connection
    :param nop: if `True`, do nothing!
    '''
    print('Chunk range: {}-{}'.format(chunk.iloc[0].name,
                                      chunk.iloc[-1].name))
    if nop:
        return
    if range_start > chunk.iloc[-1].name:
        print('Skipping this chunk for it\'s already processed!')
        return
    elif range_end != -1 and range_end < chunk.iloc[0].name:
        print('Reached specified range!')
        return
    else:
        for orig_idx, rec in chunk.iterrows():
            print('Processing {}'.format(orig_idx))
            process_record(db_connection.cursor(),
                           orig_idx,
                           rec)
        save_breakpoint()
        db_connection.commit()
        print('Breakpoint {} saved. Loading next chunk...'
              .format(range_start))


# def process_chunk_parallel(chunk, db_connection, nop=False):
#     print('Chunk range: {}-{}'.format(chunk.iloc[0].name,
#                                       chunk.iloc[-1].name))
#     if nop:
#         return
#     if range_start > chunk.iloc[-1].name:
#         print('Skipping this chunk for it\'s already processed!')
#         return
#     elif range_end != -1 and range_end < chunk.iloc[0].name:
#         print('Reached specified range!')
#         return
#     else:
#         idx_pool, rec_pool = [], []
#         for orig_idx, rec in chunk.iterrows():
#             idx_pool.append(orig_idx)
#             rec_pool.append(rec)
#             if len(idx_pool) == PROCESSES_COUNT:
#                 print('Forking worker processes...')
#                 process_records(db_connection.cursor(),
#                                 idx_pool,
#                                 rec_pool)
#                 print('Workers joined, clearing data pool...')
#                 idx_pool, rec_pool = [], []
#         if idx_pool:
#             process_records(db_connection.cursor(),
#                             idx_pool,
#                             rec_pool)
#             # idx_pool, rec_pool = [], []     # HACK: exiting, not needed
#         save_breakpoint()
#         db_connection.commit()
#         print('Breakpoint {} saved. Loading next chunk...'
#               .format(range_start))


''' Main Process '''

parser = argparse.ArgumentParser(
    description='''Extract face encodings as `np.ndarray`s in `.tsv`
file to disk'''
)
parser.add_argument('--range', nargs=2, type=int,
                    help='''Provide starting and ending range. Overriding
existing breakpoint.''')
parser.add_argument('-n', '--no-op', action='store_true',
                    help='''Do nothing, just go through the data set.''')
parser.add_argument('-cs', '--chunk-size', default=int(1e3),
                    help='''Chunk size. Large as your RAM can hold.''')

args = parser.parse_args()
print(args)

if args.range is not None:
    range_start = args.range[0]
    range_end = args.range[1]
else:
    load_breakpoint()
    range_end = -1

NOP = args.no_op
CHUNK_SIZE = int(args.chunk_size)


def main(db_connection):
    orig_reader = pd.read_csv(SOURCE_DIR,
                              sep='\t', header=None,
                              chunksize=CHUNK_SIZE)
    if not NOP:
        try:
            for chunk in orig_reader:
                print('Processing chunk...')
                process_chunk(chunk, db_connection)
            print('All done!')
        except KeyboardInterrupt:
            save_breakpoint()
            db_connection.commit()
            print('Breaked manually!')
    else:
        for chunk in orig_reader:
            process_chunk(chunk, db_connection, nop=True)
        print('All done!')


if __name__ == '__main__':
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS encodings (
                id          integer     primary key     ,
                encoding    blob        not null        ,
                belong      integer
            )'''
        )

        main(conn)
