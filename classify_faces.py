# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

# import numpy as np
import pandas as pd

import os

import base64
import pickle

import sqlite3

import face_recognition as fr


''' Configurations '''


ORIGINAL_DATA_PATH = '../FaceImageCroppedWithoutAlignment.tsv'
CHUNK_SIZE = int(1e3)

TOP_DB_PATH = './encodings.db'
SEGMENT_DB_DIR = './segment_db'

OUTPUT_ROOT_PATH = './output'


''' Helper Functions - Session Controll '''


def load_breakpoint():
    with sqlite3.connect(TOP_DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            '''
            SELECT * from breakpoints
            ORDER BY idx DESC
            LIMIT 1
            '''
        )
        bkpt = c.fetchone()
    if bkpt is None:
        bkpt = 0
        print('No breakpoint found, starting from 0...')
    else:
        bkpt = bkpt[0]
        print('Loaded breakpoint idx = {}'.format(bkpt))
    return bkpt


def save_breakpoint(conn, idx):
    c = conn.cursor()
    c.execute(
        '''
        UPDATE breakpoints
        SET
            idx = ?
        ''',
        (idx,)
    )
    conn.commit()
    print('Breakpoint @ {} saved!'.format(idx))


''' Helper Functions - DataType Casting '''


def get_encoding_from_record(record_tuple):
    '''
    extract encoding(s) from database record

    Args:
        `record_tuple`: record tuple returned from `db_cursor.fetchone()`

    Return:
        `[ np.ndarray() ]`: list of encodings
    '''
    ret = record_tuple[1]
    ret = pickle.loads(ret)
    return ret


''' Helper Functions - Database '''


def update_count(db_cursor, record):
    '''
    updates the `count` column of record in top-level database

    Note:
        This method will **NOT** commit changes to database, you **HAVE TO CALL
        `db_connection.commit()` MANUALLY**!

    Args:
        `db_cursor`: top-level database cursor
        `record`: record tuple

    Return:
        count of encodings, i.e. detected faces, in `record`
    '''
    idx = record[0]
    encodings = get_encoding_from_record(record)
    count = len(encodings)
    db_cursor.execute('''
        UPDATE encodings SET
            count = ?
        WHERE
            id = ?
        ''', (count, idx))
    return count


def increase_class_count(seg_cursor, klass):
    '''
    increase registered class member count by 1

    NOTE: `klass` must be in segment database __ALREADY__!

    Args:
        `seg_cursor`: segment database cursor
        `klass`: target class ID
    '''
    seg_cursor.execute(
        '''
        SELECT count FROM seen_classes
        WHERE
            belong = ?
        ''',
        (klass,)
    )
    orig_count = seg_cursor.fetchone()[0]
    seg_cursor.execute(
        '''
        UPDATE seen_classes
        SET
            count = ?
        WHERE
            belong = ?
        ''',
        (orig_count + 1, klass)
    )


def new_class_stat(seg_cursor, klass, record_id):
    seg_cursor.execute(
        '''
        INSERT INTO seen_classes
            (belong, first_id, count)
        VALUES (?, ?, 1)
        ''',
        (klass, record_id)
    )


def get_encoding_in_class(top_db_cursor, seg_cursor, klass):
    '''
    get typical encoding that was classifed with ID `klass` in segment
    specified with the database cursor `seg_cursor`

    Note:
        Assuming records labeled with class id have one and **ONLY ONE**
        face encoding in `encoding` blob

    Args:
        `top_db_cursor`: top-level database (containing encodings) cursor
        `seg_cursor`: segment database cursor
        `klass`: class ID

    Return:
        `[ np.ndarray() ]`: a list containing a single encoding array
    '''
    # get typical record id from table `seen_classes`
    seg_cursor.execute(
        '''
        SELECT first_id FROM seen_classes
        WHERE
            belong = ?
        ''',
        (klass,)
    )
    record_id = seg_cursor.fetchone()[0]
    # retrieve encoding of that record
    top_db_cursor.execute('''
        SELECT * FROM encodings
        WHERE
            id = ?
        ''', (record_id,))
    record = top_db_cursor.fetchone()
    ret = get_encoding_from_record(record)
    return ret


def get_class_list_order_by_count(seg_cursor):
    '''
    get class IDs ordered by member counts

    Args:
        `seg_cursor`: segment database cursor

    Return:
        `[ int ]`: list of class IDs in `int`, could be empty list `[]`
    '''
    seg_cursor.execute(
        '''
        SELECT belong FROM seen_classes
        ORDER BY count DESC
        '''
    )
    ret = seg_cursor.fetchall()
    ret = list(map(lambda tup: tup[0], ret))
    return ret


''' Helper Functions - File I/O '''


def dump_to_folder(orig_data_rec, klass, format='jpg'):
    '''
    dump image file to corresponding folder

    Args:
        `orig_data_rec`: original data read in by `pandas`
        `klass`: class ID in its segment
    '''
    # segment folder
    target_path = os.path.join(OUTPUT_ROOT_PATH, orig_data_rec[0])
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    # class folder
    target_path = os.path.join(target_path, '{}'.format(klass))
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    # end point path
    target_path = os.path.join(target_path,
                               '{}.{}'.format(orig_data_rec.name, format))
    # dump to file
    with open(target_path, 'wb') as of:
        of.write(base64.b64decode(orig_data_rec[6]))


''' Processes '''


def record_process(top_db_cursor, seg_cursor, orig_rec):
    '''
    individual record process

    Args:
        `top_db_cursor`: top-level database cursor
        `seg_cursor`: segment database cursor
        `orig_rec`: original data read by `pandas`
    '''
    idx = orig_rec.name
    print('Processing {}...'.format(idx))
    # get encoding record
    top_db_cursor.execute(
        '''
        SELECT * FROM encodings
        WHERE
            id = ?
        ''',
        (idx,)
    )
    record = top_db_cursor.fetchone()
    if record is None:
        print('Not found in database, uncached record!')
        return
    # get encodings list data
    encoding = get_encoding_from_record(record)
    top_db_cursor.execute(
        '''
        UPDATE encodings
        SET
            count = ?
        WHERE
            id = ?
        ''',
        (len(encoding), idx)
    )
    # encodings count validation
    if len(encoding) != 1:
        print('{} faces detected in {}, skipping...'
              .format(len(encoding), idx))
        return
    encoding = encoding[0]
    # find its class id
    found_match = False
    klasses = get_class_list_order_by_count(seg_cursor)
    for klass in klasses:
        if fr.compare_faces(
                    get_encoding_in_class(top_db_cursor,
                                          seg_cursor,
                                          klass),
                    encoding,
                    tolerance=0.55      # TODO: TUNE THIS!!!
                )[0]:
            found_match = True
            target_klass = klass
            # register in stat table
            increase_class_count(seg_cursor, klass)
    if not found_match:
        # HACK: if class IDs are `0`-indexed, and no deletions are
        #       applied, they would be sequential naturally in this
        #       fashion
        target_klass = len(klasses)
        new_class_stat(seg_cursor, target_klass, idx)
    # dump to file
    dump_to_folder(orig_rec, target_klass)


def chunk_process(top_db_connection, chunk, last_breakpoint):
    print('Processing chunk {} - {}...'
          .format(chunk.iloc[0].name, chunk.iloc[-1].name))
    if last_breakpoint > chunk.iloc[-1].name:
        print('Skipping this chunk: already processed...')
        return
    top_db_cursor = top_db_connection.cursor()
    last_seg = 'some random text that would never appear as face id'
    seg_connection = None
    for _, orig_rec in chunk.iterrows():
        # manage segment database connection
        # NOTE: Performance: you have to keep `sqlite3.commit()` called as few
        #       as possible, therefore database connections must be kept
        #       accross record processes!
        current_seg = orig_rec[0]
        # check if need to change segment database connection
        if last_seg != current_seg:     # segment changed (or no previous connection)
            # commit changes if there is a previous connection
            if seg_connection is not None:
                seg_connection.commit()
                seg_connection.close()
            # open new segment database connection
            seg_db_path = os.path.join(SEGMENT_DB_DIR,
                                       '{}.db'.format(current_seg))
            seg_connection = sqlite3.connect(seg_db_path)
            seg_connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS seen_classes (
                    belong      integer     primary key     ,
                    first_id    integer     not null unique ,
                    count       integer     not null        )
                '''
            )
        seg_cursor = seg_connection.cursor()
        record_process(top_db_cursor, seg_cursor, orig_rec)
        last_seg = current_seg
    # close segment connection
    seg_connection.commit()
    seg_connection.close()
    # save checkpoint at top-level database
    top_db_connection.commit()
    breakpoint = chunk.iloc[-1].name
    save_breakpoint(top_db_connection, breakpoint)
    return breakpoint


''' Main '''


def main():

    orig_reader = pd.read_csv(ORIGINAL_DATA_PATH,
                              sep='\t', header=None,
                              chunksize=CHUNK_SIZE)

    last_breakpoint = load_breakpoint()

    with sqlite3.connect(TOP_DB_PATH) as top_db_connection:
        for chunk in orig_reader:
            chunk_process(top_db_connection, chunk, last_breakpoint)


if __name__ == '__main__':
    print('Staring...')
    main()
    print('All Done!')

