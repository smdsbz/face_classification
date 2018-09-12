# -*- coding: utf-8 -*-

__author__ = '__main__'

''''''

import pandas as pd

import sqlite3

orig_reader = pd.read_csv('../FaceImageCroppedWithoutAlignment.tsv',
                          sep='\t', header=None, chunksize=1000)

with sqlite3.connect('./encodings.db') as conn:
    c = conn.cursor()
    for chunk in orig_reader:
        for idx, rec in chunk.iterrows():
            orig_class = rec[0]
            print('ID {} : {}'.format(rec.name, orig_class))
            c.execute(
                '''
                UPDATE encodings
                SET
                    orig_class = ?
                WHERE
                    id = ?
                ''',
                (orig_class, rec.name)
            )
        conn.commit()
    print('All done!')

