# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
from csv import DictReader
import os
import mxnet as mx
import numpy as np
# import random

from config import *
import logging
from multiprocessing import pool
from multiprocessing.pool import ThreadPool as Pool

seed = 103

def my_randint(L, R):
    global seed
    seed = ((seed + (R + L)) % 110100917)
    return (seed)%(R-L) + L

def get_synthesized_data(data_dir, data_name, is_merge=False):
    print("Begin Generating random dataset")
    dns, label = preprocess_data(data_name, is_merge)
    print("Generation done")
    return dns, label

def preprocess_data(data_name, is_merge):
    """Some tricks of feature engineering are adapted
    from tensorflow's wide and deep tutorial.
    """
    if data_name not in ['train', 'test']:
        raise Exception

    dns_ncols = EMBEDDING_FIELD_NUM

    label_list = []
    dns_list = []

    data_set_size = TRAINING_SET_SIZE if data_name == 'train' else EVAL_SET_SIZE
    for i in range(data_set_size):
        label_list.append(my_randint(0, 1))
        dns_row = [0] * dns_ncols
        tmp = 0
        for r in range(dns_ncols):
            dns_row[r] = my_randint(0, EMBEDDING_FEATURE_NUM[r]-1) + (0 if not is_merge else tmp)
            tmp += EMBEDDING_FEATURE_NUM[r]
        dns_list.append(dns_row)

    # convert to ndarrays
    dns = np.array(dns_list)
    label = np.array(label_list)
    return dns, label

class CriteoIterator(mx.io.DataIter):
    def __init__(self,
                data_name,
                label_name,
                batch_size,
                src_file_name=None,
                rank=0,
                size=1,
                in_mem=False):
        super(CriteoIterator, self).__init__(batch_size)
        self.batch_size = batch_size
        self.in_mem = in_mem
        self.data_iter = self.data_gen() if not self.in_mem else self.in_mem_data_gen()
        self.cur_batch = 0
        self.src_file_name = src_file_name
        self._provide_data = [(data_name, (batch_size, CRITEO_FIELD_NUM))]
        self._provide_label = [(label_name, (batch_size,))]
        self.rank=rank
        self.size=size
        self.pool = Pool(16)

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0
        self.data_iter = self.data_gen() if not self.in_mem else self.in_mem_data_gen()

    def __next__(self):
        return self.next()

    def in_mem_data_gen(self):
        feat_sizes = CRITEO_FEATURE_NUM
        with open(self.src_file_name) as _file_:
            print ("generate data")
            lines = _file_.readlines()
            sample_num = len(lines)
            data_x = [[0 for i in range(len(feat_sizes))] for j in range(sample_num//self.size)]
            data_y = [[0] for j in range(sample_num//self.size)]
            for i in range(sample_num // (self.size*self.batch_size)):
                for j in range(self.batch_size):
                    line = lines[i*self.size*self.batch_size + j + self.rank*self.batch_size]
                    data_y[i*self.batch_size + j] = int(line[0])
                    line = line.split(',')[1:]
                    data_x[i*self.batch_size + j] = [hash(line[j])%feat_sizes[j]\
                            for j in range(len(line))]
            for i in range(sample_num //(self.size*self.batch_size)):
                yield np.array(data_x[self.cur_batch*self.batch_size: (self.cur_batch+1)*self.batch_size]), \
                        np.array(data_y[self.cur_batch*self.batch_size: (self.cur_batch+1)*self.batch_size])

    def data_gen(self):
        """
        feat_sizes = [4389, 8000, 329, 7432, 2646, 428, 233, 6301, 295, 11, 173, 176642,
                      585, 147117, 19845, 14830, 6916, 18687, 4, 6646, 1272, 46, 141085, 64381,
                      63692, 11, 2156, 7806, 61, 5, 928, 15, 147387, 116331, 145634, 57186, 9307, 63, 34]
        """
        feat_sizes = CRITEO_FEATURE_NUM
        def fn_x(_line_):
            _line_ = _line_.split(',')[1:]
            return [hash(_line_[j])%feat_sizes[j] for j in range(len(_line_))]
        def fn_y(_line_):
            return int(_line_[0])
        with open(self.src_file_name) as _file_:
            eof = False
            while (not eof):
                data_x = [[0 for i in range(len(feat_sizes))] for j in range(self.batch_size)]
                data_y = [[0] for j in range(self.batch_size)]
                for i in range(self.rank * self.batch_size):
                    line = _file_.readline()
                lines = []
                for i in range(self.batch_size):
                    lines.append(_file_.readline())
                data_x = self.pool.map(fn_x, lines)
                data_y = self.pool.map(fn_y, lines)
                for i in range((self.size - self.rank - 1) * self.batch_size):
                    line = _file_.readline()
                yield np.array(data_x), np.array(data_y)

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        try:
            data_x, data_y = self.data_iter.__next__()
            data = mx.nd.array(data_x)
            label = mx.nd.array(data_y)
            self.cur_batch += 1
            re = mx.io.DataBatch([data], [label])
            return re
        except:
            raise StopIteration
