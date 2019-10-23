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

# Related to feature engineering, please see preprocess in data.py

## Parameters used to synthesize artifical data

TRAINING_SET_SIZE = 2**17
EVAL_SET_SIZE = 2**10

# must dividable by 4
EMBEDDING_FIELD_NUM = 32 # 40 # 48

EMBEDDING_FEATURE_NUM = [100 for i in range(EMBEDDING_FIELD_NUM//4)] + \
        [10000 for i in range(EMBEDDING_FIELD_NUM//4)] + \
        [100000 for i in range(EMBEDDING_FIELD_NUM//4)] + \
        [1000000 for i in range(EMBEDDING_FIELD_NUM//4)] # 100000000

ADULT = {
    'train': 'train',
    'test': 'test',
    'train_size': TRAINING_SET_SIZE,
    'test_size': EVAL_SET_SIZE,
    'num_embed_features': EMBEDDING_FIELD_NUM,
    'embed_input_dims': EMBEDDING_FEATURE_NUM,
    'hidden_units': [80, 1024, 512, 256, 128], # embedding dim = 80 or 96
}

## Criteo parameters

CRITEO_FIELD_NUM = 39
CRITEO_FEATURE_NUM=[4389, 8000, 329, 7432, 2646, 428, 233, 6301, 295, 11, 173, 176642,\
        585, 147117, 19845, 14830, 6916, 18687, 4, 6646, 1272, 46, 141085, 64381,\
        63692, 11, 2156, 7806, 61, 5, 928, 15, 147387, 116331, 145634, 57186, 9307, 63, 34]

