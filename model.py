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

import mxnet as mx


def deep_cross_model(batch_size, num_embed_features, input_dims, hidden_units, cross_layer=3):
    label = mx.symbol.Variable("softmax_label", shape=(batch_size,))

    dns_data = mx.symbol.Variable("dns_data", shape=(batch_size, num_embed_features))
    # embedding features
    x = mx.symbol.slice(data=dns_data, begin=(0, 0),
                        end=(batch_size, num_embed_features))
                        #end=(None, num_embed_features))
    embeds = mx.symbol.split(data=x, num_outputs=num_embed_features, squeeze_axis=1)
    features = []
    for i, embed in enumerate(embeds):
        embed_weight = mx.symbol.Variable('embed_%d_weight' % i, stype='row_sparse')
        features.append(mx.symbol.sparse.Embedding(data=embed, weight=embed_weight,
                        input_dim=input_dims[i], output_dim=hidden_units[0], sparse_grad=True))
    concated_input = mx.symbol.concat(*features, dim=1)
    # cross model
    cross_hidden = concated_input
    weight_dim = hidden_units[0]*num_embed_features
    for i in range(cross_layer):
        weight = mx.symbol.Variable('cross_weight_%d'%i, shape=(weight_dim, 1), stype='default',
                init = mx.initializer.Normal(sigma=0.01))
        bias = mx.symbol.Variable('cross_bias_%d'%i, shape=(weight_dim), stype='default',
                init = mx.initializer.Normal(sigma=0.01))
        cross_hidden = mx.symbol.broadcast_add(mx.symbol.broadcast_mul(concated_input,
                            mx.symbol.dot(cross_hidden,
                            weight)), bias) + cross_hidden
    # deep model
    hidden = mx.symbol.FullyConnected(data=concated_input, num_hidden=hidden_units[1])
    hidden = mx.symbol.Activation(data=hidden, act_type='relu')
    for i in range(2, len(hidden_units)):
        hidden = mx.symbol.FullyConnected(data=hidden, num_hidden=hidden_units[i])
        hidden = mx.symbol.Activation(data=hidden, act_type='relu')
    cross_deep_hidden = mx.symbol.concat(hidden, cross_hidden, dim=1)
    cross_deep_hidden = mx.symbol.Activation(data=cross_deep_hidden, act_type='relu')
    cross_deep_out = mx.symbol.FullyConnected(data=cross_deep_hidden, num_hidden=2)

    out = mx.symbol.SoftmaxOutput(cross_deep_out, label, name='model')
    return out
