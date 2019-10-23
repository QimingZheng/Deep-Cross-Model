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
from mxnet import kv
from mxnet.test_utils import *
from config import *
from data import get_synthesized_data, CriteoIterator
from model import deep_cross_model
import argparse
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description="Run sparse wide and deep classification ",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epoch', type=int, default=2,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=2048,
                    help='number of examples per batch')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--gpu-num', type=int, default=1,
                    help='gpu num')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='what optimizer to use',
                    choices=["ftrl", "sgd", "adam"])
parser.add_argument('--synthesize', type=int, default=0,
                    help="use synthesize data or not")
parser.add_argument('--log-interval', type=int, default=1,
                    help='number of batches to wait before logging training status')


if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    store = kv.create('local_allreduce_device')
    # arg parser
    args = parser.parse_args()
    logging.info(args)
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    optimizer = args.optimizer
    log_interval = args.log_interval
    lr = args.lr
    ctx = [mx.gpu(i) for i in range(args.gpu_num)]

    # synthesized dataset
    if args.synthesize:
        data_dir = os.path.join(os.getcwd(), 'data')
        train_data = os.path.join(data_dir, ADULT['train'])
        val_data = os.path.join(data_dir, ADULT['test'])
        train_dns, train_label = get_synthesized_data(data_dir, ADULT['train'])
        val_dns, val_label = get_synthesized_data(data_dir, ADULT['test'])

        model = deep_cross_model(batch_size//args.gpu_num, ADULT['num_embed_features'], ADULT['embed_input_dims'], ADULT['hidden_units'])
        mx.visualization.print_summary(model)
        # data iterator
        train_data = mx.io.NDArrayIter({'dns_data': train_dns},
                                        {'softmax_label': train_label}, batch_size,
                                        shuffle=True, last_batch_handle='discard')
        eval_data = mx.io.NDArrayIter({'dns_data': val_dns},
                                        {'softmax_label': val_label}, batch_size,
                                        shuffle=True, last_batch_handle='discard')
    else:
        """
        train_data = CriteoIterator('dns_data', 'softmax_label', batch_size=batch_size,
                                        src_file_name='data/tiny.train.csv', in_mem=True)
        eval_data = CriteoIterator('dns_data', 'softmax_label', batch_size=batch_size,
                                        src_file_name='data/test.csv', in_mem=True)
        """
        train_data = mx.io.CSVIter(data_csv="data/train.csv.data.csv", data_shape= (CRITEO_FIELD_NUM,), \
                                   label_csv="data/train.csv.label.csv", label_shape = (1,), \
                                   batch_size=batch_size, round_batch=False,\
                                   prefetching_buffer=4)
        eval_data = mx.io.CSVIter(data_csv="data/test.csv.data.csv", data_shape= (CRITEO_FIELD_NUM,), \
                                   label_csv="data/test.csv.label.csv", label_shape = (1,), \
                                   batch_size=batch_size, round_batch=False,\
                                   prefetching_buffer=4)
        model = deep_cross_model(batch_size//args.gpu_num, CRITEO_FIELD_NUM, CRITEO_FEATURE_NUM, ADULT['hidden_units'])
    # module
    mod = mx.mod.Module(symbol=model, context=ctx, data_names=['dns_data'],
                        label_names=['softmax_label'])

    #mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.bind(data_shapes=[('dns_data', (batch_size, CRITEO_FIELD_NUM))], \
            label_shapes=[('softmax_label', (batch_size,))])
    mod.init_params()
    optim = mx.optimizer.create(optimizer, learning_rate=lr, rescale_grad=1.0/batch_size, lazy_update=False)
    mod.init_optimizer(kvstore=store, optimizer=optim)
    # use accuracy as the metric
    metric = mx.metric.create(['acc'])
    # get the sparse weight parameter
    speedometer = mx.callback.Speedometer(batch_size, log_interval)

    logging.info('Training started ...')

    data_iter = iter(train_data)
    for epoch in range(num_epoch):
        nbatch = 0
        metric.reset()
        start = time.time()
        for batch in data_iter:
            nbatch += 1
            mod.forward_backward(batch)
            # update all parameters (including the weight parameter)
            mod.update()
            # update training metric
            mod.update_metric(metric, batch.label)
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=metric, locals=locals())
            speedometer(speedometer_param)
        # evaluate metric on validation dataset
        elapsed = time.time() - start
        logging.info("Epoch [%d]: %f samples / sec"%(epoch, nbatch * batch_size / elapsed ))
        score = mod.score(eval_data, ['acc'])
        logging.info('epoch %d, accuracy = %s' % (epoch, score[0][1]))

        # mod.save_checkpoint("checkpoint/checkpoint", epoch, save_optimizer_states=True)
        # reset the iterator for next pass of data
        data_iter.reset()

    logging.info('Training completed.')
