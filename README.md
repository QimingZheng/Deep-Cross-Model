# Deep-Cross-Model
Deep & Cross model on Mxnet

## Dataset preparation

Criteo dataset: 
```
cd data
wget https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
tar -xzvf dac.tar.gz
python3 txt2csv.py train.txt
python3 txt2csv.py test.txt
python3 preprocessing.py train.csv
python preprocessing.py test.csv
```
Usually, you only want a (small) subset of the training dataset in training-throughput benchmarks.

## Train with multi-gpus

```
python3 train.py train.py \
    --gpu-num ${YOUR_DEVICE_NUM} \
    --batch-size ${YOUR_BATCH_SIZE} \
    --num-epoch ${EPOCH_NUM}
```


