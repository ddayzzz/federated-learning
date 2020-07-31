#!/usr/bin/env bash

cd sent140

if [ ! -f 'glove.6B.300d.txt' ]; then
    wget https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip
    unzip glove.6B.zip
    rm glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt glove.6B.zip
fi

if [ ! -f embs.json ]; then
    python3 get_embs.py
fi