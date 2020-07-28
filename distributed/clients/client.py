# -*- coding: utf-8 -*-

import numpy as np
import pickle
import codecs

import logging

import time
import json
import torch
from tornado import ioloop
import tornado.iostream

from distributed.clients.base_client import WebSocketClient
from distributed.clients.workers import BaseWorker
from distributed.utils.data_model_loader import DataModel


APPLICATION_JSON = 'application/json'

DEFAULT_CONNECT_TIMEOUT = 60
DEFAULT_REQUEST_TIMEOUT = 60


class FederatedClient(WebSocketClient):
    MAX_DATASET_SIZE_KEPT = 1000

    # logging.basicConfig(level=logging.DEBUG)

    def __init__(self, datamodel: DataModel, io_loop=None,
                 connect_timeout=DEFAULT_CONNECT_TIMEOUT,
                 request_timeout=DEFAULT_REQUEST_TIMEOUT):

        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self._io_loop = io_loop or ioloop.IOLoop.current()
        self.ws_url = None
        self.auto_reconnect = False
        self.last_active_time = 0
        self.datamodel = datamodel

        super(FederatedClient, self).__init__(self.connect_timeout, self.request_timeout)

    def connect(self, url, auto_reconnet=True, reconnet_interval=10):
        self.ws_url = url
        self.auto_reconnect = auto_reconnet
        self.reconnect_interval = reconnet_interval

        super(FederatedClient, self).connect(self.ws_url)

    def send(self, msg):
        super(FederatedClient, self).send(msg)
        self.last_active_time = time.time()

    def emit(self, msg_head, msg_body):
        self.send({"data_head": msg_head, "data_content": msg_body})

    def _on_message(self, msg):
        self.message_handler(msg)
        self.last_active_time = time.time()

    def _on_connection_success(self):

        # self.send("client_wake_up")
        self.last_active_time = time.time()

    def _on_connection_close(self, reason="unknown"):
        print('Connection closed reason=%s' % (reason,))
        self.reconnect()

    def reconnect(self):
        print('reconnect')
        if not self.is_connected() and self.auto_reconnect:
            self._io_loop.call_later(self.reconnect_interval,
                                     super(FederatedClient, self).connect, self.ws_url)

    def message_handler(self, message):
        message = json.loads(message)  # received message is the format of str
        data_head = message["data_head"]
        if data_head == "init":
            self.on_init(message["data_content"])
        elif data_head == "request_update":
            self.on_request_update(message["data_content"])
        elif data_head == "stop_and_eval":
            self.on_stop_and_eval(message["data_content"])
        else:
            logging.error("unsupported massage: {}", message)
            print("unsupported massage!")

    def on_init(self, *args):
        """
        初始化模型: 定义模型
        :param args:
        :return:
        """
        model_config = args[0]
        """
        model_config = {
            current_weight: bytes,
            num_epochs: int,
            num_rounds: int,
            batch_size: int,
            num_workers: int
            lr: float,
            seed: int,
        }
        """
        # print('on init', model_config)
        print('preparing local data based on server model_config')
        # ([(Xi, Yi)], [], []) = train, test, valid
        # 加载模型
        torch.cuda.set_device(self.datamodel.worker.device)
        self.datamodel.init(model_config)
        # 加载模型参数
        self.datamodel.worker.set_model_params(model_config['current_weight'])
        # ready to be dispatched for training
        print("sent client_ready: " + self.datamodel.worker.client_id)
        self.emit('client_ready', {
                'num_train_data': len(self.datamodel.train_dataloader.dataset),
                'num_test_data': len(self.datamodel.test_dataloader.dataset)
            })

    def on_request_update(self, *args):
        req = args[0]
        print(self.datamodel.worker.client_id + " update requested")

        # if req['weights_format'] == 'pickle':
        #     weights = pickle_string_to_obj(req['current_weights'])
        weights = req['current_weight']  # BytesIO
        run_test = req['run_test']
        run_val = req['run_validation']
        # 加载对应的权重
        self.datamodel.worker.set_model_params(weights)
        new_weights, stats = self.datamodel.train_one_round(num_epoch=req['num_epochs'], round_i=req['round_number'])
        data = {
            'train': stats,
            'new_weight': new_weights,
            'round_number': req['round_number'],
            'client_id': self.datamodel.worker.client_id

        }
        # 接着进行验证和测试
        if run_test:
            test_stats = self.datamodel.run_metric_on_test()
            data['test'] = test_stats
        if run_val and self.datamodel.use_eval:
            val_stats = self.datamodel.run_metric_on_val()
            data['val'] = val_stats
        self.emit('client_update', data)

    def on_stop_and_eval(self, *args):
        req = args[0]
        weights = req['current_weight']  # BytesIO
        # 加载对应的权重
        self.datamodel.worker.set_model_params(req['current_weight'])
        # 接着进行验证和测试
        test_stats = self.datamodel.run_metric_on_test()
        self.emit('client_eval', {'test': test_stats})


def data_with_message(data_head, data_content):
    return {"data_head": data_head, "data_content": data_content}


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)


def main():
    logging.basicConfig(level=logging.INFO)
    client = FederatedClient("127.0.0.1", 11112, datasource.Tumor(data_train_path, data_test_path, dataseed=12))
    ws_url = 'ws://127.0.0.1:11112/'
    client.connect(ws_url, auto_reconnet=True, reconnet_interval=10)

    try:
        ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        client.close()


if __name__ == '__main__':
    main()
