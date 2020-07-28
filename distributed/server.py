#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import uuid
import random
import io
import codecs
import numpy as np
import json
import time
import sys
import time

import logging

import tornado.web
import tornado.websocket
import tornado.ioloop
import tornado.options
from distributed.server.server_helper import ServerHelper


trainer_server = None  # 不同的算法会不同
# websocket多线程，每一个链接生成一个线程，WebSocketHandler运行在这个线程上


class MainHandler(tornado.websocket.WebSocketHandler):

    global trainer_server
    remote_address = "unknown"

    def check_origin(self, origin):
        return True

    def open(self):
        # self.remote_address = self.request.remote_ip
        self.remote_address = self
        logging.info("A client connected: {}".format(self.remote_address))
        self.handle_wake_up()

    def on_close(self):
        assert isinstance(trainer_server, ServerHelper)
        if self in trainer_server.online_clients:
            trainer_server.add_client(self)
            logging.info("A client disconnected: {}".format(self.remote_address))

    def on_message(self, message):
        self.message_handler(message)
        logging.info("received message from client: {}".format(self.remote_address))

    def message_handler(self, message):
        message = json.loads(message)  # received message is the format of str
        data_head = message["data_head"]
        if data_head == "client_ready":
            self.handle_client_ready(message["data_content"])
        elif data_head == "client_update":
            self.handle_client_update(message["data_content"])
        elif data_head == "client_eval":
            self.handle_client_eval(message["data_content"])
        else:
            logging.error("unsupported massage: {}", message)

    def handle_wake_up(self):
        assert isinstance(trainer_server, ServerHelper)
        print("client wake_up")
        weights = trainer_server.dump_weights()
        self.write_message(data_with_message('init', {
            'current_weight': weights,
            'num_epochs': trainer_server.num_epochs,
            'num_rounds': trainer_server.num_rounds,
            'batch_size': trainer_server.batch_size,
            'num_loader_worker': 2,
            'seed': 0,
        }))

    def handle_client_ready(self, data):
        assert isinstance(trainer_server, ServerHelper)
        print("client ready for training", data)
        trainer_server.add_client(self)
        if trainer_server.num_online_clients >= trainer_server.min_train_client and trainer_server.current_round == -1:
            # 初始化且满足一定客户端数量, 启动训练
            self.train_next_round()

    def handle_client_update(self, data):
        """
        处理客户端运行一次 update所做的工作
        :param data:
        :return:
        """
        assert isinstance(trainer_server, ServerHelper)
        print("received client update of bytes: ", sys.getsizeof(data))
        print("handle client_update", self)
        for x in data:
            if x != 'new_weight':
                print(x, data[x])

        # 确保是当前发送的结果
        if data['round_number'] == trainer_server.current_round:
            # 将接收的内容存入缓冲区
            trainer_server.received_train_stats_from_client_this_round.append(data)

            # 如果当前的缓冲区达到了最低训练的客户端的数量, 就进行聚合
            assert trainer_server.min_train_client == trainer_server.num_clients_per_round, '目前保持一致'
            if len(trainer_server.received_train_stats_from_client_this_round) >= trainer_server.min_train_client:
                # 相关的数据保存到了 received_train_stats_from_client_this_round
                # 聚合相关的参数
                trainer_server.aggregate()
                # 输出训练和测试的结果
                info = trainer_server.compute_train_stats()
                info.to_csv('round_at_{}.csv'.format(trainer_server.current_round))
                trainer_server.log_metrics(df=info)
                # 写入上一次运行的 loss, 以确保收敛
                # trainer_server.global_model.prev_train_loss = aggr_train_loss

                if trainer_server.current_round >= trainer_server.num_rounds:
                    self.stop_and_eval()
                else:
                    # 准备洗一个轮次
                    self.train_next_round()

    def handle_client_eval(self, data):
        assert isinstance(trainer_server, ServerHelper)
        if trainer_server.received_eval_stats_from_client_this_round is None:
            return
        print("handle client_eval", self)
        print("eval_resp", data)
        trainer_server.received_eval_stats_from_client_this_round += [data]

        # tolerate 30% unresponsive clients
        if len(trainer_server.received_eval_stats_from_client_this_round) > trainer_server.min_train_client:


            print("== done ==")
            trainer_server.received_eval_stats_from_client_this_round = None  # special value, forbid evaling again

    def train_next_round(self):
        """
        只有初始化(满足一定数量的客户端已经在线)或者运行下一轮次才会执行
        :return:
        """
        assert isinstance(trainer_server, ServerHelper)
        trainer_server.current_round += 1
        # buffers all client updates
        trainer_server.received_train_stats_from_client_this_round = []

        print("### Round ", trainer_server.current_round, "###")
        client_num = min(trainer_server.num_clients_per_round, trainer_server.min_train_client)
        client_sids_selected = random.sample(list(trainer_server.online_clients), client_num)
        print("request updates from", client_sids_selected)

        weights = trainer_server.dump_weights()
        # by default each client cnn is in its own "room"
        for client in client_sids_selected:
            client.write_message(data_with_message('request_update', {
                # 用来识别接收的护具是否准确
                'round_number': trainer_server.current_round,
                'num_epochs': trainer_server.num_epochs,
                'current_weight': weights,
                'run_validation': trainer_server.current_round % trainer_server.eval_every == 0,
                'run_test': trainer_server.current_round % trainer_server.test_every == 0
            }))
            print("have sent request updates to", client)

    def stop_and_eval(self):
        assert isinstance(trainer_server, ServerHelper)
        trainer_server.eval_client_updates = []
        for client in trainer_server.online_clients:
            client.write_message(data_with_message('stop_and_eval', {
                    'current_weight': trainer_server.dump_weights(),
                }))


def data_with_message(data_head, data_content):
    return {"data_head": data_head, "data_content": data_content}


def make_app():
    return tornado.web.Application([(r"/", MainHandler)], websocket_max_message_size=1024*1024*1024)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)

    parser.add_argument('--host', help='选择使用的算法', type=str, default='localhost')
    parser.add_argument('--port', help='选择使用的算法', type=int, default=10000)
    parser.add_argument('--algo', help='选择使用的算法', type=str, default='fedavg')
    parser.add_argument('--model', help='name of model;', type=str, default='mclr')
    parser.add_argument('--num_epochs', help='number of rounds to simulate;', type=int, default=10)
    parser.add_argument('--clients_per_round', help='number of rounds to simulate;', type=int, default=10)
    parser.add_argument('--min_clients', help='number of rounds to simulate;', type=int, default=3)
    parser.add_argument('--num_rounds',  help='number of rounds to simulate;', type=int, default=200)
    parser.add_argument('--save_every', help='save global model every ____ rounds;', type=int,  default=50)
    parser.add_argument('--eval_every', help='save global model every ____ rounds;', type=int, default=2)
    parser.add_argument('--test_every', help='save global model every ____ rounds;', type=int, default=1)
    parser.add_argument('--batch_size', help='batch size when clients train on data;', type=int, default=10)
    # parser.add_argument('--lr', help='learning rate for inner solver;', type=float, default=0.01)
    parser.add_argument('--seed', help='seed for randomness;', type=int, default=0)
    parser.add_argument('--quiet', help='hide client\'s output; only show eval results', action='store_true')

    return parser.parse_args()


def main():
    global trainer_server
    from distributed.server.server_helper import FedAvgHelper
    from distributed.run_client import get_model
    args = parse_args()
    options = args.__dict__
    #
    model, _ = get_model('brats2018', 'unet')
    trainer_server = FedAvgHelper(model, params=options)
    logging.basicConfig(level=logging.INFO)

    print('Running federated learning server on {}:{} '.format(options['host'], options['port']))
    app = make_app()
    app.listen(options['port'])
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()