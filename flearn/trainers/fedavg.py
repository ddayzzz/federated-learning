import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        opt = tf.train.GradientDescentOptimizer(params['lr'])
        super(Server, self).__init__(params, learner, dataset, optimizer=opt)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.start_round, self.num_rounds):
            # test model
            if (i + 1) % self.eval_every_round == 0:
                stats = self.local_test_only_acc(round_i=i, on_train=False, sync_params=True)  # have set the latest model for all clients
                stats_train = self.local_test_only_acc(round_i=i, on_train=True, sync_params=False)

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling

            csolns = []  # buffer for receiving client solutions
            sstats = []  # 记录客户端运行的数据
            for idx, c in enumerate(selected_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally

                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size, round_i=i, hide_output=self.hide_client_output)

                # gather solutions from client
                csolns.append(soln)
                sstats.append(stats)
                # track communication cost
                # self.metrics.update(rnd=i, cid=c.id, stats=stats)
            # 更新 计算量, 读取字节, 写入字节等信息
            self.metrics.extend_commu_stats(round_i=i, stats_list=sstats)
            # update models
            self.latest_model = self.aggregate(csolns)

            if (i + 1) % self.save_every_round == 0:
                self.save_model(i)
                self.metrics.write()

        # final test model
        # stats = self.local_test_only_acc(round_i=self.num_rounds, on_train=False,
        #                                  sync_params=True)  # have set the latest model for all clients
        # stats_train = self.local_test_only_acc(round_i=self.num_rounds, on_train=True, sync_params=False)
        self.eval_to_file(round_i=self.num_epochs, sync_params=True)
        self.metrics.write()
        self.save_model(self.num_rounds)
