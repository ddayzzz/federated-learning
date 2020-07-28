import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from flearn.utils.model_utils import gen_batch
from .fedbase import BaseFedarated


class FedMetaBaseServer(BaseFedarated):

    def __init__(self, params, learner, dataset):
        print('Using Federated-Meta to Train')
        inner_opt = tf.train.GradientDescentOptimizer(params['lr'])
        #
        append = f'meta_algo{params["meta_algo"]}_outer_lr{params["outer_lr"]}'
        super(FedMetaBaseServer, self).__init__(params, learner, dataset, optimizer=inner_opt, append2metric=append)
        self.split_clients()
        self.train_support_batches, self.train_query_batches = self.generate_mini_batch_generator(self.train_clients)
        self.test_support_batches, self.test_query_batches = self.generate_mini_batch_generator(self.train_clients)
        if params['meta_algo'] == 'maml':
            self.impl = self._impl_maml
        else:
            raise NotImplementedError
        print('Using ', params['meta_algo'], "as implement of FedMeta")
        self.outer_lr = params['outer_lr']

    def split_clients(self):
        """
        拆分客户端
        :return:
        """
        train_rate = int(0.8 * self.num_clients)
        val_rate = int(0.1 * self.num_clients)
        test_rate = self.num_clients - train_rate - val_rate
        ind = np.random.permutation(self.num_clients)
        arryed_cls = np.asarray(self.clients)
        self.train_clients = arryed_cls[ind[:train_rate]]
        self.eval_clients = arryed_cls[ind[train_rate:train_rate + val_rate]]
        self.test_clients = arryed_cls[ind[train_rate + val_rate:]]

    def select_clients(self, round_i, num_clients=20):
        np.random.seed(round_i)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.train_clients)), num_clients, replace=False)
        return indices, self.train_clients[indices]

    def generate_mini_batch_generator(self, clients):
        train_batches = {}
        for c in clients:
            train_batches[c.id] = gen_batch(c.train_data, self.batch_size, self.num_rounds + 1)

        test_batches = {}
        for c in clients:
            test_batches[c.id] = gen_batch(c.eval_data, self.batch_size, self.num_rounds + 1)
        return train_batches, test_batches

    def aggregate_gd(self, weights_before, grads):
        """
        这里的 sols 定义为梯度
        :param wsolns:
        :return:
        """
        factor = self.outer_lr / len(grads)
        new_solutions = [(u - v) * 1.0 for u, v in zip(weights_before, grads)]
        return new_solutions

    def _impl_maml(self, clients):
        """
        FedMeta - MAML 的实现
        :param clients:
        :return:
        """
        grads = []  # 记录客户端运行的数据
        comps = []
        weight_before = []
        for c in clients:  # simply drop the slow devices
            # communicate the latest model
            c.set_params(self.latest_model)
            weight_before.append(c.get_params())
            support_batch = next(self.train_support_batches[c.id])
            query_batch = next(self.train_query_batches[c.id])
            grads1, loss1, weights1, comp1 = c.solve_sgd(support_batch)
            # 基于 query, 这时候网络的参数为 theta'
            grads2, loss2, weights2, comp2 = c.solve_sgd(query_batch)
            grads.append(grads2)
            comps.append(comp1 + comp2)
        return weight_before, grads, comps

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.start_round, self.num_rounds):
            # test model
            if (i + 1) % self.eval_every_round == 0:
                # stats = self.local_test_only_acc(round_i=i, on_train=False, sync_params=True)  # have set the latest model for all clients
                # stats_train = self.local_test_only_acc(round_i=i, on_train=True, sync_params=False)
                pass

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling

            weight_before, grads, comps = self.impl(selected_clients)

            # update models
            self.latest_model = self.aggregate_gd(weight_before, grads)

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
