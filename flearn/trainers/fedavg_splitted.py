import numpy as np
from tqdm import trange, tqdm
import time
import tensorflow as tf

from .fedbase import BaseFedarated


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        # 这个算法和 FedAvg 的区别就是客户端是分开的
        print('Using Federated avg splitted to Train')
        opt = tf.train.AdamOptimizer(params['lr'])
        self.num_fine_tune = params['meta_num_fine_tune']
        append = f'optimizer_adam_finetune{self.num_fine_tune}'
        super(Server, self).__init__(params, learner, dataset, optimizer=opt, append2metric=append)
        self.split_clients()

    def split_clients(self):
        """
        拆分客户端
        :return:
        """
        train_rate = int(0.8 * self.num_clients)
        val_rate = int(0.1 * self.num_clients)
        test_rate = self.num_clients - train_rate - val_rate

        assert test_rate > 0 and val_rate > 0 and test_rate > 0, '不能为空'

        ind = np.random.permutation(self.num_clients)
        arryed_cls = np.asarray(self.clients)
        self.train_clients = arryed_cls[ind[:train_rate]]
        self.eval_clients = arryed_cls[ind[train_rate:train_rate + val_rate]]
        self.test_clients = arryed_cls[ind[train_rate + val_rate:]]
        #
        print('用于测试的客户端数量{}, 用于验证:{}, 用于测试: {}'.format(len(self.train_clients), len(self.eval_clients), len(self.test_clients)))

    def local_test_only_acc(self, round_i, on_train=False, sync_params=True):
        """
        基于测试集的客户端
        :param round_i:
        :param on_train:
        :param sync_params:
        :return:
        """
        num_samples = []
        tot_correct = []
        tot_losses = []
        begin_time = time.time()
        for c in self.test_clients:
            if sync_params:
                self.client_model.set_params(self.latest_model)
            # correct, loss, ds = c.test(on_train=on_train)
            if self.num_fine_tune > 1:
                # 这里需要使用
                spt = c.train_data
                qry = c.eval_data

            else:
                correct, loss, ds = c.model.test_all_data_points(c.train_data, c.eval_data)
            tot_correct.append(correct)
            num_samples.append(ds)
            tot_losses.append(loss)
        end_time = time.time()
        # 计算平均的数据
        # 平均的准确率
        avg_correct = np.sum(tot_correct) * 1.0 / np.sum(num_samples)
        # 注意, 所有的 loss 都是 reduce_mean 的
        avg_loss = np.dot(tot_losses, num_samples) * 1.0 / np.sum(num_samples)
        stats = {'loss': avg_loss, 'acc': avg_correct, 'time': end_time - begin_time}
        # 始终不隐藏
        print('>>> On {} dataset: round: {} / acc: {:.3%} / '
              'loss: {:.4f} / Time: {:.2f}s'.format(
            'Train' if on_train else 'Test',
            round_i, stats['acc'],
            stats['loss'], stats['time']))
        if on_train:
            self.metrics.update_train_stats_only_acc_loss(round_i=round_i, train_stats=stats)
        else:
            self.metrics.update_eval_stats(round_i=round_i, eval_stats=stats)
        return stats

    def select_clients(self, round_i, num_clients=20):
        """
        使用随机采样
        :param round_i:
        :param num_clients:
        :return:
        """
        np.random.seed(round_i)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.train_clients)), num_clients, replace=False)
        return indices, self.train_clients[indices]

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.start_round, self.num_rounds):
            # test model
            if (i + 1) % self.eval_every_round == 0:
                stats = self.local_test_only_acc(round_i=i, on_train=False, sync_params=True)  # have set the latest model for all clients
                # stats_train = self.local_test_only_acc(round_i=i, on_train=True, sync_params=False)

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling

            csolns = []  # buffer for receiving client solutions
            # sstats = []  # 记录客户端运行的数据
            for idx, c in enumerate(selected_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally

                # soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size, round_i=i, hide_output=self.hide_client_output)
                # 这里的使用完整的数据集进行训练, 测试就是使用 test_client 的完整的数据
                soln, comp, sz = c.model.solve_all_data_points(c.train_data, c.eval_data)
                # gather solutions from client
                csolns.append((sz, soln))

                # sstats.append(stats)
                # track communication cost
                # self.metrics.update(rnd=i, cid=c.id, stats=stats)
            # 更新 计算量, 读取字节, 写入字节等信息
            # self.metrics.extend_commu_stats(round_i=i, stats_list=sstats)
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
