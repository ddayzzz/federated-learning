import numpy as np
import tqdm
import time
import tensorflow as tf

from flearn.utils.model_utils import gen_batch
from .fedbase import BaseFedarated


class FedMetaBaseServer(BaseFedarated):

    def __init__(self, params, learner, dataset):
        print('Using Federated-Meta to Train')
        self.meta_algo = params["meta_algo"]
        self.num_fine_tune = params['meta_num_fine_tune']
        if self.meta_algo == 'meta_sgd':
            inner_opt = tf.train.AdamOptimizer(1.0)
        else:
            inner_opt = tf.train.AdamOptimizer(1.0)
        #
        append = f'meta_algo[{self.meta_algo}]_outer_lr{params["outer_lr"]}_finetune{self.num_fine_tune}'
        super(FedMetaBaseServer, self).__init__(params, learner, dataset, optimizer=inner_opt, append2metric=append)
        self.split_clients()
        # 对于所有的客户端均生成 generator

        self.train_support_batches, self.train_query_batches = self.generate_mini_batch_generator(self.train_clients)
        self.test_support_batches, self.test_query_batches = self.generate_mini_batch_generator(self.test_clients, self.num_fine_tune)
        if self.meta_algo in ['maml', 'meta_sgd']:
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
                c.set_params(self.latest_model)
            # 是不是需要新运行一遍 query
            # for _ in range(self.num_fine_tune):
            #     support_batch = next(self.test_support_batches[c.id])
            #     c.solve_sgd(support_batch)
            # 不需要结果
            # c.model.solve_inner(data=c.train_data, client_id=c.id, round_i=round_i, num_epochs=1, batch_size=self.batch_size, hide_output=True)
            # all_x, all_y = np.concatenate((c.train_data['x'], c.eval_data['x']), axis=0), np.concatenate((c.train_data['y'], c.eval_data['y']), axis=0)
            # correct, loss = c.model.test((all_x, all_y))
            # ds = len(all_y)
            # 这里的参数已经更新
            # correct, loss, ds = c.test(on_train=False)
            support_batch = next(self.test_support_batches[c.id])
            query_batch = next(self.test_query_batches[c.id])
            correct, loss = c.model.test_meta(support_batch, query_batch)
            ds = len(query_batch)
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

    def generate_mini_batch_generator(self, clients, num_fine_tune=1):
        train_batches = {}
        for c in clients:
            train_batches[c.id] = gen_batch(c.train_data, self.batch_size, (self.num_rounds + 1) * num_fine_tune)

        test_batches = {}
        for c in clients:
            test_batches[c.id] = gen_batch(c.eval_data, self.batch_size, (self.num_rounds + 1) * num_fine_tune)
        return train_batches, test_batches

    def aggregate_gd(self, weights_before, grads, train_size):
        """
        这里的 sols 定义为梯度
        :param wsolns:
        :return:
        """
        # m = len(grads)
        # g = []
        # for i in range(len(grads[0])):
        #     # i 表示的当前的梯度的 index
        #     # 总是 client 1 的梯度的形状
        #     grad_sum = np.zeros_like(grads[0][i])
        #     num_sz = 0
        #     for ic in range(len(grads)):
        #         grad_sum += grads[ic][i] * train_size[ic]
        #         num_sz += train_size[ic]
        #     grad_sum /= num_sz
        #     # 累加之后, 进行梯度下降
        #     g.append(grad_sum)
        # return [u - (v * self.outer_lr) for u, v in zip(weights_before, g)]
        m = len(grads)
        g = []
        for i in range(len(grads[0])):
            # i 表示的当前的梯度的 index
            # 总是 client 1 的梯度的形状
            grad_sum = np.zeros_like(grads[0][i])
            num_sz = 0
            for ic in range(len(grads)):
                grad_sum += grads[ic][i]
                num_sz += train_size[ic]
            # grad_sum /= num_sz
            # 累加之后, 进行梯度下降
            g.append(grad_sum)
        return [u - (v * self.outer_lr / m) for u, v in zip(weights_before, g)]

    def _impl_maml(self, clients, round_i):
        """
        FedMetaMAML
        :param clients:
        :return:
        """
        grads = []  # 记录客户端运行的数据
        comps = []
        weight_before = clients[0].get_params()
        train_size = []
        for c in clients:  # simply drop the slow devices
            # communicate the latest model
            c.set_params(self.latest_model)
            support_batch = next(self.train_support_batches[c.id])
            query_batch = next(self.train_query_batches[c.id])
            # grads1, loss1, weights1, comp1 = c.model.solve_sgd_meta(support_batch)
            # 基于 query, 这时候网络的参数为 theta'
            grads2, loss2, weights2, comp2 = c.model.solve_sgd_meta(support_batch, query_batch)
            # _, comp1 = c.model.solve_inner(data=c.train_data, client_id=c.id, round_i=round_i, num_epochs=1, batch_size=self.batch_size, hide_output=self.hide_client_output)
            # client_grads, comp2 = c.model.solve_inner_support_query(data=c.eval_data, client_id=c.id, round_i=round_i, num_epochs=1,
            #                            batch_size=self.batch_size, hide_output=self.hide_client_output)
            grads.append(grads2)
            comps.append(comp2)
            train_size.append(c.num_train_samples + c.num_test_samples)
        return weight_before, grads, comps, train_size

    def _impl_meta_sgd(self, clients):
        """
        FedMetaSGD
        :param clients:
        :return:
        """
        # 与 FedMetaMAML的却别主要是引入了参数向量
        pass

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.start_round, self.num_rounds):
            # test model
            if (i + 1) % self.eval_every_round == 0:
                stats = self.local_test_only_acc(round_i=i, on_train=False, sync_params=True)  # have set the latest model for all clients
                # 接下来再运行必须重新设置网络的参数
                # stats_train = self.local_test_only_acc(round_i=i, on_train=True, sync_params=False)

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling

            weight_before, grads, comps, train_size = self.impl(selected_clients, i)

            # update models
            self.latest_model = self.aggregate_gd(weight_before, grads, train_size)

            if (i + 1) % self.save_every_round == 0:
                self.save_model(i)
                self.metrics.write()

        # final test model
        stats = self.local_test_only_acc(round_i=self.num_rounds, on_train=False,
                                         sync_params=True)  # have set the latest model for all clients
        # stats_train = self.local_test_only_acc(round_i=self.num_rounds, on_train=True, sync_params=False)
        self.eval_to_file(round_i=self.num_epochs, sync_params=True)
        self.metrics.write()
        self.save_model(self.num_rounds)
