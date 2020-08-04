import torch
import time
import pandas as pd
import numpy as np
from flmod.solvers.fedbase_adv import BaseFedaratedAdvanced, Adam
from flmod.clients.base_client import BaseClient
from flmod.models.base_models import ModelWithMetaLearn


class Client(BaseClient):

    def __init__(self, id, train_dataset, test_dataset, options, optimizer, model: ModelWithMetaLearn):
        self.meta_inner_step = options['meta_inner_step']
        super(Client, self).__init__(id, train_dataset, test_dataset, options, optimizer, model)
        # 定义客户端执行的操作的

    def solve_epochs(self, round_i, record_grads=None, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs
        # acc, loss, comp
        stats = self.model.solve_meta_one_epoch(round_i=round_i, client_id=self.id, support_data_loader=self.train_dataset_loader, query_data_loader=self.test_dataset_loader)
        bytes_w = self.model.model_bytes
        bytes_r = self.model.model_bytes
        flop_stats = {'id': self.id, 'bytes_w': bytes_w, 'comp': stats['comp'], 'bytes_r': bytes_r}
        return stats, flop_stats, stats['grads']

    def mini_batch_generator(self, dataloader, mini_batch_gen_size):
        """
        产生基于 dataloader 的若干的 mini-batch
        :param dataloader:
        :param mini_batch_gen_size: 产生多少次?
        :return:
        """
        try:
            for x, y in dataloader:
                pass
        except StopIteration:
            pass


class FedMeta(BaseFedaratedAdvanced):

    def __init__(self, options, all_data_info, model_obj):
        self.meta_algo = options['meta_algo']
        self.outer_lr = options['outer_lr']
        self.outer_opt = Adam(lr=self.outer_lr)
        print('>>> Using FedMeta')
        a = f'outerlr_{self.outer_lr}_metaalgo_{self.meta_algo}'
        super(FedMeta, self).__init__(options=options, model=model_obj, read_dataset=all_data_info, append2metric=a, client_class=Client,
                                      more_metric_to_train=['query_acc', 'query_loss'])
        # 拆分客户端
        self.split_train_validation_test_clients()
        # self.train_support, self.train_query = self.generate_batch_generator(self.train_clients)

    def select_clients(self, round_i, num_clients=20):
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round_i)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.train_clients, num_clients, replace=False).tolist()

    def generate_batch_generator(self, client):
        tr, te = dict(), dict()
        for c in client:
            tr[c.id] = c.train_dataset_loader.__iter__()
            te[c.id] = c.test_dataset_loader.__iter__()
        return tr, te

    def eval_on(self, round_i, clients, use_test_data=True, use_train_data=True, use_val_data=False):
        # 设置写入的数据
        df = pd.DataFrame(columns=['client_id', 'spt_acc', 'spt_loss', 'spt_size', 'qry_size', 'qry_acc', 'qry_loss'])
        # 设置网络
        self.model.set_parameters_list(self.latest_model)

        spt_corr, spt_loss, spt_sz = 0, 0.0, 0
        qry_corr, qry_loss, qry_sz = 0, 0.0, 0
        for c in clients:
            support_loss, support_correct, support_num_sample, query_loss, query_correct, query_num_sample = self.model.test_meta_one_epoch(c.train_dataset_loader, c.test_dataset_loader)
            spt_corr += support_correct
            spt_loss += support_loss
            spt_sz += support_num_sample
            qry_sz += query_num_sample
            qry_corr += query_correct
            qry_loss += query_loss
            #
            df = df.append({'client_id': c.id,
                            'spt_acc': support_correct / support_num_sample,
                            'spt_loss': support_loss / support_num_sample,
                            'qry_acc': query_correct / query_num_sample,
                            'qry_loss': query_loss / query_num_sample,
                            'qry_size': query_num_sample,
                            'spt_size': support_num_sample,}, ignore_index=True)

        fn = 'eval_at_round_{}.csv'.format(round_i)
        mean_spt_loss, mean_qry_loss = spt_loss / spt_sz, qry_loss / qry_sz
        mean_spt_acc, mean_qry_acc = spt_corr / spt_sz, qry_corr / qry_sz
        if not self.quiet:
            print(f'Round {round_i}, eval on meta-test client mean spt loss: {mean_spt_loss:.5f}, mean spt acc: {mean_spt_acc:.3%}', end='; ')
            print(f'mean qry loss: {mean_qry_loss:.5f}, mean qry acc: {mean_qry_acc:.3%}')
        self.metrics.update_eval_stats(round_i, df=df, on_which='meta-test', filename=fn, other_to_logger={
            'spt_loss': mean_spt_loss, 'spt_acc': mean_spt_acc, 'qry_acc': mean_qry_acc, 'qry_loss': mean_qry_loss
        })

    def solve_epochs(self, round_i, clients, epoch=None):
        spt_corrects = 0
        spt_loss = 0.0
        qry_loss = 0.0
        qry_corrects = 0
        qry_sz, spt_sz = 0, 0

        solns = []
        num_qry_size = []
        for c in clients:
            c.set_parameters_list(self.latest_model)
            # 保存信息
            stat, flop_stat, grads = c.solve_epochs(round_i=round_i, num_epochs=1)
            # 总共正确的个数
            spt_corrects += stat['support_correct']
            qry_corrects += stat['query_correct']
            # loss 和
            spt_loss += stat['support_loss_sum']
            qry_loss += stat['query_loss_sum']
            #
            spt_sz += stat['support_num_samples']
            qry_sz += stat['query_num_samples']
            num_qry_size.append(stat['query_num_samples'])
            solns.append(grads)
            # 写入测试的相关信息
            self.metrics.update_commu_stats(round_i, flop_stat)

        mean_spt_loss, mean_qry_loss = spt_loss / spt_sz, qry_loss / qry_sz
        mean_spt_acc, mean_qry_acc = spt_corrects / spt_sz, qry_corrects / qry_sz

        stats = {
            'acc': mean_spt_acc, 'loss': mean_spt_loss,
            'query_acc': mean_qry_acc, 'query_loss': mean_qry_loss
        }
        if not self.quiet:
            print(f'Round {round_i}, meta-train, mean spt loss: {mean_spt_loss:.5f}, mean spt acc: {mean_spt_acc:.3%}', end='; ')
            print(f'mean qry loss: {mean_qry_loss:.5f}, mean qry acc: {mean_qry_acc:.3%}')

        self.metrics.update_train_stats_only_acc_loss(round_i, stats)
        return solns, num_qry_size

    def aggregate_grads_simple(self, solns, lr, weights_before):
        # 使用 adam
        m = len(solns)
        g = []
        for i in range(len(solns[0])):
            # i 表示的当前的梯度的 index
            # 总是 client 1 的梯度的形状
            grad_sum = torch.zeros_like(solns[0][i])
            for ic in range(m):
                grad_sum += solns[ic][i]
                # 累加之后, 进行梯度下降
            g.append(grad_sum)
        # 普通的梯度下降 [u - (v * self.outer_lr / m) for u, v in zip(weights_before, g)]
        self.outer_opt.increase_n()
        for i in range(len(weights_before)):
            # 这是一个 in-place 的函数
            self.outer_opt(weights_before[i], g[i] / m, i=i)

    def aggregate_grads_weighted(self, solns, num_samples, weights_before):
        # 使用 adam
        m = len(solns)
        g = []
        for i in range(len(solns[0])):
            # i 表示的当前的梯度的 index
            # 总是 client 1 的梯度的形状
            grad_sum = torch.zeros_like(solns[0][i])
            total_sz = 0
            for ic, sz in enumerate(num_samples):
                grad_sum += solns[ic][i] * sz
                total_sz += sz
                # 累加之后, 进行梯度下降
            g.append(grad_sum / total_sz)
        # 普通的梯度下降 [u - (v * self.outer_lr / m) for u, v in zip(weights_before, g)]
        self.outer_opt.increase_n()
        for i in range(len(weights_before)):
            # 这是一个 in-place 的函数
            self.outer_opt(weights_before[i], g[i], i=i)

    def aggregate(self, solns, weight_before, num_qry_samples):
        # self.aggregate_grads_simple(solns=solns, weights_before=weight_before, lr=None)
        self.aggregate_grads_weighted(solns=solns, weights_before=weight_before, num_samples=num_qry_samples)

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')

            selected_client_indices = self.select_clients(round_i=round_i, num_clients=self.clients_per_round)
            weight_before = self.latest_model
            solns, qry_num = self.solve_epochs(round_i=round_i, clients=selected_client_indices)

            self.aggregate(solns, weight_before=weight_before, num_qry_samples=qry_num)
            # eval on test
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.test_clients)

            if (round_i + 1) % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()
        self.metrics.write()
