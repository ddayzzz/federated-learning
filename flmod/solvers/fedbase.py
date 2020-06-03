import numpy as np
import os
import time
import abc
import torch
from flmod.clients.base_client import BaseClient
from flmod.utils.data_utils import DatasetSplit
from flmod.utils.metrics import Metrics
from flmod.models.workers import choose_worker


class BaseFedarated(object):

    def __init__(self, options, model, dataset, optimizer, criterion, worker=None, append2metric=None):
        """
        定义联邦学习的基本的服务器, 这里的模型是在所有的客户端之间共享使用
        :param options: 参数配置
        :param model: 模型
        :param dataset: 数据集参数
        :param optimizer: 优化器
        :param criterion: 损失函数类型(交叉熵,Dice系数等等)
        :param worker: Worker 实例
        :param append2metric: 自定义metric
        """
        # if worker is None:
        #     from flmod.models.workers import Worker
        #     worker = Worker(model=model, criterion=criterion, optimizer=optimizer, options=options)
        worker_class = choose_worker(options)
        self.worker = worker_class(model=model, criterion=criterion, optimizer=optimizer, options=options)
        self.device = options['device']
        # 记录总共的训练数据
        self.num_train_data = 0
        self.options = options
        self.clients = self.setup_clients(dataset=dataset,
                                          criterion=criterion,
                                          worker=worker,
                                          batch_size=options['batch_size'])
        self.num_epochs = options['num_epochs']
        self.num_rounds = options['num_rounds']
        self.clients_per_round = options['clients_per_round']
        self.save_every_round = options['save_every']
        self.eval_on_test_every_round = options['eval_every']
        self.eval_on_train_every_round = options['eval_train_every']
        self.num_clients = len(self.clients)
        self.latest_model = self.worker.get_flat_model_params().detach()
        self.name = '_'.join(['', f'wn{options["clients_per_round"]}', f'tn{self.num_clients}'])
        self.metrics = Metrics(clients=self.clients, options=options, name=self.name, append2suffix=append2metric)
        self.print_result = True


    def setup_clients(self, dataset, criterion, worker, batch_size):
        users, groups, train_data, test_data, entire_train_dataset, entire_test_dataset, dataset_cfg = dataset
        # 配置相关的参数
        dataset_wrapper = dataset_cfg.get('dataset_wrapper')
        # if dataset_wrapper is None:
        #     from flmod.utils.worker_utils import MiniDataset
        #     dataset_wrapper = MiniDataset

        if entire_test_dataset is None and entire_train_dataset is not None:
            # train  和 test 的数据是合并的
            if len(groups) == 0:
                groups = [None for _ in users]
            all_clients = []
            for user, group in zip(users, groups):
                # if isinstance(user, str) and len(user) >= 5:
                #     user_id = int(user[-5:])
                # else:
                #     user_id = int(user)
                self.num_train_data += len(train_data[user]['x_index'])
                # c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
                c = BaseClient(id=user, worker=self.worker, batch_size=batch_size, criterion=criterion,
                               train_dataset=DatasetSplit(entire_train_dataset, idxs=train_data[user]['x_index']),
                               test_dataset=DatasetSplit(entire_train_dataset, test_data[user]['x_index']))
                all_clients.append(c)
            return all_clients
        elif entire_test_dataset is None and entire_train_dataset is None:
            # 没有 index 的数据, train 和 test 包含了所有的数据. train_data[user] 是一个 dict, 可以是 {x, y} 也可以是 {x_index,y_index}
            if len(groups) == 0:
                groups = [None for _ in users]

            all_clients = []
            for user, group in zip(users, groups):
                # if isinstance(user, str) and len(user) >= 5:
                #     user_id = int(user[-5:])
                # else:
                #     user_id = int(user)
                tr = dataset_wrapper(train_data[user], options=self.options)
                te = dataset_wrapper(test_data[user], options=self.options)
                self.num_train_data += len(tr)
                c = BaseClient(id=user, worker=self.worker, batch_size=batch_size, criterion=criterion,
                               train_dataset=tr,
                               test_dataset=te)
                all_clients.append(c)
            return all_clients
        else:
            # 采用 index 的数据
            if len(groups) == 0:
                groups = [None for _ in users]

            all_clients = []
            for user, group in zip(users, groups):
                # if isinstance(user, str) and len(user) >= 5:
                #     user_id = int(user[-5:])
                # else:
                #     user_id = int(user)
                self.num_train_data += len(train_data[user]['x_index'])
                # c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
                c = BaseClient(id=user, worker=self.worker, batch_size=batch_size, criterion=criterion,
                               train_dataset=DatasetSplit(entire_train_dataset, idxs=train_data[user]['x_index']),
                               test_dataset=DatasetSplit(entire_test_dataset, test_data[user]['x_index']))
                all_clients.append(c)
            return all_clients

    def local_train(self, selected_clients, round_i):
        solns = []  # 从客户端中接收的参数
        stats = []  # 通信代价 stats 信息
        for i, c in enumerate(selected_clients, start=1):
            # 把最新的模型广播出去
            c.set_flat_model_params(self.latest_model)

            # 运行训练
            soln, stat = c.local_train(round_i, self.num_epochs)
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    def select_clients(self, round, num_clients=20):
        """
        选择客户端, 采用的均匀的无放回采样
        :param round:
        :param num_clients:
        :return:
        """
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def aggregate(self, solns, **kwargs):
        """
        聚合模型
        :param solns: 列表. [(样本数量, 扁平化的参数)]
        :param kwargs:
        :return: 聚合后的参数
        """
        averaged_solution = torch.zeros_like(self.latest_model)
        num_all_samples = 0
        for num_sample, local_solution in solns:
            num_all_samples += num_sample
            averaged_solution += local_solution * num_sample # 加和, 乘以对应客户端的样本数量
        averaged_solution /= num_all_samples  # 除以运行样本的整体数量
        return averaged_solution.detach()

    def test_latest_model_on_traindata(self, round_i):
        """
        在训练数据集上测试
        :param round_i:
        :return:
        """
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        # 记录梯度用
        # flatten后的模型长度
        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        # 计算公式为 (客户端模型 - 上一次聚合后的模型) ^ 2, 一定程度上, 上一次聚合后的模型为平均的模型, 解释为方差
        difference = 0.
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference /= len(self.clients)
        stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
                  ' Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_train_data['acc'], stats_from_train_data['loss'],
                   stats_from_train_data['gradnorm'], difference, end_time-begin_time))
        return global_grads

    def calc_client_grads(self, round_i):
        """
        仅仅计算客户端梯度信息, 不在完整的训练集上运行测试
        :param round_i:
        :return:
        """
        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []
        start_time = time.time()
        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        gradnorm = np.linalg.norm(global_grads)


        difference = 0.
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference /= len(self.clients)

        end_time = time.time()
        # 设置stats
        stats = {'gradnorm': gradnorm, 'graddiff': difference}
        print('\n>>> Round: {: >4d} / Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
            round_i, gradnorm, difference, end_time - start_time))
        self.metrics.update_grads_stats(round_i, stats)

    def test_latest_model_on_evaldata(self, round_i):
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result:
            print('>>> Test on eval: round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            # print('=' * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)

    def test_latest_model_on_traindata_only_acc_loss(self, round_i):
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=False)
        end_time = time.time()

        if self.print_result:
            print('>>> Test on train: round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            # print('=' * 102 + "\n")

        self.metrics.update_train_stats_only_acc_loss(round_i, stats_from_eval_data)

    def local_test(self, use_eval_data=True):
        assert self.latest_model is not None
        self.worker.set_flat_model_params(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.clients:
            tot_correct, num_sample, loss = c.local_test(use_eval_data=use_eval_data)

            tot_corrects.append(tot_correct)
            num_samples.append(num_sample)
            losses.append(loss)

        ids = [c.id for c in self.clients]
        # groups = [c.group for c in self.clients]

        stats = {'acc': sum(tot_corrects) / sum(num_samples),
                 'loss': sum(losses) / sum(num_samples),
                 'num_samples': num_samples, 'ids': ids}

        return stats

    def save_model(self, round_i):
        self.worker.save(path=os.path.sep.join((self.metrics.result_path, self.metrics.exp_name, f'model_at_round_{round_i}.pkl')))
