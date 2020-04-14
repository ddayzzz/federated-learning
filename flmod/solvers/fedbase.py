import numpy as np
import os
import time
import abc
import torch
from flmod.utils.worker_utils import MiniDataset
from flmod.clients.base_client import BaseClient
from flmod.utils.data_utils import DatasetSplit
from flmod.utils.metrics import Metrics


class BaseFedarated(object):

    def __init__(self, options, model, dataset, optimizer, criterion, eval_criterion, worker=None):
        """
        定义联邦学习的基本的服务器, 这里的模型是在所有的客户端之间共享使用
        :param learner:
        :param dataset:
        :param model:
        """
        if worker is None:
            from flmod.models.worker import Worker
            worker = Worker(model=model, criterion=criterion, eval_criterion=eval_criterion, optimizer=optimizer, options=options)
        self.worker = worker
        self.device = options['device']
        # 记录总共的训练数据
        self.num_train_data = 0
        self.clients = self.setup_clients(dataset=dataset,
                                          criterion=criterion,
                                          eval_criterion=eval_criterion,
                                          worker=worker,
                                          batch_size=options['batch_size'])
        self.num_epochs = options['num_epochs']
        self.latest_model = self.worker.get_flat_model_params()
        self.name = '_'.join(['', f'wn{options["clients_per_round"]}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(clients=self.clients, options=options, name=self.name)
        self.print_result = True

    def setup_clients(self, dataset, criterion, eval_criterion, worker, batch_size):
        users, groups, train_data, test_data, entire_train_dataset, entire_test_dataset = dataset
        if entire_test_dataset is None and entire_train_dataset is not None:
            # train  和 test 的数据是合并的
            if len(groups) == 0:
                groups = [None for _ in users]
            all_clients = []
            for user, group in zip(users, groups):
                if isinstance(user, str) and len(user) >= 5:
                    user_id = int(user[-5:])
                else:
                    user_id = int(user)
                self.num_train_data += len(train_data[user])
                # c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
                c = BaseClient(id=user_id, worker=self.worker, batch_size=batch_size, criterion=criterion,
                               eval_criterion=eval_criterion,
                               train_dataset=DatasetSplit(entire_train_dataset, idxs=train_data[user]['x_index']),
                               test_dataset=DatasetSplit(entire_train_dataset, test_data[user]['x_index']))
                all_clients.append(c)
            return all_clients
        elif entire_test_dataset is None and entire_train_dataset is None:
            # 没有 index 的数据, train 和 tes 直接带有数据
            if len(groups) == 0:
                groups = [None for _ in users]

            all_clients = []
            for user, group in zip(users, groups):
                if isinstance(user, str) and len(user) >= 5:
                    user_id = int(user[-5:])
                else:
                    user_id = int(user)
                self.num_train_data += len(train_data[user])
                c = BaseClient(id=user_id, worker=self.worker, batch_size=batch_size, criterion=criterion,
                               eval_criterion=eval_criterion,
                               train_dataset=MiniDataset(train_data[user]['x'], train_data[user]['y']),
                               test_dataset=MiniDataset(test_data[user]['x'], test_data[user]['y']))
                all_clients.append(c)
            return all_clients
        else:
            # 采用 index 的数据
            if len(groups) == 0:
                groups = [None for _ in users]

            all_clients = []
            for user, group in zip(users, groups):
                if isinstance(user, str) and len(user) >= 5:
                    user_id = int(user[-5:])
                else:
                    user_id = int(user)
                self.num_train_data += len(train_data[user])
                # c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
                c = BaseClient(id=user_id, worker=self.worker, batch_size=batch_size, criterion=criterion,
                               eval_criterion=eval_criterion,
                               train_dataset=DatasetSplit(entire_train_dataset, idxs=train_data[user]['x_index']),
                               test_dataset=DatasetSplit(entire_test_dataset, test_data[user]['x_index']))
                all_clients.append(c)
            return all_clients



    def local_train(self, selected_clients, round_i):
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train(round_i, self.num_epochs)

            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients

        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def aggregate(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """
        # flatten
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        num = 0
        for num_sample, local_solution in solns:
            num += 1
            averaged_solution += local_solution  # simply sum the flatten parameters
        averaged_solution /= num  # divide the num of clients
        return averaged_solution.detach()

    def test_latest_model_on_traindata(self, round_i):
        """
        在训练数据集上测试
        :param round_i:
        :return:
        """
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        # Record the global gradient
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

        # Measure the gradient difference
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
        # Collect stats from total train data
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

        # Measure the gradient difference
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
        # Collect stats from total eval data
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result:
            print('>>> Test: round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            # print('=' * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)

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
