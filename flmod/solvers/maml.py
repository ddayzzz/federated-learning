import torch
import time
import numpy as np
from flmod.solvers.fedbase import BaseFedarated
from flmod.models.models import choose_model_criterion
from flmod.optimizers.gd import GradientDescend
from flmod.models.workers import Worker
from flmod.clients.base_client import BaseClient


class Client(BaseClient):

    def __init__(self, id, worker: Worker, batch_size: int, criterion, train_dataset, test_dataset):
        super(Client, self).__init__(id, worker, batch_size, criterion, train_dataset, test_dataset)
        # 这里将自定义处理


class MAMLWorker(Worker):

    def __init__(self, model, criterion, optimizer, options):
        """
        每一个客户端等价于
        :param model:
        :param criterion:
        :param optimizer:
        :param options:
        """
        super(MAMLWorker, self).__init__(model, criterion, optimizer, options)


class MAML(BaseFedarated):

    def __init__(self, options, all_data_info):
        model, crit = choose_model_criterion(options=options)
        self.optimizer = GradientDescend(model.parameters(), lr=options['lr'])
        self.q = options['q_coef']
        suffix = f'q[{self.q}]'
        super(MAML, self).__init__(options=options, model=model, dataset=all_data_info, optimizer=self.optimizer, criterion=crit, append2metric=suffix)

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i + 1}')
            # eval on test
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.test_latest_model_on_evaldata(round_i)
            # eval on train
            if (round_i + 1) % self.eval_on_train_every_round == 0:
                self.test_latest_model_on_traindata(round_i)
            selected_clients = self.select_clients(round=round_i, num_clients=self.clients_per_round)
            solutions, stats = self.local_train(selected_clients, round_i=round_i)
            self.metrics.extend_commu_stats(round_i, stats)

            # update global weights
            self.latest_model = self.aggregate(solutions)

            if (round_i + 1) % self.save_every_round == 0:
                self.save_model(round_i)
                self.metrics.write()

        self.test_latest_model_on_traindata(self.num_rounds)
        self.test_latest_model_on_evaldata(self.num_rounds)
        self.metrics.write()