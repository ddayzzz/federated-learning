from flmod.solvers.fedbase import BaseFedarated
from flmod.models.models import choose_model_criterion
from flmod.optimizers.gd import GradientDescend
import numpy as np
import torch


class FedAvgSchemes(BaseFedarated):

    def __init__(self, options, all_data_info):
        model, crit = choose_model_criterion(options=options)
        self.optimizer = GradientDescend(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        suffix = f'mu{options["mu"]}_dp{options["drop_rate"]}_scheme{options["scheme"]}'
        super(FedAvgSchemes, self).__init__(options=options, model=model, dataset=all_data_info, optimizer=self.optimizer, criterion=crit, append2metric=suffix)
        self.clients_select_prob = self.get_clients_prob()  # p_k
        self.scheme = options["scheme"]  # scheme 1,2, transformed 2
        assert self.num_clients == 100, '客户端数量是100'

    def get_clients_prob(self):
        num_alldata = []
        for client in self.clients:
            num_alldata.append(client.num_train_data)
        return np.array(num_alldata) / sum(num_alldata)

    def select_clients_with_prob(self, round):
        num_clients = min(self.clients_per_round, self.num_clients)
        np.random.seed(round)
        index = np.random.choice(len(self.clients), num_clients, p=self.clients_select_prob)
        index = sorted(index.tolist())

        select_clients = []
        select_index = []
        repeated_times = []
        for i in index:
            if i not in select_index:
                select_clients.append(self.clients[i])
                select_index.append(i)
                repeated_times.append(1)
            else:
                # 对应的顺序(即对应的客户端的index)出现的次数
                repeated_times[-1] += 1
        return select_clients, repeated_times

    def aggregate(self, solns, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # p_k*(N/K) * W_k
        if self.scheme.startswith('2'):
            # scheme 2; transformed scheme 2
            for num_sample, local_solution in solns:
                averaged_solution += local_solution * num_sample  # simply sum the flatten parameters
            averaged_solution /= self.num_train_data  # 这里是总共的数据量,不是运行客户端的数量
            averaged_solution *= (self.num_clients / self.clients_per_round)
        else:
            # scheme 1
            repeated_times = kwargs['repeated_times']
            assert repeated_times == len(solns)
            for i, (num_sample, local_solution) in enumerate(solns):
                averaged_solution += local_solution * repeated_times[i]
            averaged_solution /= self.clients_per_round  # TODO 一般情况下 clients_per_round = min(self.clients_per_round, self.num_clients)
        return averaged_solution.detach()

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i + 1}')
            # eval on train
            if (round_i + 1) % self.eval_on_train_every_round == 0:
                self.test_latest_model_on_traindata_only_acc_loss(round_i)
            # eval on test
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.test_latest_model_on_evaldata(round_i)

            if self.scheme.startswith('2'):
                selected_clients, repeated_times = self.select_clients(round=round_i, num_clients=self.clients_per_round), None
            else:
                selected_clients, repeated_times = self.select_clients_with_prob(round_i)
            solutions, stats = self.local_train(selected_clients, round_i=round_i)

            self.metrics.extend_commu_stats(round_i, stats)

            # update global weights
            self.latest_model = self.aggregate(solutions, repeated_times=repeated_times)

            lr = self.optimizer.get_current_lr()
            self.metrics.update_custom_scalars(round_i, decaied_lr=lr)

            # 调整学习率
            self.optimizer.inverse_prop_decay_learning_rate(round_i=round_i)
            if (round_i + 1) % self.save_every_round == 0:
                self.save_model(round_i)
                self.metrics.write()

        self.test_latest_model_on_traindata(self.num_rounds)
        self.test_latest_model_on_evaldata(self.num_rounds)
        self.metrics.write()