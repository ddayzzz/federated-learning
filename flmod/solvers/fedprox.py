import torch
import time
import numpy as np
from flmod.solvers.fedbase import BaseFedarated
from flmod.models.models import choose_model_criterion
from flmod.optimizers.pgd import PerturbedGradientDescent


class FedProx(BaseFedarated):

    def __init__(self, options, all_data_info):
        """
        FedProx
        原始论文给出的 tf 的代码的实现不完全相同, 主要的区别在于:
        1. 选取设备的方式(tf代码是完全随机的)
        2. 合并权重的时候除以了运行的客户端总数 K
        :param options:
        :param all_data_info:
        """
        model, crit = choose_model_criterion(options=options)
        self.optimizer = PerturbedGradientDescent(model.parameters(), lr=options['lr'], mu=options['mu'], weight_decay=options['wd'])
        self.scheme = 'unfo_prob' if options['scheme'] == '' else options['scheme']
        # FedProx 也接受 scheme 参数, 含义如下
        assert self.scheme in ['prob_simp', 'unfo_prob'], 'prob_simp for sampling clients with their probilities and simply average the weights; unfo_prob(default): uniformly chose clients and average weight by prob'

        suffix = f'mu{options["mu"]}_dp{options["drop_rate"]}_scheme[{self.scheme}]'
        super(FedProx, self).__init__(options=options, model=model, dataset=all_data_info, optimizer=self.optimizer,
                                     criterion=crit, append2metric=suffix)
        self.drop_rate = options['drop_rate']
        self.prob = self.get_clients_prob()

    def get_clients_prob(self):
        num_alldata = []
        for client in self.clients:
            num_alldata.append(client.num_train_data)
        assert sum(num_alldata) == self.num_train_data
        return np.array(num_alldata) / sum(num_alldata)

    def prob_select_simple_average(self):
        return self.scheme == 'prob_simp'

    def select_clients_with_prob(self, round, num_clients=20):
        """
        这里只输出客户端的编号和被选择的客户端
        :param round:
        :param num_clients:
        :return:
        """
        num_clients = min(num_clients, self.num_clients)  # 选择的客户端
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        target_clients = []
        client_indices = []
        multiple_times = []
        selected_clients_indx = np.random.choice(self.num_clients, num_clients, replace=True, p=self.prob)
        selected_clients_indx = sorted(selected_clients_indx)
        for i in selected_clients_indx:
            if i not in client_indices:
                client_indices.append(i)
                multiple_times.append(1)
                target_clients.append(self.clients[i])
            else:
                # 注意这是有序的
                multiple_times[-1] += 1
        return target_clients, multiple_times, client_indices

    def aggregate_simply(self, solns, multiple_times):
        averaged_solution = torch.zeros_like(self.latest_model)
        for i, (num_sample, local_solution) in enumerate(solns):
            averaged_solution += local_solution * multiple_times[i]
        averaged_solution /= self.clients_per_round  # TODO 一般情况下 clients_per_round = min(self.clients_per_round, self.num_clients)
        return averaged_solution.detach()

    def select_clients(self, round, num_clients=20):
        """
        这里只输出客户端的编号和被选择的客户端
        :param round:
        :param num_clients:
        :return:
        """
        num_clients = min(num_clients, self.num_clients)  # 选择的客户端
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        selected_clients_indx = np.random.choice(self.num_clients, num_clients, replace=False)
        return selected_clients_indx

    def local_train(self, selected_clients, round_i):
        raise NotImplementedError

    def update_optimizer_weights(self):
        # 运行万所有的数据之后才进行 - w, 所以worker中的每一个 batch 不需要进行 - w 的操作;只要在聚合完成所有数据的
        oldw = self.worker.to_model_params(self.latest_model)
        self.optimizer.set_old_weights(old_weights=oldw)

    def update_optimizer_mu(self):
        """
        更新 mu, 对于 iid 数据, mu 初始化为1, 对于生成的数据 Synthetic(non-iid), 初始化为0; 原论文的 Figure 11
        :return:
        """
        pass

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i + 1}')
            # eval on test
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.test_latest_model_on_evaldata(round_i)
            if (round_i + 1) % self.eval_on_train_every_round == 0:
                # 仅仅计算梯度
                self.calc_client_grads(round_i)
            if self.prob_select_simple_average():
                target_clients, multiple_times, client_indices = self.select_clients_with_prob(round_i, self.clients_per_round)
                # 那些客户端的编号可以完成所有的操作
                # 由于有重复的客户端以及不放回的抽样, 这里可能并没有足够的数据. 如果 dp > 0, 则将所有的客户端展开而不是利用
                if self.drop_rate > 0:
                    # TODO 这段代码还有问题, 一个思路是把所有客户端展开, 然后采样;如果判断只有一次执行的客户端就仅仅运行一次, 否则运行多次
                    expand_clients = []
                    for i, ci in enumerate(client_indices):
                        expand_clients.extend([ci] * multiple_times[i])
                    # 肯定是有重复的
                    activated_clients_indx = np.random.choice(expand_clients,
                                                              round(self.clients_per_round * (1 - self.drop_rate)),
                                                              replace=False)

                else:
                    activated_clients_indx = client_indices
                # 能够顺利完成任务客户端, 其余的不在 active 但是被选择的客户端被视为 starggle
                self.update_optimizer_weights()
                solns = []
                stats = []
                for client_index, client in zip(client_indices, target_clients):
                    client.set_flat_model_params(self.latest_model)
                    if client_index in activated_clients_indx:
                        # 运行相同的 num_epoch
                        soln, stat = client.local_train(round_i, self.num_epochs)
                    else:
                        soln, stat = client.local_train(round_i, num_epochs=np.random.randint(low=1, high=self.num_epochs))
                    solns.append(soln)
                    stats.append(stat)
                self.latest_model = self.aggregate_simply(solns, multiple_times)
            else:
                selected_clients_indices = self.select_clients(round=round_i, num_clients=self.clients_per_round)
                activated_clients_indx = np.random.choice(selected_clients_indices,
                                                          round(self.clients_per_round * (1 - self.drop_rate)),
                                                          replace=False)
                # 能够顺利完成任务客户端, 其余的不在 active 但是被选择的客户端被视为 starggle
                self.update_optimizer_weights()
                solns = []
                stats = []
                for selected_i in selected_clients_indices:
                    c = self.clients[selected_i]
                    c.set_flat_model_params(self.latest_model)
                    if selected_i in activated_clients_indx:
                        # 运行相同的 num_epoch
                        soln, stat = c.local_train(round_i, self.num_epochs)
                    else:
                        soln, stat = c.local_train(round_i, num_epochs=np.random.randint(low=1, high=self.num_epochs))
                    solns.append(soln)
                    stats.append(stat)
                self.latest_model = self.aggregate(solns)

            self.metrics.extend_commu_stats(round_i, stats)

            # update global weights
            if (round_i + 1) % self.save_every_round == 0:
                self.save_model(round_i)
                self.metrics.write()
            # 写入 mu  和 lr
            # lr = self.optimizer.get_current_lr()
            # mu = self.optimizer.get_mu()
            # self.metrics.update_custom_scalars(round_i, lr=lr, mu=mu)

        self.test_latest_model_on_traindata(self.num_rounds)
        self.test_latest_model_on_evaldata(self.num_rounds)
        self.metrics.write()
