import numpy as np
import time
from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad


class Server(BaseFedarated):

    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        inner_opt = PerturbedGradientDescent(params['lr'], params['mu'])
        append2metric = f'mu{params["mu"]}'
        super(Server, self).__init__(params, learner, dataset, optimizer=inner_opt, append2metric=append2metric)
        self.drop_rate = params['drop_rate']

    def local_test_grads(self, round_i):
        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)
        client_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []
        begin = time.time()
        for c in self.clients:
            num, client_grad = c.get_grads(model_len)
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads = np.add(global_grads, client_grad * num)
        global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))
        gradnorm = np.linalg.norm(global_grads)
        difference = 0
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference = difference * 1.0 / len(self.clients)
        stats = {'graddiff': difference, 'gradnorm': gradnorm}
        end = time.time()
        # 输出
        if self.print_result:
            print('\n>>> Round: {: >4d} / Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
                round_i, gradnorm, difference, end - begin))
        return stats

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        for i in range(self.num_rounds):
            # test model
            if (i + 1) % self.eval_every_round == 0:
                stats = self.local_test_only_acc(round_i=i, on_train=False,
                                                 sync_params=False)  # have set the latest model for all clients
                stats_train = self.local_test_only_acc(round_i=i, on_train=True, sync_params=False)

            # 输出计算的梯度信息
            stats_grads = self.local_test_grads(round_i=i)
            self.metrics.update_grads_stats(i, stats_grads)
            # 训练
            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_rate)),
                                              replace=False)

            csolns = []  # buffer for receiving client solutions
            cstats = []  # 用于记录信息
            # 将当前模型的slot 与模型的参数的值关联起来
            self.optimizer.set_params(self.latest_model, self.client_model)

            for idx, c in enumerate(selected_clients.tolist()):
                # communicate the latest model
                c.set_params(self.latest_model)

                # total_iters = int(self.num_epochs * c.num_samples / self.batch_size) + 2  # randint(low,high)=[low,high)

                # solve minimization locally
                if c in active_clients:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size, round_i=i)
                else:
                    soln, stats = c.solve_inner(num_epochs=np.random.randint(low=1, high=self.num_epochs), batch_size=self.batch_size, round_i=i)

                # gather solutions from client
                csolns.append(soln)
                cstats.append(stats)
                # track communication cost
                # self.metrics.update_client(rnd=i, cid=c.id, stats=stats)
            self.metrics.extend_commu_stats(round_i=i, stats_list=cstats)
            # update models
            self.latest_model = self.aggregate(csolns)
            # 这里直接更改了模型. 所以测试的时候不再同步参数
            self.client_model.set_params(self.latest_model)
            if (i + 1) % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()

        # final test model
        stats = self.local_test_only_acc(round_i=self.num_rounds, on_train=False,
                                         sync_params=True)  # have set the latest model for all clients
        stats_train = self.local_test_only_acc(round_i=self.num_rounds, on_train=True, sync_params=False)
        self.metrics.write()
