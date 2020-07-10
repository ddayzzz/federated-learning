import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf


from .fedbase import BaseFedarated
from flearn.utils.tf_utils import norm_grad
from flearn.utils.model_utils import gen_batch


class Server(BaseFedarated):

    def __init__(self, params, learner, dataset):
        """
        定义 MAML 在联邦学习环境下的实现
        :param params:
        :param learner:
        :param dataset:
        """
        print('Using MAML to Train')
        inner_opt = tf.train.GradientDescentOptimizer(params['lr'])
        num_fine_tune = params['num_fine_tune']
        append2metric = f"num_fine_tune[{num_fine_tune}]"
        super(Server, self).__init__(params, learner, dataset, optimizer=inner_opt, append2metric=append2metric)
        self.num_fine_tune = num_fine_tune
        self.held_out = 100  # TODO 100个作为 test
        assert len(self.clients) == 400, '按照 fair resource 的要求使用400个客户端'
        # 计算客户端拥有的数据比例
        num_samples = []
        for client in self.clients[:300]:
            num_samples.append(client.num_train_samples)
        total_samples = np.sum(np.asarray(num_samples))
        self.pk = [item * 1.0 / total_samples for item in num_samples]

    def select_clients_with_held_out(self, round_i, num_clients=20):
        """
        前300为训练, 后100不管
        :param round_i:
        :param num_clients:
        :return:
        """
        np.random.seed(round_i)
        # 这里的策略为: 按照拥有的数据的比例 + 普通形式聚合
        indices = np.random.choice(300, num_clients, replace=False, p=self.pk)
        # 可能有重复, 如果 pk 不均匀的话
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns):
        """
        MAML 的聚合是普通的聚合
        :param wsolns:
        :return:
        """
        # TODO 这里假设所有的设备的数据量相同, 这一点在元学习中是可以保障的; 另, 这里直接修改了值应该是不存在引用
        num_clients = len(wsolns)
        new_weights = wsolns[0]
        for var_name in new_weights.keys():
            vars = [soln[var_name] for soln in wsolns[1:]]
            for var_other in vars:
                new_weights[var_name] += var_other
            new_weights[var_name] /= num_clients
        return new_weights

    def local_test_only_acc(self, round_i, on_train=False, sync_params=True):
        """
        测试
        :param round_i:
        :param on_train:
        :param sync_params:
        :return:
        """
        corrects, losses = [], []
        for c in self.clients[300:]:
            if sync_params:
                self.client_model.set_params(self.latest_model)
            if on_train:
                data = c.train_data
            else:
                data = c.eval_data
            correct, loss = c.model.test(data)
            corrects.append(correct)
            losses.append(loss)
        return corrects, losses

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        # 这个地方是生成 train 和 test 的迭代器. 使用 next 即可进行一次迭代. batch
        # train_batches = {}
        # for c in self.clients:
        #     train_batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds + 2)
        #
        # test_batches = {}
        # for c in self.clients:
        #     test_batches[c] = gen_batch(c.eval_data, self.batch_size, self.num_rounds + 2)

        print('Have generated training and testing batches for all devices/tasks...')

        alpha = 0
        beta = 0
        K = 5
        for i in range(self.num_rounds + 1):
            if (i + 1) % self.eval_every_round:
                corrects, losses = self.local_test_only_acc(i, on_train=False, sync_params=True)
                mean_corr, mean_loss = float(np.mean(corrects)), float(np.mean(losses))
                test_stats = {'loss': mean_loss, 'acc': mean_corr}
                self.metrics.update_eval_stats(i, test_stats)
            print('Round', i)
            # TODO 训练 train 的数据, 这里client 会重复? 如果有重复就没必要再进行更新吧?
            indices, selected_clients = self.select_clients_with_held_out(round_i=i, num_clients=self.clients_per_round)

            selected_clients = selected_clients.tolist()
            wsolns = []

            for c in selected_clients:
                # communicate the latest model
                c.set_params(self.latest_model)
                # 直接输入完整的数据
                params, comp, (support_loss, support_acc, query_losses, query_accs) = c.model.solve_gd(c.train_data,
                                                                                                       c.eval_data)
                # print(c.id, support_loss, support_acc, query_losses, query_accs)
                train_stats = {'loss': float(support_loss), 'acc': float(support_acc)}

                self.metrics.update_train_stats_only_acc_loss(round_i=i, train_stats=train_stats)
                self.metrics.update_custom_scalars(i, query_acc=float(query_accs[-1]), query_loss=float(query_losses[-1]))
                wsolns.append(params)

            self.latest_model = self.aggregate(wsolns)

            if (i + 1) % self.save_every_round:
                self.metrics.write()
        print("###### finish meta-training, start meta-testing ######")


        # test_accuracies = []
        # initial_accuracies = []
        # # 前300为 train 后 100为
        # for c in self.clients[len(self.clients)-self.held_out:]:  # meta-test on the held-out tasks
        #     # start from the same initial model that is learnt using q-FFL + MAML
        #     c.set_params(self.latest_model)
        #     ct, cl, ns = c.test_error_and_loss()
        #     initial_accuracies.append(ct * 1.0/ns)
        #     # solve minimization locally
        #     for iters in range(self.num_fine_tune):  # run k-iterations of sgd
        #         batch = next(train_batches[c])
        #         _, grads1, loss1 = c.solve_sgd(batch)
        #     ct, cl, ns = c.test_error_and_loss()
        #     test_accuracies.append(ct * 1.0/ns)
        # print("initial mean: ", np.mean(np.asarray(initial_accuracies)))
        # print("initial variance: ", np.var(np.asarray(initial_accuracies)))
        # print(self.output)
        # print("personalized mean: ", np.mean(np.asarray(test_accuracies)))
        # print("personalized variance: ", np.var(np.asarray(test_accuracies)))
        # np.savetxt(self.output+"_"+"test.csv", np.asarray(test_accuracies), delimiter=",")
        self.metrics.write()


