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
        self.held_out_for_train = 300  # TODO 100个作为 test
        # assert len(self.clients) == 400, '按照 fair resource 的要求使用400个客户端'
        # 计算客户端拥有的数据比例
        num_samples = []
        for client in self.clients[:self.held_out_for_train]:
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
        indices = np.random.choice(self.held_out_for_train, num_clients, replace=False, p=self.pk)
        # 可能有重复, 如果 pk 不均匀的话
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns):
        """
        MAML 的聚合是普通的聚合
        :param wsolns:
        :return:
        """
        # TODO 这里假设所有的设备的数据量相同, 这一点在元学习中是可以保障的; 另, 这里直接修改了值应该是不存在引用
        new_weights = dict()
        weight_keys = wsolns[0][1].keys()
        for k in weight_keys:
            new_weights[k] = np.zeros_like(wsolns[0][1][k])
            total_num = 0
            for num_samples, sol in wsolns:
                new_weights[k] += sol[k] * num_samples
                total_num += num_samples
            new_weights[k] /= total_num
        return new_weights

    def local_test_only_acc_test_after_k_support(self, round_i, train_batches, k=5, sync_params=True):
        """
        测试
        :param round_i:
        :param on_train: 无用
        :param sync_params:
        :return:
        """
        total_sup_losses = []
        total_sup_corrects = []
        total_sup_data_num = 0
        sup_data_num = []
        total_query_k_losses = []
        total_query_k_corrects = []
        total_query_data_num = 0
        query_data_num = []
        for c in self.clients[self.held_out_for_train:]:
            if sync_params:
                self.client_model.set_params(self.latest_model)
            support_data, query_data = c.train_data, c.eval_data
            # support 都是一维, acc 等都是更新次数的维度
            support_loss, support_correct, query_losses, query_correct = c.model.test(support_data, query_data)
            #
            # for update_step in range(k):
            #     train_batch = next(train_batches[c])
            #     num_sprt_samples, num_qry_samples, support_loss, support_cnt, query_losses, query_cnt = c.model.solve_sgd(train_batch, c.eval_data)
            #     total_query_data_num += num_qry_samples
            #     total_query_k_losses.append(query_losses)
            #     total_query_k_corrects.append(query_cnt)
            #     query_data_num.append(num_qry_samples)
            # g = 5
            # 此时的模型参数如同之前的
            train_sz = c.num_train_samples
            test_sz = c.num_test_samples
            sup_data_num.append(train_sz)
            total_sup_corrects.append(support_correct)
            total_sup_losses.append(support_loss)
            total_sup_data_num += train_sz
            # 处理 query
            total_query_data_num += test_sz
            total_query_k_losses.append(query_losses)
            total_query_k_corrects.append(query_correct)
            query_data_num.append(test_sz)
        # 相关的计算
        avg_sup_acc = np.sum(total_sup_corrects) / total_sup_data_num
        avg_sup_loss = np.dot(total_sup_losses, sup_data_num) / total_sup_data_num
        print(f'\n\nSupport set average loss : {avg_sup_loss:.4f}, average accuracy: {avg_sup_acc:.3%}')
        # query : [num_clients, K]
        query_k_corr = np.stack(total_query_k_corrects, axis=0)
        query_k_loss = np.stack(total_query_k_losses, axis=0)
        avg_query_loss = []
        avg_query_acc = []
        for k in range(5):
            avg_query_loss.append(np.round(np.dot(query_k_loss[:, k], query_data_num) / total_query_data_num, 4))
            avg_query_acc.append(np.round(np.sum(query_k_corr[:, k]) / total_query_data_num, 3))
        print(f'Query set average loss : {avg_query_loss}, average accuracy: {avg_query_acc}', end='\n\n')

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        # 这个地方是生成 train 和 test 的迭代器. 使用 next 即可进行一次迭代. batch
        train_batches = {}
        for c in self.clients:
            train_batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds + 2)

        test_batches = {}
        for c in self.clients:
            test_batches[c] = gen_batch(c.eval_data, self.batch_size, self.num_rounds + 2)

        print('Have generated training and testing batches for all devices/tasks...')


        for i in range(self.num_rounds + 1):
            if (i + 1) % self.eval_every_round:
                self.local_test_only_acc_test_after_k_support(i, train_batches=train_batches, k=5, sync_params=True)
                # corr和loss都混合了 support 和 query 的准确率
                # mean_corr = np.mean(corrects)
                # mean_loss = np.mean(losses)
                # print(f'Mean corr: {mean_corr}, mean loss: {mean_loss}')
            print('Round', i)
            # TODO 训练 train 的数据, 这里client 会重复? 如果有重复就没必要再进行更新吧?
            indices, selected_clients = self.select_clients_with_held_out(round_i=i, num_clients=self.clients_per_round)

            selected_clients = selected_clients.tolist()
            wsolns = []
            samples = []
            for c in selected_clients:
                # communicate the latest model
                c.set_params(self.latest_model)
                # 使用 mini-bacth
                support_bacth = next(train_batches[c])
                query_batch = next(test_batches[c])
                # print('TRAIN TEST DIFF:', np.sum(support_bacth[0] - query_batch[0]))
                params, comp, num_samples, (support_loss, support_acc, query_losses, query_accs) = c.model.solve_gd(support_bacth, query_batch)
                # print(c.id, support_loss, support_acc, query_losses, query_accs)
                # train_stats = {'loss': float(support_loss), 'acc': float(support_acc)}

                # self.metrics.update_train_stats_only_acc_loss(round_i=i, train_stats=train_stats)
                # stats = dict()
                # for k, (k_step_corr, k_step_loss) in enumerate(zip(query_accs, query_losses)):
                #     stats['step_' + str(k) + '_loss'] = float(k_step_loss)
                #     stats['step_' + str(k) + '_acc'] = float(k_step_corr)
                # self.metrics.update_custom_scalars(round_i=i, **stats)
                wsolns.append(params)
                samples.append(num_samples)

            self.latest_model = self.aggregate(list(zip(samples, wsolns)))

            # if (i + 1) % self.save_every_round:
            #     self.metrics.write()
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


