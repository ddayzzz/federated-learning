import numpy as np
import pandas as pd
import os
import tensorflow as tf
import time
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics2
# from flmod.utils.metrics import Metrics
from flearn.utils.tf_utils import process_grad


class BaseFedarated(object):

    def __init__(self, params, learner, dataset, optimizer, append2metric=None):
        """
        联邦学习框基类
        :param params: 参数
        :param learner: 需要学习的模型
        :param dataset: 数据集
        :param optimizer: 优化器, 这个用于创建静态图的 loss 的 op
        """
        # transfer parameters to self
        # for key, val in params.items():
        #     setattr(self, key, val);
        # 显式指定参数
        self.optimizer = optimizer
        self.seed = params['seed']
        self.num_epochs = params['num_epochs']
        self.num_rounds = params['num_rounds']
        self.clients_per_round = params['clients_per_round']
        self.save_every_round = params['save_every']
        # 在 train 和 test
        self.eval_every_round = params['eval_every']
        self.batch_size = params['batch_size']
        self.hide_client_output = params['quiet']
        # create worker nodes
        tf.reset_default_graph()
        # 客户端的模型对象
        self.client_model = learner(*params['model_options'], self.optimizer, self.seed)
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))

        self.num_clients = len(self.clients)
        #
        if not os.path.exists(params['result_dir']):
            self.name = '_'.join(['', f'wn{params["clients_per_round"]}', f'tn{self.num_clients}'])
            self.metrics = Metrics2(clients=self.clients, options=params, name=self.name, append2suffix=append2metric)
            self.metric_prefix = os.path.join(self.metrics.result_path, self.metrics.exp_name)
            self.start_round = 0
            self.checkpoint_prefix = os.path.join(self.metric_prefix, 'checkpoints')
            if not os.path.exists(self.checkpoint_prefix):
                os.mkdir(self.checkpoint_prefix)
        else:
            self.metric_prefix = params['result_dir']
            self.checkpoint_prefix = os.path.join(self.metric_prefix, 'checkpoints')
            # 加载checkpoints, 默认加载最新吧
            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_prefix)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.client_model.load_checkpoint(checkpoint.model_checkpoint_path)
                print(">>> Checkpoint loaded:", checkpoint.model_checkpoint_path)
                last = checkpoint.model_checkpoint_path.find('.ckpt')
                start = checkpoint.model_checkpoint_path.rfind('round_')
                self.start_round = int(checkpoint.model_checkpoint_path[start + 6:last])
                print(">>> Resume from round: ", self.start_epoch)

        self.latest_model = self.client_model.get_params()  # 永远是最新的模型

    def __del__(self):
        """
        关闭 session
        :return:
        """
        if hasattr(self, 'client_model'):
            self.client_model.close()

    def setup_clients(self, dataset, model=None):
        """
        设置客户端
        :param dataset: 数据集元素
        :param model: 模型
        :return:
        """
        users, groups, train_data, test_data = dataset
        assert 'x' in train_data[users[-1]], '只能支持处理后的数据'
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def show_grads(self):
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)

        intermediate_grads = []
        samples = []

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model)
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads.append(global_grads)

        return intermediate_grads

    def local_test_only_acc(self, round_i, on_train=False, sync_params=True):
        """
        测试模型在所有客户端上的准确率
        :param round_i:
        :param on_train: 是否是训练数据
        :param sync_params: 同步参数(如果量此调用, 第二次可以设为 False)
        :return: {'loss': loss, 'acc': acc, 'time': time_diff }
        """
        num_samples = []
        tot_correct = []
        tot_losses = []
        if sync_params:
            self.client_model.set_params(self.latest_model)
        begin_time = time.time()
        for c in self.clients:
            correct, loss, ds = c.test(on_train=on_train)
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

    def save_model(self, round_i):
        """
        保存为 checkpoint
        :param round_i:
        :return:
        """
        filename = os.path.join(self.checkpoint_prefix, 'checkpoints_at_round_{}.ckpt'.format(round_i + 1))
        self.client_model.save(filename)

    def select_clients(self, round_i, num_clients=20):
        """
        选择客户端(均匀选择)
        :param round_i:
        :param num_clients:
        :return: 选择的客户端索引, 客户端实例列表
        """
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round_i)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns):
        """
        聚合模型(平均聚合)
        :param wsolns:
        :return:
        """
        total_weight = 0.0
        base = [0] * len(wsolns[0][1])  # 有多少的网络层? wsolns[i] : (num_samples, 参数)
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w  # 样本的数量
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)
        # 每一层 / 样本的数量
        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def eval_to_file(self, round_i, sync_params=True):
        """
        测试模型在所有客户端上的准确率
        :param round_i:
        :param on_train: 是否是训练数据
        :param sync_params: 同步参数(如果量此调用, 第二次可以设为 False)
        :return: {'loss': loss, 'acc': acc, 'time': time_diff }
        """
        save_path = os.path.join(self.metric_prefix, 'eval_result_at_round_{}.csv'.format(round_i))
        df = pd.DataFrame(columns=['id', 'train_acc', 'train_loss', 'test_loss', 'test_acc'])
        if sync_params:
            self.client_model.set_params(self.latest_model)
        # begin_time = time.time()
        for i, c in enumerate(self.clients):
            train_correct, train_loss, train_ds = c.test(on_train=True)
            test_correct, test_loss, test_ds = c.test(on_train=False)
            # 添加数据信息
            df = df.append({'id': c.id, 'train_acc': train_correct / train_ds, 'test_acc': test_correct / test_ds,
                       'train_loss': train_loss, 'test_loss': test_loss}, ignore_index=True)
            print('Eval on client:', c.id)
        # end_time = time.time()
        # 保存为路径
        df.to_csv(save_path)
        print(f'>>> Saved eval result to "{save_path}"')

    def eval(self):
        print('Eval with {} workers ---'.format(self.clients_per_round))
        self.eval_to_file(round_i=self.num_rounds, sync_params=True)
