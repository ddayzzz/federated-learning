from flmod.solvers.fedbase import BaseFedarated, BaseClient, BaseModel
import numpy as np
import torch


class Adam:

    """
    全局 Adam, 用来基于从客户端收集的梯度, 来更新全局网络的参数
    """

    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        """

        :param lr:
        :param betas:
        :param eps:
        """
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = dict()
        self.v = dict()
        self.n = 0
        self.creted_momtem_grad_index = set()

    def __call__(self, params, grads, i):
        # 创建对应的 id
        if i not in self.m:
            self.m[i] = torch.zeros_like(params)
        if i not in self.v:
            self.v[i] = torch.zeros_like(params)

        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * torch.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        params.sub_(alpha * self.m[i] / (torch.sqrt(self.v[i]) + self.eps))

    def increase_n(self):
        self.n += 1


class BaseFedaratedAdvanced(BaseFedarated):

    def __init__(self, options, model: BaseModel, read_dataset, client_class=BaseClient, append2metric=None, more_metric_to_train=None):
        super(BaseFedaratedAdvanced, self).__init__(options=options, model=model, read_dataset=read_dataset, append2metric=append2metric, more_metric_to_train=more_metric_to_train, client_class=client_class)
        self.train_clients = None
        self.eval_clients = None
        self.test_clients = None

    def split_train_validation_test_clients(self, train_rate=0.8, val_rate=0.1):
        np.random.seed(self.options['seed'])
        train_rate = int(train_rate * self.num_clients)
        val_rate = int(val_rate * self.num_clients)
        test_rate = self.num_clients - train_rate - val_rate

        assert test_rate > 0 and val_rate > 0 and test_rate > 0, '不能为空'

        ind = np.random.permutation(self.num_clients)
        arryed_cls = np.asarray(self.clients)
        self.train_clients = arryed_cls[ind[:train_rate]].tolist()
        self.eval_clients = arryed_cls[ind[train_rate:train_rate + val_rate]].tolist()
        self.test_clients = arryed_cls[ind[train_rate + val_rate:]].tolist()

        print('用于训练的客户端数量{}, 用于验证:{}, 用于测试: {}'.format(len(self.train_clients), len(self.eval_clients),
                                                       len(self.test_clients)))

    def aggregate_grads_weights(self, solns, lr, num_samples, weights_before):
        """
        合并梯度, 这个和 fedavg 不相同
        :param grads:
        :return:
        """
        m = len(solns)
        g = []
        for i in range(len(solns[0])):
            # i 表示的当前的梯度的 index
            # 总是 client 1 的梯度的形状
            grad_sum = torch.zeros_like(solns[0][i])
            all_sz = 0
            for ic, sz in enumerate(num_samples):
                grad_sum += solns[ic][i] * sz
                all_sz += sz
            # 累加之后, 进行梯度下降
            g.append(grad_sum / all_sz)
        return [u - (v * lr) for u, v in zip(weights_before, g)]

    def aggregate_grads_simple(self, solns, lr, weights_before):
        """
        合并梯度(直接合并后除以参数的数量), 这个和 fedavg 不相同
        :param grads:
        :return:
        """
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
        new_weights = [u - (v * lr / m) for u, v in zip(weights_before, g)]
        return new_weights