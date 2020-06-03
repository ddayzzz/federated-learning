import time
from typing import Dict, List
from torch.utils.data import DataLoader
from flmod.models.workers import Worker


class BaseClient(object):

    def __init__(self, id, worker: Worker, batch_size: int, criterion, train_dataset, test_dataset):
        """
        定义客户端类型
        :param id: 客户端的 id
        :param worker: 客户端和模型连接器 Worker
        :param batch_size: mini-batch 所用的 batch 大小
        :param criterion: 分类任务的评价器
        :param train_dataset: 训练集
        :param test_dataset: 测试集/验证集
        """
        self.id = id
        self.criterion = criterion
        self.num_train_data = len(train_dataset)
        self.num_test_data = len(test_dataset)
        self.worker = worker
        self.train_data_loader = DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True, num_workers=2)
        self.test_data_loader = DataLoader(test_dataset,
                                           batch_size=batch_size, shuffle=False)

    def get_model_params_list(self) -> List:
        """
        获得的模型参数
        :return: 参数 List
        """
        return self.worker.get_model_params_list()

    def set_model_params(self, model_params_dict: Dict):
        """
        设置模型的参数
        :param model_params_dict:参数 Dict
        :return:
        """
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        """
        获得扁平化的参数
        :return: Tensor 对象的向量
        """
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        """
        将扁平化的参数应用到当前的模型
        :param flat_params:
        :return:
        """
        self.worker.set_flat_model_params(flat_params)

    def get_flat_grads(self):
        """
        获得扁平化处理的梯度
        :return: Numpy的 Array 对象, 样本数量
        """
        grad_in_tensor, total_samples = self.worker.get_flat_grads(self.train_data_loader)
        # if not isinstance(grad_in_tensor, np.ndarray):
        #     grad_in_tensor = grad_in_tensor.cpu().detach().numpy()
        grad_in_tensor = grad_in_tensor.numpy()
        return grad_in_tensor, total_samples

    def solve_grad(self):
        """
        计算梯度
        :return:
        """
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * self.num_train_data
        bytes_r = self.worker.model_bytes
        stats = {'id': self.id, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}

        grads, total_samples = self.get_flat_grads()  # Return grad in numpy array

        return (total_samples, grads), stats

    def local_train(self, round_i, num_epochs, **kwargs):
        """
        计算利用当前客户端的数据进行模型参数更新
        :param round_i:
        :param num_epochs:
        :param kwargs:
        :return:
        1: num_samples: 训练所使用的样本的数量
        2: soln: 更新后的模型
        3. Statistic Dict contain
            3.1: bytes_write: number of bytes transmitted
            3.2: comp: number of FLOPs executed in training process
            3.3: bytes_read: number of bytes received
            3.4: other stats in train process
        """
        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats = self.worker.local_train(num_epochs, self.train_data_loader, round_i, self.id)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.id, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (self.num_train_data, local_solution), stats

    def local_test(self, use_eval_data=True):
        """
        利用测试集/训练集进行模型测试
        :param use_eval_data: 是否使用测试集的数据,否则使用训练集的数据
        :return:
        1. 正确分类的样本数(具体的值取决于评价器的选取)
        2. 样本的总数量
        3. 损失函数值的和
        """
        if use_eval_data:
            dataloader = self.test_data_loader
        else:
            dataloader = self.train_data_loader

        tot_correct, num_samples, loss = self.worker.local_test(dataloader)

        return tot_correct, num_samples, loss
