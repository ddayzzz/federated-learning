import time
from typing import Dict, List
from torch.utils.data import DataLoader
from flmod.models.base_models import BaseModel


class BaseClient(object):

    def __init__(self, id, train_dataset, test_dataset, options, optimizer, model: BaseModel):
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
        # 这个必须是客户端相关的
        self.optimizer = optimizer
        self.num_train_data = len(train_dataset)
        self.num_test_data = len(test_dataset)
        self.num_epochs = options['num_epochs']
        self.num_batch_size = options['batch_size']
        self.options = options
        self.train_dataset_loader = self.create_data_loader(train_dataset)
        self.test_dataset_loader = self.create_data_loader(test_dataset)
        self.device = options['device']
        self.quiet = options['quiet']
        self.model = model
        #
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def solve_epochs(self, round_i, record_grads=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs
        # acc, loss, comp
        if record_grads:
            stats = self.model.solve_epochs_record_grad(round_i=round_i, client_id=self.id, data_loader=self.train_dataset_loader, optimizer=self.optimizer, num_epochs=num_epochs, hide_output=self.quiet)
        else:
            stats = self.model.solve_epochs(round_i=round_i, client_id=self.id, data_loader=self.train_dataset_loader,
                                            optimizer=self.optimizer, num_epochs=num_epochs, hide_output=self.quiet)
        bytes_w = self.model.model_bytes
        bytes_r = self.model.model_bytes
        flop_stats = {'id': self.id, 'bytes_w': bytes_w, 'comp': stats['comp'], 'bytes_r': bytes_r}
        return stats, flop_stats, self.get_parameters_list()

    def create_data_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.num_batch_size, shuffle=True)

    def get_parameters_list(self):
        p = self.model.get_parameters_list()
        return p

    def set_parameters_list(self, params_list):
        self.model.set_parameters_list(params_list)