import abc
import io
import torch
import pickle
import codecs
from torch.utils.data import Dataset, DataLoader
from distributed.clients.workers import BaseWorker


class DataModel(abc.ABC):

    def __init__(self, worker: BaseWorker, train_dataset: Dataset, test_dataset: Dataset, eval_dataset: Dataset=None):
        self.worker = worker
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset

    def init(self, options):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=options['batch_size'], shuffle=True,
                                           num_workers=options['num_loader_worker'],
                                           pin_memory=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=options['batch_size'], shuffle=False,
                                          num_workers=options['num_loader_worker'],
                                          pin_memory=True)
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=options['batch_size'], shuffle=False,
                                              num_workers=options['num_loader_worker'],
                                              pin_memory=True)
            self.use_eval = True
        else:
            self.use_eval = False

    def train_one_round(self, round_i, num_epoch):
        local_solution_dict, return_dict = self.worker.local_train(num_epochs=num_epoch, data_loader=self.train_dataloader, round_i=round_i)
        new_cpu_torch = dict()
        for k, v in local_solution_dict.items():
            # 转换为 cpu 的 torch
            new_cpu_torch[k] = v.detach().cpu()
        # 这里需要处理成 str
        buffer = io.BytesIO()
        torch.save(new_cpu_torch, buffer)
        buffer.seek(0)
        x = codecs.encode(pickle.dumps(buffer), "base64").decode()
        return x, return_dict

    def run_metric_on_test(self):
        stats = self.worker.local_test(data_loader=self.test_dataloader)
        return stats

    def run_metric_on_val(self):
        assert self.use_eval
        stats = self.worker.local_test(data_loader=self.eval_dataloader)
        return stats




