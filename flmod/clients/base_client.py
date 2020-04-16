import time
from torch.utils.data import DataLoader


class BaseClient(object):

    def __init__(self, id, worker, batch_size, criterion, train_dataset, test_dataset):
        """
        定义基本的客户端
        :param model:
        """
        self.train_data_loader = DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(test_dataset,
                                           batch_size=batch_size, shuffle=False)

        self.id = id
        self.criterion = criterion
        self.num_train_data = len(self.train_data_loader.dataset)
        self.num_test_data = len(self.test_data_loader.dataset)
        self.worker = worker

    def get_model_params(self):
        """Get model parameters"""
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        """Set model parameters"""
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def get_flat_grads(self):
        """Get model gradient"""
        grad_in_tensor = self.worker.get_flat_grads(self.train_data_loader)
        return grad_in_tensor.cpu().detach().numpy()

    def solve_grad(self):
        """Get model gradient with cost"""
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * self.num_train_data
        bytes_r = self.worker.model_bytes
        stats = {'id': self.id, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}

        grads = self.get_flat_grads()  # Return grad in numpy array

        return (self.num_train_data, grads), stats

    def local_train(self, round_i, num_epochs, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats = self.worker.local_train(num_epochs, self.train_data_loader, round_i, self.id, **kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.id, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (self.num_train_data, local_solution), stats

    def local_test(self, use_eval_data=True):
        """Test current model on local eval data

        Returns:
            1. tot_correct: total # correct predictions
            2. test_samples: int
        """
        if use_eval_data:
            dataloader, ds_size = self.test_data_loader, self.num_test_data
        else:
            dataloader, ds_size = self.train_data_loader, self.num_train_data

        tot_correct, loss = self.worker.local_test(dataloader)

        return tot_correct, ds_size, loss
