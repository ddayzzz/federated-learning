import torch
import numpy as np
import tqdm
from flmod.solvers.fedbase import BaseFedarated
from flmod.models.models import choose_model_criterion
from flmod.optimizers.gd import GradientDescend
from flmod.workers.workers import Worker
from flmod.clients.base_client import BaseClient


class Client(BaseClient):

    def __init__(self, id, worker: Worker, batch_size: int, criterion, train_dataset, test_dataset):
        super(Client, self).__init__(id, worker, batch_size, criterion, train_dataset, test_dataset)
        # 这里没什么不同


class MAMLWorker(Worker):

    def __init__(self, model, criterion, optimizer, options):
        """
        每一个客户端等价于
        :param model:
        :param criterion:
        :param optimizer:
        :param options:
        """
        super(MAMLWorker, self).__init__(model, criterion, optimizer, options)

    def local_train(self, num_epochs, support_query_dataloaders, round_i, client_id):
        # self.model.train()
        support_loader, query_loader = support_query_dataloaders
        with tqdm.trange(num_epochs, disable=self.hide_output) as t:

            outer_loss = torch.tensor(0., device=self.device)
            accuracy = torch.tensor(0., device=self.device)

            for epoch in t:
                t.set_description(f'Client: {client_id}, Round: {round_i + 1}, Epoch :{epoch + 1}')
                for batch_idx, (x, y) in enumerate(train_dataloader):

                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer.zero_grad()
                    pred = self.model(x)

                    loss = self.criterion(pred, y)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                    self.optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()

                    target_size = y.size(0)
                    # TODO 一般的损失函数会进行平均(mean), 但是这里不需要, 一种做法是指定损失函数仅仅用 sum, 但是考虑到pytorch中的损失函数默认为mean,故这里进行了些修改
                    single_batch_loss = loss.item() * target_size
                    train_loss += single_batch_loss
                    train_acc += correct
                    train_total += target_size
                    if self.verbose and (batch_idx % 10 == 0):
                        # 纯数值, 这里使用平均的损失
                        t.set_postfix(mean_loss=loss.item())

            local_solution = self.get_flat_model_params()
            # 计算模型的参数值
            param_dict = {"norm": torch.norm(local_solution).item(),
                          "max": local_solution.max().item(),
                          "min": local_solution.min().item()}
            comp = num_epochs * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total}
            return_dict.update(param_dict)
            return local_solution, return_dict



class MAML(BaseFedarated):

    def __init__(self, options, all_data_info):
        model, crit = choose_model_criterion(options=options)
        self.optimizer = GradientDescend(model.parameters(), lr=options['lr'])
        # self.q = options['q_coef']
        # suffix = f'q[{self.q}]'
        super(MAML, self).__init__(options=options, model=model, dataset=all_data_info, optimizer=self.optimizer, criterion=crit, append2metric='_debug', worker=MAMLWorker, client=Client)
        # 前三百用于 train
        self.held_out_for_train = 300

    def select_clients(self, round, num_clients=20):
        # 这里使用 unifrom
        np.random.seed(round)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.clients[:self.held_out_for_train], num_clients, replace=False).tolist()

    def generate_task_wise_data(self, clients):
        while True:
            support_data_batch = torch.tensor(0, dtype=torch.float32, device=)

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i + 1}')
            # eval on test
            # if (round_i + 1) % self.eval_on_test_every_round == 0:
            #     self.test_latest_model_on_evaldata(round_i)
            # # eval on train
            # if (round_i + 1) % self.eval_on_train_every_round == 0:
            #     self.test_latest_model_on_traindata(round_i)
            selected_clients = self.select_clients(round=round_i, num_clients=self.clients_per_round)
            solutions, stats = self.local_train(selected_clients, round_i=round_i)
            self.metrics.extend_commu_stats(round_i, stats)

            # update global weights
            self.latest_model = self.aggregate(solutions)

            if (round_i + 1) % self.save_every_round == 0:
                self.save_model(round_i)
                self.metrics.write()

        self.test_latest_model_on_traindata(self.num_rounds)
        self.test_latest_model_on_evaldata(self.num_rounds)
        self.metrics.write()