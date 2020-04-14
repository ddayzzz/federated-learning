import torch
import time
import numpy as np
from flmod.solvers.fedbase import BaseFedarated
from flmod.models.models import choose_model_criterion
from flmod.optimizers.pgd import PerturbedGradientDescent
from flmod.clients.base_client import BaseClient
from flmod.utils.data_utils import DatasetSplit
from flmod.models.worker import Worker
import tqdm


class FedProxWorker(Worker):

    def __init__(self, model, criterion, eval_criterion, optimizer, options):
        """
        与普通的 Worker 不同, 一次客户端的计算返回的累计的梯度信息
        :param model:
        :param criterion:
        :param eval_criterion:
        :param optimizer:
        :param options:
        """
        super(FedProxWorker, self).__init__(model, criterion, eval_criterion, optimizer, options)

    def local_train(self, num_epochs, train_dataloader, round_i, client_id):
        self.model.train()
        with tqdm.trange(num_epochs) as t:
            train_loss = train_acc = train_total = 0
            for epoch in t:
                t.set_description(f'Client: {client_id}, Round: {round_i + 1}, Epoch :{epoch + 1}')
                for batch_idx, (x, y) in enumerate(train_dataloader):
                    # from IPython import embed
                    # embed()
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer.zero_grad()
                    pred = self.model(x)

                    if torch.isnan(pred.max()):
                        from IPython import embed
                        embed()

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

class FedProx(BaseFedarated):

    def __init__(self, options, all_data_info):
        model, crit, eval_crit = choose_model_criterion(options=options)
        self.optimizer = PerturbedGradientDescent(model.parameters(), lr=options['lr'], momentum=0.5, mu=options['mu'])
        super(FedProx, self).__init__(options=options, model=model, dataset=all_data_info, optimizer=self.optimizer,
                                     criterion=crit, eval_criterion=eval_crit)
        self.drop_rate = options['drop_rate']
        self.num_rounds = options['num_rounds']
        self.clients_per_round = options['clients_per_round']
        self.save_every_round = options['save_every']
        self.eval_on_test_every_round = options['eval_every']
        self.eval_on_train_every_round = options['eval_train_every']
        self.num_epochs = options['num_epochs']
        self.num_clients = len(self.clients)
        # 保存当前的模型的参数, 在聚合之前的

    def select_clients(self, round, num_clients=20):
        """
        这里只输出客户端的编号和被选择的客户端
        :param round:
        :param num_clients:
        :return:
        """
        num_clients = min(num_clients, self.num_clients)  # 选择的客户端
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        selected_clients_indx = np.random.choice(self.num_clients, num_clients, replace=False)
        return selected_clients_indx

    def local_train(self, selected_clients, round_i):
        raise NotImplementedError

    def update_optimizer_weights(self):
        # 运行万所有的数据之后才进行 - w, 所以worker中的每一个 batch 不需要进行 - w 的操作;只要在聚合完成所有数据的
        oldw = self.worker.to_model_params(self.latest_model)
        self.optimizer.set_old_weights(old_weights=oldw)

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i + 1}')
            if (round_i + 1) % self.eval_on_train_every_round == 0:
                self.calc_client_grads(round_i)
            selected_clients_indices = self.select_clients(round=round_i, num_clients=self.clients_per_round)
            activated_clients_indx = np.random.choice(selected_clients_indices, round(self.clients_per_round * (1 - self.drop_rate)), replace=False)
            # 能够顺利完成任务客户端, 其余的不在 active 但是被选择的客户端被视为 starggle
            self.update_optimizer_weights()
            solns = []
            stats = []
            for selected_i in selected_clients_indices:
                c = self.clients[selected_i]
                c.set_flat_model_params(self.latest_model)
                if selected_i in activated_clients_indx:
                    # 运行相同的 num_epoch
                    soln, stat = c.local_train(round_i, self.num_epochs)
                else:
                    soln, stat = c.local_train(round_i, num_epochs=np.random.randint(low=1, high=self.num_epochs))
                solns.append(soln)
                stats.append(stat)

            self.metrics.extend_commu_stats(round_i, stats)

            # update global weights
            self.latest_model = self.aggregate(solns)

            # eval on test
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.test_latest_model_on_evaldata(round_i)

            if (round_i + 1) % self.save_every_round == 0:
                self.save_model(round_i)

        # self.test_latest_model_on_traindata(self.num_rounds)
        # self.test_latest_model_on_evaldata(self.num_rounds)
        self.metrics.write()