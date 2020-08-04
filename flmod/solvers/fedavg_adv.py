from flmod.solvers.fedbase_adv import BaseFedaratedAdvanced, BaseClient
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader


class UsingAllDataClient(BaseClient):

    def __init__(self, id, train_dataset, test_dataset, options, optimizer, model):
        super(UsingAllDataClient, self).__init__(id, train_dataset, test_dataset, options, optimizer, model)
        # 定义客户端执行的操作的
        self.all_dataset = ConcatDataset([train_dataset, test_dataset])
        self.all_dataset_loader = DataLoader(self.all_dataset, batch_size=self.num_batch_size)

    def create_data_loader(self, dataset):
        return None

    def solve_epochs(self, round_i, record_grads=None, num_epochs=None):
        # 总是在所有的数据集上执行
        if num_epochs is None:
            num_epochs = self.num_epochs
        # acc, loss, comp
        stats = self.model.solve_epochs(round_i=round_i, client_id=self.id, data_loader=self.all_dataset_loader, optimizer=self.optimizer, num_epochs=num_epochs, hide_output=self.quiet)
        bytes_w = self.model.model_bytes
        bytes_r = self.model.model_bytes
        flop_stats = {'id': self.id, 'bytes_w': bytes_w, 'comp': stats['comp'], 'bytes_r': bytes_r}
        return stats, flop_stats, self.get_parameters_list()


class FedAvgAdv(BaseFedaratedAdvanced):

    def __init__(self, options, all_data_info, model_obj):
        """
        这个类的不同之处在于分开了测试客户端和训练客户端
        :param options:
        :param all_data_info:
        :param model_obj:
        """
        self.use_all_data = options['use_all_data']
        if self.use_all_data:
            super(FedAvgAdv, self).__init__(options=options, model=model_obj, read_dataset=all_data_info,
                                            append2metric=None, client_class=UsingAllDataClient)
        else:
            super(FedAvgAdv, self).__init__(options=options, model=model_obj, read_dataset=all_data_info,
                                            append2metric=None)
        self.split_train_validation_test_clients()

    def select_clients(self, round_i, num_clients):
        num_clients = min(num_clients, self.num_clients)
        np.random.seed(round_i)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.train_clients, num_clients, replace=False).tolist()

    def aggregate(self, solns, num_samples):
        return self.aggregate_parameters_weighted(solns, num_samples)

    def eval_on(self, round_i, clients, use_test_data=False, use_train_data=False, use_val_data=False):
        if not self.use_all_data:
            return super(FedAvgAdv, self).eval_on(round_i, clients, use_test_data, use_train_data, use_val_data)

        df = pd.DataFrame(columns=['client_id', 'mean_acc', 'mean_loss', 'num_samples'])
        # 设置网络
        self.model.set_parameters_list(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for c in clients:
            stats = self.model.test(c.all_dataset_loader)

            tot_corrects.append(stats['sum_corrects'])
            num_samples.append(stats['num_samples'])
            losses.append(stats['sum_loss'])
            #
            df = df.append({'client_id': c.id, 'mean_loss': stats['loss'], 'mean_acc': stats['acc'],
                            'num_samples': stats['num_samples'], }, ignore_index=True)

        # ids = [c.id for c in self.clients]
        # groups = [c.group for c in self.clients]
        mean_loss = sum(losses) / sum(num_samples)
        mean_acc = sum(tot_corrects) / sum(num_samples)
        #
        fn, on = 'eval_on_all_at_round_{}.csv'.format(round_i), 'all'
        #
        if not self.quiet:
            print(f'Round {round_i}, eval on "{on}" dataset mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        self.metrics.update_eval_stats(round_i, df, filename=fn, on_which=on, other_to_logger={'acc': mean_acc, 'loss': mean_loss})

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')

            selected_clients = self.select_clients(round_i=round_i, num_clients=self.clients_per_round)

            solns, num_samples = self.solve_epochs(round_i, clients=selected_clients)


            self.latest_model = self.aggregate(solns, num_samples)
            # eval on test
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.eval_on(use_test_data=True, round_i=round_i, clients=self.test_clients)

            if (round_i + 1) % self.eval_on_train_every_round == 0:
                self.eval_on(use_train_data=True, round_i=round_i, clients=self.train_clients)

            if (round_i + 1) % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()

        self.metrics.write()