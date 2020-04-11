import torch
from flmod.solvers.fedbase import BaseFedarated
from flmod.models.models import choose_model_criterion


class FedAvg(BaseFedarated):

    def __init__(self, options, all_data_info):
        model, crit, eval_crit = choose_model_criterion(options=options)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=options['lr'], momentum=0.5)
        super(FedAvg, self).__init__(options=options, model=model, dataset=all_data_info, optimizer=self.optimizer,
                                     criterion=crit, eval_criterion=eval_crit)
        self.num_rounds = options['num_rounds']
        self.clients_per_round = options['clients_per_round']
        self.save_every_round = options['save_every']
        self.eval_on_test_every_round = options['eval_every']
        self.eval_on_train_every_round = options['eval_train_every']

    def train(self):
        for round_i in range(self.num_rounds):
            local_weights, local_losses = [], []
            print(f'>>> Global Training Round : {round_i + 1}')

            selected_clients = self.select_clients(round=round_i, num_clients=self.clients_per_round)
            solutions, stats = self.local_train(selected_clients, round_i=round_i)
            self.metrics.extend_commu_stats(round_i, stats)

            # update global weights
            self.latest_model = self.aggregate(solutions)
            # eval on test
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.test_latest_model_on_evaldata(round_i)
            if (round_i + 1) % self.eval_on_train_every_round == 0:
                self.test_latest_model_on_traindata(round_i)
            if (round_i + 1) % self.save_every_round == 0:
                self.save_model(round_i)

        self.test_latest_model_on_traindata(self.num_rounds)
        self.test_latest_model_on_evaldata(self.num_rounds)
        self.metrics.write()