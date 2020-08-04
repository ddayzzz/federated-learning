from flmod.solvers.fedbase import BaseFedarated


class FedAvg(BaseFedarated):

    def __init__(self, options, all_data_info, model_obj):
        super(FedAvg, self).__init__(options=options, model=model_obj, read_dataset=all_data_info, append2metric=None)

    def aggregate(self, solns, num_samples):
        return self.aggregate_parameters_weighted(solns, num_samples)

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')

            selected_clients = self.select_clients(round_i=round_i, num_clients=self.clients_per_round)

            solns, num_samples = self.solve_epochs(round_i, clients=selected_clients)


            self.latest_model = self.aggregate(solns, num_samples)
            # eval on test
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.eval_on(use_test_data=True, round_i=round_i, clients=self.clients)

            if (round_i + 1) % self.eval_on_train_every_round == 0:
                stats = self.eval_on(use_train_data=True, round_i=round_i, clients=self.clients)

            if (round_i + 1) % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()

        self.metrics.write()