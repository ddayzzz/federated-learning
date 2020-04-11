from flmod.utils.torch_utils import get_flat_params_from, set_flat_params_to, get_flat_grad
from flmod.utils.flops_counter import get_model_complexity_info
import torch.nn as nn
import tqdm
import torch


class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """

    def __init__(self, model, criterion, eval_criterion, optimizer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval_criterion = eval_criterion
        self.num_epoch = options['num_epochs']
        self.device = options['device']
        self.verbose = True
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, options['input_shape'], device=self.device)

    def get_model_params(self):
        """
        get parameter values
        :return:
        """
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss += self.criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def local_train(self, train_dataloader, round_i, client_id):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        self.model.train()
        with tqdm.trange(self.num_epoch) as t:
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

                    train_loss += loss.item()
                    train_acc += correct
                    train_total += target_size
                    if self.verbose and (batch_idx % 10 == 0):
                        # 纯数值
                        t.set_postfix(loss=loss.item())

            local_solution = self.get_flat_model_params()
            # param_dict = {"norm": torch.norm(local_solution).item(),
            #               "max": local_solution.max().item(),
            #               "min": local_solution.min().item()}
            comp = self.num_epoch * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total}
            # return_dict.update(param_dict)
            return local_solution, return_dict

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # print("test")
                # from IPython import embed
                # embed()
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = self.eval_criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item()
                test_total += y.size(0)

        return test_acc, test_loss

    def save(self, path):
        torch.save(self.get_model_params(), path)