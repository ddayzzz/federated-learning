import abc
import tqdm
import numpy as np
# from flmod.clients.base_client import BaseClient
from flmod.utils.data_utils import MiniDataset
import torch
from torch import nn
from flmod.utils.flops_counter import get_model_complexity_info


class BaseModel(abc.ABC):

    def __init__(self, model: nn.Module, criterion, options):
        self.options = options
        self.device = options['device']
        self.criterion = criterion.to(options['device'])
        self.model = model.to(self.device)
        #
        input_shape = model.input_shape
        input_type = model.input_type if hasattr(model, 'input_type') else None
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, input_shape, input_type=input_type, device=self.device)

    def solve_epochs(self, round_i, client_id, data_loader, optimizer, num_epochs, hide_output: bool = False):
        device = self.device
        criterion = self.criterion
        self.model.train()

        with tqdm.trange(num_epochs, disable=hide_output) as t:
            train_loss = train_acc = train_total = 0
            for epoch in t:
                t.set_description(f'Client: {client_id}, Round: {round_i}, Epoch :{epoch}')
                for batch_idx, (X, y) in enumerate(data_loader):
                    # from IPython import embed
                    # embed()
                    X, y = X.to(device), y.to(device)

                    optimizer.zero_grad()
                    pred = self.model(X)

                    # if torch.isnan(pred.max()):
                    #     from IPython import embed
                    #     embed()

                    loss = criterion(pred, y)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                    optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()

                    target_size = y.size(0)
                    # TODO 一般的损失函数会进行平均(mean), 但是这里不需要, 一种做法是指定损失函数仅仅用 sum, 但是考虑到pytorch中的损失函数默认为mean,故这里进行了些修改
                    single_batch_loss = loss.item() * target_size
                    train_loss += single_batch_loss
                    train_acc += correct
                    train_total += target_size
                    if (batch_idx % 10 == 0):
                        # 纯数值, 这里使用平均的损失
                        t.set_postfix(mean_loss=loss.item())

        # 这一步非常关键, 清除优化器指向的全局模型的梯度
        optimizer.zero_grad()

        comp = num_epochs * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total,
                       'sum_loss': train_loss,
                       'sum_corrects': train_acc,
                       'num_samples': train_total}
        return return_dict

    def solve_epochs_record_grad(self, round_i, client_id, data_loader, optimizer, num_epochs, hide_output: bool = False):
        device = self.device
        criterion = self.criterion
        self.model.train()

        grads = []
        for param in optimizer.param_groups['params']:
            if not param.requires_grad:
                continue
            grads.append(torch.zeros_like(param.data))

        with tqdm.trange(num_epochs, disable=hide_output) as t:
            train_loss = train_acc = train_total = 0
            for epoch in t:
                t.set_description(f'Client: {client_id}, Round: {round_i}, Epoch :{epoch}')
                for batch_idx, (X, y) in enumerate(data_loader):
                    # from IPython import embed
                    # embed()
                    X, y = X.to(device), y.to(device)

                    optimizer.zero_grad()
                    pred = self.model(X)

                    # if torch.isnan(pred.max()):
                    #     from IPython import embed
                    #     embed()

                    loss = criterion(pred, y)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                    optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()

                    target_size = y.size(0)
                    # TODO 一般的损失函数会进行平均(mean), 但是这里不需要, 一种做法是指定损失函数仅仅用 sum, 但是考虑到pytorch中的损失函数默认为mean,故这里进行了些修改
                    single_batch_loss = loss.item() * target_size
                    train_loss += single_batch_loss
                    train_acc += correct
                    train_total += target_size
                    if (batch_idx % 10 == 0):
                        # 纯数值, 这里使用平均的损失
                        t.set_postfix(mean_loss=loss.item())

                    # 从这里计算累积的梯度
                    i = 0
                    for param in optimizer.param_groups['params']:
                        if not param.requires_grad:
                            continue
                        grads[i].add_(param.grad.data, alpha=target_size)
                        i += 1

        # 这一步非常关键, 清除优化器指向的全局模型的梯度
        optimizer.zero_grad()
        # 缩放梯度
        for g in grads:
            g.mul_(1 / train_total)

        comp = num_epochs * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total,
                       'num_samples': train_total,
                       'sum_loss': train_loss,
                       'sum_corrects': train_acc,
                       'grads': grads}
        return return_dict

    def get_parameters_list(self) -> list:
        """
        得到网络的参数, 使用 detach 共享数据源, 切记不得使用 in-place 操作
        :return:
        """
        with torch.no_grad():
            ps = [p.data.clone().detach() for p in self.model.parameters()]
        return ps

    def set_parameters_list(self, params_list: list):
        """
        设置网络参数
        :param params_list: 参数列表为 tensor
        :return:
        """
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), params_list):
                # 设置参数的值
                p.data.copy_(d.data)

    def test(self, data_loader):
        self.model.eval()
        train_loss = train_acc = train_total = 0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(data_loader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()

                target_size = y.size(0)
                # TODO 一般的损失函数会进行平均(mean), 但是这里不需要, 一种做法是指定损失函数仅仅用 sum, 但是考虑到pytorch中的损失函数默认为mean,故这里进行了些修改
                single_batch_loss = loss.item() * target_size
                train_loss += single_batch_loss
                train_acc += correct
                train_total += target_size
        return_dict = {"loss": train_loss / train_total,
                       "acc": train_acc / train_total,
                       'sum_loss': train_loss,
                       'sum_corrects': train_acc,
                       'num_samples': train_total}
        return return_dict

    @abc.abstractmethod
    def create_optimizer(self, params):
        raise NotImplementedError

    @property
    def dataset_wrapper(self):
        return MiniDataset


class ModelWithMetaLearn(BaseModel):

    from learn2learn.algorithms.maml import MAML
    from learn2learn.algorithms.meta_sgd import MetaSGD

    def __init__(self, model: nn.Module, criterion, options):
        super(ModelWithMetaLearn, self).__init__(model, criterion, options)
        self.maml = self.MAML(lr=self.options['lr'], model=model)

    def set_parameters_list(self, params_list: list):
        """
        :param params_list:
        :return:
        """
        with torch.no_grad():
            for p, d in zip(self.maml.parameters(), params_list):
                # 设置参数的值
                p.data.copy_(d.data)

    def get_parameters_list(self) -> list:
        with torch.no_grad():
            ps = [p.data.clone().detach() for p in self.maml.parameters()]
        return ps

    def count_correct(self, preds, targets):
        _, predicted = torch.max(preds, 1)
        correct = predicted.eq(targets).sum().item()
        return correct

    def solve_meta_one_epoch(self, round_i, client_id, support_data_loader, query_data_loader, hide_output: bool = False):
        self.model.train()
        # 克隆之后, learn 为中间节点, 本身不带有梯度
        learner = self.maml.clone()
        # 记录相关的信息
        support_loss, support_correct, support_num_sample = [], [], []
        for batch_idx, (x, y) in enumerate(support_data_loader):
            x, y = x.to(self.device), y.to(self.device)
            num_sample = y.size(0)
            pred = learner(x)
            loss = self.criterion(pred, y)
            # 评估
            correct = self.count_correct(pred, y)
            # 写入相关的记录, 这份 loss 是平均的
            support_loss.append(loss.item())
            support_correct.append(correct)
            support_num_sample.append(num_sample)
            # 计算 loss 关于当前参数的导数, 并更新目前网络的参数(回传到 model)
            learner.adapt(loss)

        # 此使的参数基于 query
        query_loss, query_correct, query_num_sample = [], [], []
        loss_sum = 0.0
        for batch_idx, (x, y) in enumerate(query_data_loader):
            x, y = x.to(self.device), y.to(self.device)
            num_sample = y.size(0)
            pred = learner(x)
            loss = self.criterion(pred, y)
            # batch_sum_loss
            # 评估
            correct = self.count_correct(pred, y)
            # 写入相关的记录, 这份 loss 是平均的
            query_loss.append(loss.item())
            query_correct.append(correct)
            query_num_sample.append(num_sample)
            #
            loss_sum += loss * num_sample

        spt_sz = np.sum(support_num_sample)
        qry_sz = np.sum(query_num_sample)
        mean_loss = loss_sum / qry_sz
        # 这个优化器的唯一作用是清除网络多余的梯度信息
        # self.optimizer.zero_grad()
        mean_loss.backward()
        # 获取此使的梯度, 这个梯度为一个 tensor
        grads = [p.grad.data.clone().detach() for p in self.maml.parameters()]
        for p in self.maml.parameters():
            if p.grad is not None:
                p.grad.zero_()
        #

        comp = (spt_sz + qry_sz) * self.flops
        return {
            'grads': grads,
            'support_loss_sum': np.dot(support_loss, support_num_sample),
            'query_loss_sum': np.dot(query_loss, query_num_sample),
            'support_correct': np.sum(support_correct),
            'query_correct': np.sum(query_correct),
            'support_num_samples': spt_sz,
            'query_num_samples': qry_sz,
            'comp': comp
        }

    def test_meta_one_epoch(self, train_loader, test_loader):
        # 这里觉得必须是有 adapt 的过程
        self.model.eval()
        learner = self.maml.clone()
        # 清空目前指向的参数的梯度信息
        # 记录相关的信息
        support_loss, support_correct, support_num_sample = 0.0, 0, 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            num_sample = y.size(0)
            pred = learner(x)
            loss = self.criterion(pred, y)
            # 评估
            correct = self.count_correct(pred, y)
            # 写入相关的记录, 这份 loss 是平均的
            support_loss += loss.item() * num_sample
            support_correct += correct
            support_num_sample += num_sample
            # 计算 loss 关于当前参数的导数, 并更新目前网络的参数
            learner.adapt(loss)

        # 此使的参数基于 query
        query_loss, query_correct, query_num_sample = 0.0, 0, 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                num_sample = y.size(0)
                pred = learner(x)
                loss = self.criterion(pred, y)
                # batch_sum_loss
                # 评估
                correct = self.count_correct(pred, y)
                # 写入相关的记录, 这份 loss 是平均的
                query_loss += loss.item() * num_sample
                query_correct += correct
                query_num_sample += num_sample

        return support_loss, support_correct, support_num_sample, query_loss, query_correct, query_num_sample