from flmod.utils.torch_utils import get_flat_params_from, set_flat_params_to, get_flat_grad, model_parameters_shape_list, from_flatten_to_parameter
from flmod.utils.flops_counter import get_model_complexity_info
import tqdm
import numpy as np
import torch
from flmod.models.dice import bchw_dice_coeff
from flmod.utils.torch_utils import get_flat_grad_from_sparse


class Worker(object):

    def __init__(self, model, criterion, optimizer, options):
        """
        基本的 Worker, 完成客户端和模型之间的交互, 适用于串行化的模型
        :param model: 模型
        :param criterion: 模型评估器
        :param optimizer: 共享的优化器
        :param options:
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = options['device']
        self.verbose = True
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, options['input_shape'], device=self.device, input_type=options.get('input_type'))
        self.model_shape_info = model_parameters_shape_list(model)
        self.hide_output = True if options['quiet'] == 0 else False

    def get_model_params_dict(self):
        """
        获得网络模型的参数
        :return:
        """
        state_dict = self.model.state_dict()
        return state_dict

    def get_model_params_list(self):
        """
        列表形式的参数,按照顺序
        :return:
        """
        p = self.model.parameters()
        return list(p)

    def set_model_params(self, model_params_dict: dict):
        """
        将参数赋值给当前的模型(模拟: 将参数发送给客户端的过程)
        :param model_params_dict: 参数字典
        :return:
        """
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def to_model_params(self, flat_params):
        return from_flatten_to_parameter(self.model_shape_info, flat_params)

    def get_flat_grads(self, dataloader, mini_batchsize=None, grad_processor=get_flat_grad):
        """
        获取梯度, 结构扁平化
        :param dataloader: 数据加载器
        :param use_alldata: 是否使用全部的数据. 如果是,则安州某个批次的大小计算梯度
        :return:
        """
        dataset = dataloader.dataset
        num_samples = len(dataset)
        self.model.train()
        self.optimizer.zero_grad()
        loss, total_num = 0.0, 0
        if mini_batchsize is None or (num_samples < mini_batchsize):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss += self.criterion(pred, y) * y.size(0)
                total_num += y.size(0)
            loss /= total_num
            flat_grads = grad_processor(loss, self.model.parameters(), create_graph=True)
            return flat_grads.cpu().detach(), total_num
        # 我觉得也可以通过将i
        flatted_grads = torch.zeros(self.params_num, dtype=torch.float32)
        rds = min(int(num_samples / mini_batchsize), 4)
        for i in range(rds):
            # 自行累计样本 batch
            x = []
            y = []
            for i in range(mini_batchsize * i, mini_batchsize * (i + 1)):
                img, label = dataset[i]
                x.append(img)
                y.append(label)
            x = torch.from_numpy(np.stack(x, axis=0)).to(self.device)
            y = torch.from_numpy(np.stack(y, axis=0)).to(self.device)
            pred = self.model(x)
            # 平均的 loss
            batch_size = y.size(0)
            loss += self.criterion(pred, y) * batch_size
            #
            total_num += batch_size
            grads = grad_processor(loss, self.model.parameters(), create_graph=True)
            flatted_grads = flatted_grads + grads.cpu().detach()
        # 这个梯度除以运行的batch的次数, 因为loss是平均的
        # flatted_grads = flatted_grads / rds
        flatted_grads = flatted_grads / total_num
        return flatted_grads, total_num


    def local_train(self, num_epochs, train_dataloader, round_i, client_id):
        """
        训练模型
        :param num_epochs: epoch 数量
        :param train_dataloader: 训练集的加载器
        :param round_i: 第几个 round? (用于显示)
        :param client_id: 客户端 id (用于显示)
        :return:更新后的参数(Tensor), stat(Dict: comp->total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS); loss->损失函数, acc->准确率)
        """
        self.model.train()
        with tqdm.trange(num_epochs, disable=self.hide_output) as t:
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

    def local_test(self, test_dataloader):
        """
        在测试集合上运行数据
        :param test_dataloader:
        :return: 准确判断的个数, 总体的损失函数值
        """
        self.model.eval()
        test_all_loss = test_all_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # print("test")
                # from IPython import embed
                # embed()
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()
                size = y.size(0)
                test_all_acc += correct.item()
                test_all_loss += loss.item() * size  # 需要乘以数据的维度, 最后除以总数量得到平均的损失
                test_total += size

        return test_all_acc, test_total, test_all_loss

    def save(self, path):
        torch.save(self.get_model_params_dict(), path)


class SegmentationWorker(Worker):

    def __init__(self, model, criterion, optimizer, options):
        super(SegmentationWorker, self).__init__(model, criterion, optimizer, options)

    def local_train(self, num_epochs, train_dataloader, round_i, client_id):
        self.model.train()
        with tqdm.trange(num_epochs, disable=self.hide_output) as t:
            train_loss = dcs = train_total = 0
            for epoch in t:
                t.set_description(f'Client: {client_id}, Round: {round_i + 1}, Epoch :{epoch + 1}')
                for batch_idx, (x, y) in enumerate(train_dataloader):
                    # from IPython import embed
                    # embed()
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer.zero_grad()
                    pred = self.model(x)

                    # if torch.isnan(pred.max()):
                    #     from IPython import embed
                    #     embed()

                    loss = self.criterion(pred, y)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                    self.optimizer.step()
                    # 分割使用的 DC
                    mask = (pred > 0.5).float().detach()
                    dcs += bchw_dice_coeff(mask, y).item()
                    target_size = y.size(0)
                    # TODO 一般的损失函数会进行平均(mean), 但是这里不需要, 一种做法是指定损失函数仅仅用 sum, 但是考虑到pytorch中的损失函数默认为mean,故这里进行了些修改
                    single_batch_loss = loss.item() * target_size
                    train_loss += single_batch_loss
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
                           "acc": dcs / train_total}
            return_dict.update(param_dict)
            return local_solution, return_dict

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = test_total = 0.
        dc = 0.0
        with torch.no_grad():
            for x, y in test_dataloader:
                # print("test")
                # from IPython import embed
                # embed()
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                mask = (pred > 0.5).float().detach()
                dc += bchw_dice_coeff(mask, y).item()

                size = y.size(0)
                test_loss += loss.item() * size  # 需要乘以数据的维度, 最后除以总数量得到平均的损失
                test_total += size

        return dc, test_total, test_loss


class StackedLSTMWorker(Worker):

    def __init__(self, model, criterion, optimizer, options):
        super(StackedLSTMWorker, self).__init__(model, criterion, optimizer, options)

    def get_flat_grads(self, dataloader, mini_batchsize=50, grad_processor=get_flat_grad_from_sparse):
        grads = super(StackedLSTMWorker, self).get_flat_grads(dataloader, mini_batchsize, grad_processor=grad_processor)
        # 必须处理 spares 的梯度
        return grads


class LRDecayWorker(Worker):

    def __init__(self, model, criterion, optimizer, options):
        """
        这里实现的是 ICLR,2020 Li Xiang等人提出的改版的 FedAvg. 与原始的 McMahan 等人提出的 FedAvg 不同,
        :param model:
        :param criterion:
        :param optimizer:
        :param options:
        """
        super(LRDecayWorker, self).__init__(model, criterion, optimizer, options)

    def local_train(self, num_epochs, train_dataloader, round_i, client_id):
        """
        按照 ICLR2020 给出的模式训练,重点是 epoch 为何要乘以10?
        :param num_epochs:
        :param train_dataloader:
        :param round_i:
        :param client_id:
        :return:
        """
        # 论文中每个 epoch 使用一个batch的数据.. 这里有不一样的地方
        self.model.train()
        with tqdm.trange(num_epochs, disable=self.hide_output) as t:
            train_loss = train_acc = train_total = 0
            for epoch in t:
                t.set_description(f'Client: {client_id}, Round: {round_i + 1}, Epoch :{epoch + 1}')
                # 这里有个不同的地方, 就是 one-shot 训练
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
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
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



def choose_worker(options):
    model = options['model']
    algo = options['algo']
    if model == 'unet':
        return SegmentationWorker
    elif model == 'stacked_lstm':
        return StackedLSTMWorker
    elif model in ['logistic', 'cnn']:
        workers = {'fedavg_schemes': LRDecayWorker}
        return workers.get(algo, Worker)
    else:
        raise ValueError('No suitable worker!')