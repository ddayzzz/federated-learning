import abc
import torch
import pickle
import codecs
import tqdm
import numpy as np
from flmod.utils.flops_counter import get_model_complexity_info
from flmod.utils.torch_utils import get_flat_params_from, set_flat_params_to, get_flat_grad, model_parameters_shape_list, from_flatten_to_parameter


class BaseWorker(abc.ABC):

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
        self.client_id = options['client_name']
        self.verbose = True
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, options['input_shape'], device=self.device, input_type=options.get('input_type'))
        # 模型和对应的shape
        self.model_shape_info = model_parameters_shape_list(model)
        self.hide_output = options['quiet']

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

    def set_model_params(self, from_buffer):
        """
        将参数赋值给当前的模型(模拟: 将参数发送给客户端的过程)
        :param model_params_dict: 参数字典
        :return:
        """
        weight = pickle.loads(codecs.decode(from_buffer.encode(), "base64"))
        loaded = torch.load(weight, map_location=self.device)
        # state_dict = self.model.state_dict()
        # for key, value in state_dict.items():
        #     state_dict[key] = weight[key]
        self.model.load_state_dict(loaded)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def to_model_params(self, flat_params):
        return from_flatten_to_parameter(self.model_shape_info, flat_params)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    @abc.abstractmethod
    def local_train(self, num_epochs, data_loader, round_i):
        pass

    @abc.abstractmethod
    def local_test(self, data_loader):
        pass


class ClassificationWorker(BaseWorker):

    def __init__(self, model, criterion, optimizer, options):
        super(ClassificationWorker, self).__init__(model, criterion, optimizer, options)

    def local_train(self, num_epochs, data_loader, round_i):
        """
        训练模型
        :param num_epochs: epoch 数量
        :param data_loader: 训练集的加载器
        :param round_i: 第几个 round? (用于显示)
        :param client_id: 客户端 id (用于显示)
        :return:更新后的参数(Tensor), stat(Dict: comp->total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS); loss->损失函数, acc->准确率)
        """
        self.model.train()
        with tqdm.trange(num_epochs, disable=self.hide_output) as t:
            train_loss = train_acc = train_total = 0
            for epoch in t:
                t.set_description(f'Client: {self.client_id}, Round: {round_i + 1}, Epoch :{epoch + 1}')
                for batch_idx, (x, y) in enumerate(data_loader):
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

            local_solution = self.get_model_params_dict()
            # 计算模型的参数值
            param_dict = {"norm": torch.norm(local_solution).item(),
                          "max": local_solution.max().item(),
                          "min": local_solution.min().item()}
            comp = num_epochs * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total,
                           "num_samples": train_total}
            return_dict.update(param_dict)
            return local_solution, return_dict

    def local_test(self, data_loader):
        """
        在测试集合上运行数据
        :param test_dataloader:
        :return: 准确判断的个数, 总体的损失函数值
        """
        self.model.eval()
        test_all_loss = test_all_acc = test_total = 0.
        with torch.no_grad():
            for x, y in data_loader:
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
        stats = {
            'num_samples': test_total,
            'acc': test_all_acc / test_total,
            'loss': test_all_loss / test_total
        }
        return stats


from distributed.models.metrics import ppv, dice_coef, hausdorff_distance, iou_score, sensitivity
from torch import optim


class MRISegWorker(BaseWorker):

    def __init__(self, model, criterion, optimizer, options):
        super(MRISegWorker, self).__init__(model, criterion, optimizer, options)
        # 添加 lr 调节器. https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        # 这里改为 min 模式, 如果当前的 loss 停止减少, 那么适当减少 lr. new_lr = lr * factor
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.8)

    def local_train(self, num_epochs, data_loader, round_i):
        """
        训练模型
        :param num_epochs: epoch 数量
        :param data_loader: 训练集的加载器
        :param round_i: 第几个 round? (用于显示)
        :param client_id: 客户端 id (用于显示)
        :return:更新后的参数(Tensor), stat(Dict: comp->total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS); loss->损失函数, acc->准确率)
        """
        self.model.train()
        with tqdm.trange(num_epochs, disable=self.hide_output) as t:
            train_loss = 0.0
            train_total = 0
            for epoch in t:
                t.set_description(f'Client: {self.client_id}, Round: {round_i + 1}, Epoch :{epoch + 1}')
                for batch_idx, (x, y) in enumerate(data_loader):
                    # from IPython import embed
                    # embed()
                    x, y = x.to(self.device), y.to(self.device)
                    masks_pred = self.model(x)
                    loss = self.criterion(masks_pred, y)
                    train_loss += loss.item() * y.size(0)

                    self.optimizer.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    self.optimizer.step()
                    train_total += y.size(0)
                    if self.verbose and (batch_idx % 10 == 0):
                        # 纯数值, 这里使用平均的损失
                        t.set_postfix(mean_loss=loss.item())

            local_solution = self.get_model_params_dict()
            comp = num_epochs * train_total * self.flops
            return_dict = {"comp": comp,
                           "bce_dice_loss": train_loss / train_total,
                           "num_samples": train_total,
                           "lr": self.get_lr()}
            return local_solution, return_dict

    def local_test(self, data_loader):
        """
        在测试集合上运行数据
        :param test_dataloader:
        :return: 准确判断的个数, 总体的损失函数值
        """
        self.model.eval()
        n_val = len(data_loader)
        test_total = 0
        bce_dice_loss = 0.0
        # 分割的评价指标
        dice_coeff_sum = np.zeros([3])
        hd95_sum = np.zeros([3])
        ppv_sum = np.zeros([3])
        sensitivity_sum = np.zeros([3])
        iou_sum = np.zeros([3])
        for x, y in data_loader:
            # print("test")
            # from IPython import embed
            # embed()
            x, y_to_device = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                pred = self.model(x)
            # loss
            bce_dice_loss += self.criterion(pred, y_to_device).item()
            # 其他的度量指标, mask 为 0-1的图像, 对应的区域为1, pred 激活且 >1
            pred_for_metric = (torch.sigmoid(pred).data > 0.5).float().cpu().numpy()
            y_np = y.numpy()
            # 这些值是sum, 且会自动使用
            # dc = dice_coef_channel_wise(pred_for_metric, y_np)
            # hd_95 = hausdorff_distance_channel_wise(pred_for_metric, y_np)
            # ppv = ppv_channel_wise(pred_for_metric, y_np)
            # sensitivity = sensitivity_channel_wise(pred_for_metric, y_np)
            # 计算 Dice系数
            # pred = torch.sigmoid(pred)
            # pred = (pred > 0.5).float()
            # dice_coeff_sum += dice_coeff(pred, y).item()  # dice_coeff 是在batch上平均过的
            # 这些是在 batch 的 sum
            for one_pred, one_mask in zip(pred_for_metric, y_np):
                for c, (one_channel_pred, one_channel_mask) in enumerate(zip(one_pred, one_mask)):
                    hd95_sum[c] += hausdorff_distance(one_channel_pred, one_channel_mask)
                    dice_coeff_sum[c] += dice_coef(one_channel_pred, one_channel_mask)
                    ppv_sum[c] += ppv(one_channel_pred, one_channel_mask)
                    iou_sum[c] += iou_score(one_channel_pred, one_channel_mask)
                    sensitivity_sum[c] += sensitivity(one_channel_pred, one_channel_mask)
            test_total += y.size(0)
        mean_bce_loss = bce_dice_loss / n_val
        stats = {
            'num_samples': test_total,
            'bce_dice_loss': mean_bce_loss,
        }
        for item, item_name in zip([dice_coeff_sum, hd95_sum, ppv_sum, sensitivity_sum, iou_sum],
                                   ['dice_coeff', 'hd95', 'ppv', 'sensitivity', 'iou_score']):
            for c in range(len(item)):
                # 这些是元素数量的求和, 没有像 loss 一样 mean 过
                stats[item_name + '_channel' + str(c)] = item[c] / test_total
        # 如果没有验证集的话就不管了
        self.lr_scheduler.step(mean_bce_loss)
        return stats