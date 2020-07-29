# ref: https://github.com/Merofine/UNet2D_BraTs/blob/master/metrics.py
import numpy as np
from hausdorff import hausdorff_distance
import torch
import torch.nn.functional as F


def batch_iou(output, target):
    """
    batch 版本的 IOU
    :param output: 网络的输出
    :param target:
    :return:
    """
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:,0,:,:]
    target = target[:,0,:,:]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def mean_iou(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target, smooth=1e-5):
    """
    Dice 系数
    :param output: 任何一种形式, 返回的都是累加值. 如果是 torch.Tensor 则进行 sigmoid 激活
    :param target: 同 output 的格式
    :param smooth:
    :return: 标量, 输出整体的累加和
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)


def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return  (intersection + smooth) / \
           (output.sum() + smooth)


def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)


# TODO 计算的结果不对
def dice_coef_channel_wise(output, target):
    """
    按照 channel 的维度输出 dice coef. 这里的全部是没有经过平均的数值.
    :param output: [N, C, H, W]. 网络的输出
    :param target:
    :param wise:
    :return: [C], 在 batch 的维度上平均一下
    """
    # TODO 这种形式有bug, 不知道怎么回事
    n, c, h, w = output.shape
    out = np.zeros([c])
    for channel in range(c):
        dc_sum = 0.0
        for batch in range(n):
            dc_sum += dice_coef(output[batch, channel, :, :], target[batch, channel, :, :])
        output[channel] = dc_sum
    return out


def sensitivity_channel_wise(output, target):
    outputs = []
    # 一般 C 代表 GT 的维度
    for i in range(output.shape[1]):
        one_batch = []
        for b in range(output.shape[0]):
            hd = sensitivity(output[b, i, :, :], target[b, i, :, :])
            one_batch.append(hd)  # 标量
        outputs.append(sum(one_batch))
    d = np.stack(outputs)  # 向量
    return d


def hausdorff_distance_channel_wise(output, target):
    outputs = []
    # 一般 C 代表 GT 的维度
    for i in range(output.shape[1]):
        one_batch = []
        for b in range(output.shape[0]):
            hd = hausdorff_distance(output[b, i, :, :], target[b, i, :, :])
            one_batch.append(hd)  # 标量
        outputs.append(sum(one_batch))
    d = np.stack(outputs)  # 向量
    return d


def ppv_channel_wise(output, target):
    outputs = []
    # 一般 C 代表 GT 的维度
    for i in range(output.shape[1]):
        one_batch = []
        for b in range(output.shape[0]):
            hd = ppv(output[b, i, :, :], target[b, i, :, :])
            one_batch.append(hd)  # 标量
        outputs.append(sum(one_batch))
    d = np.stack(outputs)  # 向量
    return d


if __name__ == '__main__':
    a = np.ones([10, 3, 20, 20])
    b = np.zeros([10, 3, 20, 20])
    dc = dice_coef(a, b)
    print(dc)
