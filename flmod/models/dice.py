import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 1.0  # TODO 作为参数
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def bchw_dice_coeff(input, target):
    """
    计算几个 batch 之间的 dice coefficient
    :param input: [B, 1, H, W]
    :param target: [B, 1, H, W]
    :return: dc, 仅仅sum
    """
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    eps = 1.0
    # TODO 可以作为整体进行计算
    for i, c in enumerate(zip(input, target)):
        inter = torch.dot(c[0].view(-1), c[1].view(-1))
        union = torch.sum(c[0]) + torch.sum(c[1]) + eps
        t = (2 * inter.float() + eps) / union.float()
        s = s + t
    return s