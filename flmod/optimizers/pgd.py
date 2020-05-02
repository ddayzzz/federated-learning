from torch.optim import Optimizer
from torch.optim.optimizer import required
import torch


class PerturbedGradientDescent(Optimizer):

    def __init__(self, params, lr=required, mu=0.0, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if mu < 0.0:
            raise ValueError("Invalid mu value: {}".format(mu))
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super(PerturbedGradientDescent, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PerturbedGradientDescent, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            # momentum = group['momentum']
            # dampening = group['dampening']
            # nesterov = group['nesterov']
            mu = group['mu']
            w_old = group['w_old']
            for p, w_old_p in zip(group['params'], w_old):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - dampening, d_p)
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf
                if w_old is not None:
                    d_p.add_(mu, p.data - w_old_p.data)
                p.data.add_(-group['lr'], d_p)

        return loss

    def set_mu(self, mu):
        for param_group in self.param_groups:
            param_group['mu'] = mu

    def get_mu(self):
        return self.param_groups[0]['mu']

    def set_old_weights(self, old_weights):
        for param_group in self.param_groups:
            param_group['w_old'] = old_weights

    def adjust_learning_rate(self, round_i):
        lr = self.lr * (0.5 ** (round_i // 30))
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def soft_decay_learning_rate(self):
        self.lr *= 0.99
        for param_group in self.param_groups:
            param_group['lr'] = self.lr

    def inverse_prop_decay_learning_rate(self, round_i):
        for param_group in self.param_groups:
            param_group['lr'] = self.lr/(round_i+1)

    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        return self.param_groups[0]['lr']
