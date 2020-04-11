from torch.optim.optimizer import Optimizer


class PerturbedGradientDescent(Optimizer):

    def __init__(self, params, lr=0.001, mu=0.01):
        """
        \nabla h_k(w;w_t) =\nabla F_k(w) + \mu*(w-w_t). w_t 是客户端的参数; w 是最近一次 epoch 的梯度
        :param params:
        :param lr:
        :param mu:
        """
        defaults = dict(lr=lr, mu=mu)
        super(PerturbedGradientDescent, self).__init__(params=params, defaults=defaults)

    def __setstate__(self, state):
        super(PerturbedGradientDescent, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups():
            lr = group['lr']
            mu = group['mu']
            for p in group['params']:
                # 检查参数是否需要梯度
                if p.grad is None:
                    continue
                # 已经计算好了梯度
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

if __name__ == '__main__':

    opt = PerturbedGradientDescent(lr=0.1, mu=0.001)
