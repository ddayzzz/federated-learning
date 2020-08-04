from flmod.models.base_models import BaseModel
from torch import nn, optim


class Model(BaseModel):

    def __init__(self, options, model_cfgs: dict):
        if options['model'] == 'logistic':
            from flmod.models.mnist.logistic import Logistic
            model = Logistic(**model_cfgs)
        else:
            raise ValueError(options['model'])
        super(Model, self).__init__(model=model, options=options, criterion=nn.CrossEntropyLoss(reduction='mean'))

    def create_optimizer(self, params):
        if self.options['algo'] == 'fedprox':
            from flmod.optimizers.pgd import PerturbedGradientDescent
            return PerturbedGradientDescent(params, mu=self.options['mu'], lr=self.options['lr'])
        else:
            return optim.SGD(params, lr=self.options['lr'], momentum=0.5)

