from flmod.models.base_models import ModelWithMetaLearn
from torch import nn, optim


class Model(ModelWithMetaLearn):

    def __init__(self, options, model_cfgs: dict):
        if options['model'] == 'cnn':
            from flmod.models.femnist.cnn import CNNModel
            model = CNNModel(**model_cfgs)
        else:
            raise ValueError(options['model'])
        super(Model, self).__init__(model=model, options=options, criterion=nn.CrossEntropyLoss(reduction='mean'))

    def create_optimizer(self, params):
        if self.options['algo'] == 'fedmeta':
            return None
        else:
            return optim.Adam(params, lr=self.options['lr'])

