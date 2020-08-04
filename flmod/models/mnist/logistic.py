from torch import nn
import torch


class Logistic(nn.Module):

    def __init__(self, in_dim, out_dim, weight_init=torch.nn.init.xavier_uniform):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        # 初始化, 这个初始化和 tf.layers.dense 和 tf.get_variable(用于 weight)相同,
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # https://zhuanlan.zhihu.com/p/72853886
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/layers/dense
        weight_init(self.layer.weight)
        self.layer.bias.data.fill_(0)
        self.input_shape = (in_dim, )

    def forward(self, x):
        logit = self.layer(x)
        return logit