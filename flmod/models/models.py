import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import math


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
        # 顺序和 parameters 一样
        # pt_latest_w = torch.from_numpy(np.zeros([10, 784], dtype=np.float32))
        # pt_latest_b = torch.from_numpy(np.zeros([10], dtype=np.float32))
        # pt_latest_w = torch.from_numpy(np.load('../weight.npy'))
        # pt_latest_b = torch.from_numpy(np.load('../bias.npy'))
        # self.layer.weight.data.copy_(pt_latest_w)
        # self.layer.bias.data.copy_(pt_latest_b)

    def forward(self, x):
        logit = self.layer(x)
        return logit


class CNN(nn.Module):
    def __init__(self, num_channels: int, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class SimpleCNNMNIST(nn.Module):
    def __init__(self, in_channels=1, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        input_dim = 16 * 4 * 4
        hidden_dims = [128, 84]
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TwoHiddenLayerFc(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(TwoHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_shape, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, out_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class TwoConvOneFc(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(TwoConvOneFc, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class CifarCnn(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(CifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class StackedLSTM(nn.Module):

    def __init__(self, seq_len, num_classes, num_hidden, device):
        super(StackedLSTM, self).__init__()
        self.device = device
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        # 用于将文本数据转换为对应的词向量, sent 的实验中使用训练好的 glove 词向量
        # emv 有一个记录权重的矩阵 [num_vocabulary, 8]. num_vocabulary 是句子中的字符的数量
        # 输入 [*], 索引, 输出 [*, embedding_dim]
        # TODO sparse 参数可以影响梯度的计算, 请注意
        self.embedding_layer = nn.Embedding(num_embeddings=seq_len, embedding_dim=8, sparse=False)
        torch.nn.init.xavier_uniform(self.embedding_layer.weight)
        # 输入: (seq_len, batch, input_size), hx(2,batch,hidden_size)
        # 输出: (seq_len, batch, num_directions * hidden_size), 如果 batch_first == True, 交换 0, 1
        self.stacked_lstm = nn.LSTM(input_size=8,
                                    hidden_size=num_hidden,
                                    num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=self.num_hidden, out_features=num_classes)

    def forward(self, inputs):
        x = inputs
        batch_size = x.size(0)
        x = self.embedding_layer(x)
        h0, c0 = torch.zeros(2, batch_size, self.num_hidden, device=self.device), \
                 torch.zeros(2, batch_size, self.num_hidden, device=self.device)
        # 将 embedding 前任嵌入的数据转换
        x, _ = self.stacked_lstm(x, (h0, c0))
        # 预测是那个人物, 用最后一句话?
        x = x[:, -1, :]
        x = self.fc(x)
        return x


def move_model_to(model, device):
    if not device.startswith('cpu'):
        torch.cuda.set_device(device)
        torch.backends.cudnn.enabled = True
        model.to(device)
    return model


def choose_model_criterion(options):
    """

    :param options:
    :return: model. train criterion, test criterion
    """
    model_name = str(options['model']).lower()
    algo = options['algo']
    device = options['device']
    # TODO 一般是这个
    cri = nn.CrossEntropyLoss(reduction='mean')
    if model_name == 'cnn':
        # 输入的channel, 要求输入的数据为 [B, H, W, C]
        # options['input_shape'] = [1, 28, 28]
        # model = CNN(num_classes=options['num_class'], num_channels=options['input_shape'][0])
        if algo == 'maml':
            from flmod.models.meta_learn.omniglot import ConvolutionalNeuralNetwork
            model = ConvolutionalNeuralNetwork(in_channels=1, out_features=5)
        elif algo == 'fedmeta':
            from flmod.models.femnist.cnn import CNNModel
            model = CNNModel(num_classes=options['num_classes'], image_size=options['image_size'])
        else:
            model = SimpleCNNMNIST()
    elif model_name == 'logistic':
        if algo == 'fedavg_schemes':
            model = Logistic(options['input_shape'], options['num_class'], weight_init=torch.nn.init.zeros_)
        else:
            # 默认使用类似于 tensorflow 的权重初始化方式
            model = Logistic(options['input_shape'], options['num_class'])
    elif model_name == 'stacked_lstm':
        model = StackedLSTM(seq_len=options['seq_len'], num_classes=options['num_classes'], num_hidden=options['num_hidden'], device=options['device'])
    elif model_name == 'unet':
        from flmod.models.unet import UNet
        model = UNet(n_channels=options['num_channels'], n_classes=options['num_classes'], bilinear=options['bilinear'])
        if model.n_classes <= 1:
            # 二分类使用 BCE, 多分类用 CrossEntropy
            cri = nn.BCEWithLogitsLoss(reduction='mean')
    else:
        raise ValueError("Not support model: {}!".format(model_name))
    return move_model_to(model, device=device), cri.to(device)


if __name__ == '__main__':
    model = StackedLSTM(80,80,10,'cpu:0')
    x = np.random.choice(8, 160).astype(np.int64).reshape([-1, 80])
    pred = model(torch.from_numpy(x))
    cri = nn.CrossEntropyLoss(reduction='mean')
    y = torch.from_numpy(np.array([0,5], dtype=np.int64))
    loss = cri(pred, y)
    print(pred.detach().numpy().shape)
    print(loss.item())