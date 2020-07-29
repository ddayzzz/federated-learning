import os
import torch
from torch import nn, optim
import random
import numpy as np
import argparse
from tornado import ioloop
from distributed.models.configs import get_model
from distributed.clients.client import FederatedClient
from distributed.utils.data_model_loader import DataModel
from dataset.data_reader import get_distributed_data_cfgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--client_name', type=str, required=True)
    parser.add_argument('--server', type=str, required=True)
    parser.add_argument('--model', help='name of model;', type=str, default='mclr')
    parser.add_argument('--lr', help='learning rate for inner solver;', type=float, default=3e-4)
    parser.add_argument('--seed', help='seed for randomness;', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu:0')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true', default=False)

    return parser.parse_args()


def get_dataset_wrapper(dataset_name, dataset_cfg):
    if dataset_name == 'brats2018':
        from dataset.brats2018.new_brats2018dataset import BRATS2018AllModDatasetInsWise
        train, test = BRATS2018AllModDatasetInsWise(cfg=dataset_cfg, data_key='train'), BRATS2018AllModDatasetInsWise(cfg=dataset_cfg, data_key='test')
    else:
        raise NotImplementedError

    return train, test


def get_worker_wrapper(dataset_name, model, options):
    opt = get_optimizer(optimizer=options['optimizer'], model=model, options=options)
    if dataset_name == 'brats2018':
        from distributed.models.losses import BCEDiceLoss
        from distributed.clients.workers import MRISegWorker
        cri = BCEDiceLoss().to(options['device'])
        return MRISegWorker(model=model, criterion=cri, optimizer=opt, options=options)
    else:
        cri = nn.CrossEntropyLoss(reduction='mean').to(options['device'])
        from distributed.clients.workers import ClassificationWorker
        return ClassificationWorker(model=model, criterion=cri, optimizer=opt, options=options)


def get_optimizer(optimizer, model, options):
    if optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['weight_decay'])
    elif optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=options['lr'], momentum=options['momentum'], weight_decay=options['weight_decay'], nesterov=options['nesterov'])
    elif optimizer == 'rmsprop':
        opt = optim.RMSprop(model.parameters(), lr=options['lr'], momentum=options['momentum'], weight_decay=options['weight_decay'])
    else:
        raise NotImplementedError
    return opt


def main():
    params = parse_args()
    options = params.__dict__
    # 设置种子
    os.environ['PYTHONHASHSEED'] = str(options['seed'])
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    random.seed(1234 + options['seed'])
    if options['device'].startswith('cuda'):
        torch.cuda.manual_seed_all(123 + options['seed'])
        torch.backends.cudnn.deterministic = True  # cudnn
    # 打印相关参数
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> 参数:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    # 读取数据集
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx + 1:]
    else:
        dataset_name, sub_data = options['dataset'], None

    # 初始化
    model, model_opt = get_model(dataset_name=dataset_name, net=options['model'])
    model.to(options['device'])
    options.update(model_opt)
    data_cfg = get_distributed_data_cfgs(client_id=options['client_name'], data_name=dataset_name, sub_name=sub_data)
    print('>>> 使用的数据集文件: ', data_cfg)
    train_dataset, test_dataset = get_dataset_wrapper(dataset_name, dataset_cfg=data_cfg)
    worker = get_worker_wrapper(dataset_name, model, options)
    datamodel = DataModel(worker=worker, train_dataset=train_dataset, test_dataset=test_dataset)
    client = FederatedClient(datamodel=datamodel)
    # 连接服务器
    ws_url = options['server']
    client.connect(ws_url, auto_reconnet=True, reconnet_interval=10)

    try:
        ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        client.close()

if __name__ == '__main__':
    main()