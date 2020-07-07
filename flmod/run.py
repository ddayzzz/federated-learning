import numpy as np
import argparse
import importlib
import torch
import os
import random
from flmod.utils.worker_utils import read_data_pkl
from flmod.config import DATASETS, TRAINERS
from flmod.config import model_settings
from flmod.config import base_options, add_dynamic_options

def read_options():
    parser = base_options()
    parser = add_dynamic_options(parser)
    parsed = parser.parse_args()
    options = parsed.__dict__
    # 设置种子
    os.environ['PYTHONHASHSEED'] = str(options['seed'])
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    random.seed(1234 + options['seed'])
    if options['device'].startswith('cuda'):
        torch.cuda.manual_seed_all(123 + options['seed'])
        torch.backends.cudnn.deterministic = True  # cudnn


    # 读取数据集
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # 将配置的参数添加到测试文件中
    options.update(model_settings.config(dataset_name, options['model']))

    # 加载选择的 solver 类
    trainer_path = 'flmod.solvers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # 打印参数
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> 参数:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, trainer_class, dataset_name, sub_data


def main():
    prefix = os.path.dirname(__file__)
    if prefix == '':
        prefix = '.'

    # 解析参数
    options, trainer_class, dataset_name, sub_data = read_options()

    train_path = os.path.join(prefix, 'dataset', dataset_name, 'data', 'train')
    test_path = os.path.join(prefix, 'dataset', dataset_name, 'data', 'train')

    all_data_info, (train_dataset, test_datatset) = read_data_pkl(train_path, test_path, sub_data=sub_data), \
                                                    model_settings.get_entire_dataset(dataset_name, options=options)
    all_data_info = list(all_data_info)
    all_data_info.extend([train_dataset, test_datatset, model_settings.dataset_config(dataset_name, options=options)])
    # 调用solver
    trainer = trainer_class(options, all_data_info)
    trainer.train()


if __name__ == '__main__':
    main()
