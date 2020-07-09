import numpy as np
import importlib
import tensorflow as tf
import os
import random
# 复用读取数据的 api
from flmod.utils.worker_utils import read_data_pkl
from flearn.config import DATASETS, TRAINERS_TONAMES, MODEL_PARAMS
from flearn.config import base_options, add_dynamic_options


def read_options():
    parser = base_options()
    parser = add_dynamic_options(parser)
    parsed = parser.parse_args()
    options = parsed.__dict__
    # 设置种子
    os.environ['PYTHONHASHSEED'] = str(options['seed'])
    np.random.seed(1 + options['seed'])
    tf.set_random_seed(12 + options['seed'])
    random.seed(123 + options['seed'])


    # 读取数据集
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)


    # 加载选择的 solver 类. 这里和 pytorch 不一样. 数据集有对应的模型名称
    model_path = '%s.%s.%s.%s' % ('flearn', 'models', dataset_name, options['model'])
    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # 训练器
    trainer_path = 'flearn.trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS_TONAMES[options['algo']])

    # 定义对应网络的参数 dataset.model_name
    model_options = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]
    options['model_options'] = model_options
    # 打印参数
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> 参数:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, learner, trainer_class, dataset_name, sub_data


def main():
    tf.logging.set_verbosity(tf.logging.WARN)
    # 数据的文件始终在其父目录
    dataset_prefix = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # 解析参数
    options, learner, trainer_class, dataset_name, sub_data = read_options()

    train_path = os.path.join(dataset_prefix, 'dataset', dataset_name, 'data', 'train')
    test_path = os.path.join(dataset_prefix, 'dataset', dataset_name, 'data', 'train')

    all_data_info = read_data_pkl(train_path, test_path, sub_data=sub_data)
    # 调用solver
    trainer = trainer_class(params=options, learner=learner, dataset=all_data_info)
    trainer.train()


if __name__ == '__main__':
    main()
