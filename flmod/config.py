# GLOBAL PARAMETERS
import argparse
DATASETS = ['mnist', 'synthetic', 'shakespeare', 'brats2018', 'omniglot', 'femnist']
TRAINERS = {'fedavg': 'FedAvg',
            'fedprox': 'FedProx',
            'fedprox_non_grad': 'FedProxNonGrad',
            'fedavg_schemes': 'FedAvgSchemes',
            'maml': 'MAML',
            'fedmeta': 'FedMeta',
            'fedavg_adv': 'FedAvgAdv'}

OPTIMIZERS = TRAINERS.keys()
MODEL_CONFIG = {
    'mnist.logistic': {'out_dim': 10, 'in_dim': 784},
    'femnist.cnn': {'num_classes': 62, 'image_size': 28},
    'omniglot.cnn': {'num_classes': 5, 'image_size': 28}

}

class ModelConfig(object):
    def __init__(self):
        pass

    @staticmethod
    def config(dataset, model):
        dataset = dataset.split('_')[0]
        if dataset == 'mnist':
            if model == 'logistic' or model == '2nn':  # 线性模型或者两层的MLP
                return {'input_shape': 784, 'num_class': 10}
            else:
                return {'input_shape': (1, 28, 28), 'num_class': 10}
        elif dataset == 'cifar10':
            return {'input_shape': (3, 32, 32), 'num_class': 10}
        elif dataset == 'sent140':
            sent140 = {'bag_dnn': {'num_class': 2},
                       'stacked_lstm': {'seq_len': 25, 'num_class': 2, 'num_hidden': 100},
                       'stacked_lstm_no_embeddings': {'seq_len': 25, 'num_class': 2, 'num_hidden': 100}
                       }
            return sent140[model]
        elif dataset == 'shakespeare':
            shakespeare = {'stacked_lstm': {'seq_len': 80, 'num_classes': 80, 'num_hidden': 256, 'input_shape': [80], 'input_type': 'index'}  # 句子的长度(转为对应的词汇的 index), 分类的数量, hidden
                           }  # 后面两个用于测试
            return shakespeare[model]
        elif dataset == 'synthetic':
            return {'input_shape': 60, 'num_class': 10}
        elif dataset == 'brats2018':
            brats2018 = {'unet': {'num_classes': 1, 'input_shape': [1, 128, 128], 'num_channels': 1, 'bilinear': True}}
            return brats2018[model]
        elif dataset == 'omniglot':
            return {'input_shape': (1, 28, 28)}
        elif dataset == 'femnist':
            return {'num_classes': 62, 'image_size': 28, 'input_shape': (1, 28 * 28)}
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))

    @staticmethod
    def get_entire_dataset(dataset, options):
        if dataset == 'mnist':
            # 目前使用直接序列化后的数据
            return None, None
            # from flmod.dataset.mnist.get_datset import get_dataset
            # if options['dataset'] in ['mnist_user1000_niid_0_keep_10_train_9', 'mnist_all_data_0_random_niid_org', 'mnist_user1000_niid_0_keep_10_train_9_no_flatten']:
            #     # 这个数据不使用 index
            #     return None, None
            # 下面的几种情况则是利用的 index 作为保存的数据格式
            # if options['model'] == 'logistic':
            #     # 需要扁平化
            #     return get_dataset(flatten_input=True)
            # else:
            #     return get_dataset(flatten_input=False)
        elif dataset in ['synthetic', 'shakespeare', 'brats2018', 'omniglot', 'femnist']:
            return None, None
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))

    @staticmethod
    def dataset_config(dataset, options):
        from flmod.utils.data_utils import MiniDataset  # 标准的数据封装
        cfg = {'dataset_wrapper': MiniDataset, 'worker': None}
        if dataset == 'shakespeare':
            from dataset.shakespeare.shakespeare import Shakespeare
            cfg['dataset_wrapper'] = Shakespeare
        elif dataset == 'brats2018':
            from dataset.brats2018.brats2018_dataset import BRATS2018Dataset
            cfg['dataset_wrapper'] = BRATS2018Dataset
        return cfg

model_settings = ModelConfig()


def base_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        required=True)
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='logistic')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--device',
                        help='device',
                        default='cpu:0',
                        type=str)
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_on_test_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--eval_on_train_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--eval_on_validation_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--save_every',
                        help='save global model every ____ rounds;',
                        type=int,
                        default=50)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=20)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--quiet',
                        help='仅仅显示结果的代码',
                        type=int,
                        default=0)
    parser.add_argument('--result_prefix',
                        help='保存结果的前缀路径',
                        type=str,
                        default='./result')
    parser.add_argument('--train_val_test',
                        help='数据集是否以训练集、验证集和测试集的方式存在',
                        action='store_true')
    parser.add_argument('--result_dir',
                        help='指定已经保存结果目录, 可以加载相关的 checkpoints',
                        type=str,
                        default='')
    # TODO 以后支持 之家加载 leaf 目录里的数据
    parser.add_argument('--data_format',
                        help='加载的数据格式, json 为 Leaf以及Li T.等人定义的格式, 默认为 pkl',
                        type=str,
                        default='pkl')

    # FedAvg Scheme
    parser.add_argument('--scheme',
                        help='Scheme 1;Scheme 2;Transformed scheme 2',
                        type=str,
                        default='')
    return parser


def add_dynamic_options(argparser):
    # 获取对应的 solver 的名称
    params = argparser.parse_known_args()[0]
    algo = params.algo
    # if algo in ['maml']:
    #     argparser.add_argument('--q_coef', help='q', type=float, default=0.0)
    if algo in ['fedmeta']:
        argparser.add_argument('--meta_algo', help='使用的元学习算法, 默认 maml', type=str, default='maml',
                               choices=['maml', 'reptile', 'meta_sgd'])
        argparser.add_argument('--outer_lr', help='更新元学习中的外部学习率', type=float, required=True)
        argparser.add_argument('--meta_inner_step', type=int, default=0)
    elif algo == 'fedprox':
        argparser.add_argument('--drop_rate',
                            help='number of epochs when clients train on data;',
                            type=float,
                            default=0.0)
        argparser.add_argument('--mu',
                            help='mu',
                            type=float,
                            default=0.0)
    elif algo == 'fedavg_adv':
        argparser.add_argument('--use_all_data', action='store_true', default=False)
    return argparser
