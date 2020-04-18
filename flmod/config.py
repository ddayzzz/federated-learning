# GLOBAL PARAMETERS

DATASETS = ['mnist', 'synthetic', 'shakespeare', 'brats2018']
TRAINERS = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'fedprox_non_grad': 'FedProxNonGrad'}
OPTIMIZERS = TRAINERS.keys()


class ModelConfig(object):
    def __init__(self):
        pass

    @staticmethod
    def config(dataset, model):
        dataset = dataset.split('_')[0]
        if dataset == 'mnist':
            if model == 'logistic' or model == '2nn':
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
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))

    @staticmethod
    def get_entire_dataset(dataset, options):
        if dataset == 'mnist':
            from flmod.dataset.mnist.get_datset import get_dataset
            if options['dataset'] == 'mnist_user1000_niid_0_keep_10_train_9':
                # 这个数据不使用 index
                return None, None
            if options['model'] == 'logistic':
                # 需要扁平化
                return get_dataset(flatten_input=True)
            else:
                return get_dataset(flatten_input=False)
        elif dataset in ['synthetic', 'shakespeare', 'brats2018']:
            return None, None
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))

    @staticmethod
    def dataset_config(dataset, options):
        cfg = {'data_wrapper': None, 'worker': None}
        if dataset == 'shakespeare':
            from flmod.dataset.shakespeare.shakespeare import Shakespeare
            cfg['dataset_wrapper'] = Shakespeare
        elif dataset == 'brats2018':
            from flmod.dataset.brats2018.brats2018_dataset import BRATS2018Dataset
            cfg['dataset_wrapper'] = BRATS2018Dataset
        return cfg

model_settings = ModelConfig()
