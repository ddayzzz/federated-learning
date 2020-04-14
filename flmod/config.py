# GLOBAL PARAMETERS

DATASETS = ['mnist']
TRAINERS = {'fedavg': 'FedAvg', 'fedprox': 'FedProx'}
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
            shakespeare = {'stacked_lstm': {'seq_len': 80, 'emb_dim': 80, 'num_hidden': 256}
                           }
            return shakespeare[model]
        elif dataset == 'synthetic':
            return {'input_shape': 60, 'num_class': 10}
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))

    @staticmethod
    def get_entire_dataset(dataset, options):
        if dataset == 'mnist':
            from flmod.dataset.mnist.get_datset import get_dataset
            merged = options['dataset'] == 'mnist_user1000_niid_0_keep_10_train_9'
            if options['model'] == 'logistic':
                # 需要扁平化
                return get_dataset(flatten_input=True, merge_train_test=merged)
            else:
                return get_dataset(flatten_input=False, merge_train_test=merged)

        else:
            raise ValueError('Not support dataset {}!'.format(dataset))

model_settings = ModelConfig()
