# tensorflow 版本的配置文件
import argparse
DATASETS = ['mnist', 'synthetic', 'shakespeare', 'brats2018', 'nist', 'sent140', 'omniglot']
TRAINERS_TONAMES = {'fedavg': 'Server', 'fedprox': 'Server', 'feddane': 'Server', 'maml': 'Server'}
TRAINERS = TRAINERS_TONAMES.keys()
# 模型的参数, 这里的模型都不可复用故而直接写死
MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (26,),  # num_classes
    'mnist.mclr': (10,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, ),  # num_classes
    'omniglot.cnn': (5, ),  # num_classes
}

def base_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=TRAINERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_0_equal_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='mclr')
    # parser.add_argument('--wd',
    #                     help='weight decay parameter;',
    #                     type=float,
    #                     default=0.001)
    # parser.add_argument('--device',
    #                     help='device',
    #                     default='cpu:0',
    #                     type=str)
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_every',
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
                        help='hide output; only show eval results',
                        type=int,
                        default=0)
    parser.add_argument('--drop_rate',
                        help='number of epochs when clients train on data;',
                        type=float,
                        default=0.0)
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
    if algo in ['maml']:
        argparser.add_argument('--num_fine_tune', help='number of fine-tune', type=int, default=0)
    elif algo in ['fedprox']:
        argparser.add_argument('--mu',
                            help='mu',
                            type=float,
                            default=0.0)
    return argparser
