# tensorflow 版本的配置文件
import argparse
DATASETS = ['mnist', 'synthetic', 'shakespeare', 'brats2018', 'nist', 'sent140', 'omniglot', 'femnist']
TRAINERS_TONAMES = {'fedavg': 'Server', 'fedprox': 'Server', 'feddane': 'Server', 'maml': 'Server', 'fedmeta': 'FedMetaBaseServer', 'fedmeta2': 'FedMetaBaseServer'}
TRAINERS = TRAINERS_TONAMES.keys()
# 模型的参数, 这里的模型都不可复用故而直接写死
MODEL_PARAMS = {
    'sent140.stacked_lstm': (25, 2, 100, None),  # seq_len, num_classes, n_hidden, emb_arr=None(这个之前是默认值)
    'nist.mclr': (26,),  # num_classes
    'mnist.mclr': (10,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'mnist.mclr_maml': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, ),  # num_classes
    'omniglot.cnn': (5, ),  # num_classes
    'femnist.cnn': (62, 28),  # num_classes, image_size
    'femnist.cnn2': (62, 28),  # num_classes, image_size 测试用的
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
                        help='hide client\'s output; only show eval results',
                        action='store_true')
    parser.add_argument('--result_prefix',
                        help='保存结果的前缀路径',
                        type=str,
                        default='./result')
    parser.add_argument('--drop_rate',
                        help='number of epochs when clients train on data;',
                        type=float,
                        default=0.0)
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
    parser.add_argument('--optimizer',
                        help='设置优化器',
                        type=str,
                        default='gd',
                        choices=['gd', 'adam', 'rmsprop'])
    # FedAvg Scheme
    parser.add_argument('--scheme',
                        help='Scheme 1;Scheme 2;Transformed scheme 2',
                        type=str,
                        default='')
    return parser


def get_eval_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir',
                        help='指定已经保存结果目录, 可以加载相关的 checkpoints',
                        type=str,
                        default='')
    return parser.parse_args()

def add_dynamic_options(argparser):
    # 获取对应的 solver 的名称
    params = argparser.parse_known_args()[0]
    algo = params.algo
    if algo in ['maml']:
        argparser.add_argument('--num_fine_tune', help='number of fine-tune', type=int, default=0)
    elif algo in ['fedmeta', 'fedmeta2']:
        argparser.add_argument('--meta_algo', help='使用的元学习算法, 默认 maml', type=str, default='maml', choices=['maml', 'reptile', 'meta_sgd'])
        argparser.add_argument('--outer_lr', help='更新元学习中的外部学习率', type=float, required=True)
        argparser.add_argument('--meta_num_fine_tune', type=int, default=5)
    elif algo in ['fedprox']:
        argparser.add_argument('--mu',
                            help='mu',
                            type=float,
                            default=0.0)
    return argparser
