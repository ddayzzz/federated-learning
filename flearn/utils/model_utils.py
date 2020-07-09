import json
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def batch_data(data, batch_size, shuffle=True):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    if shuffle:
        # randomly shuffle data
        np.random.seed(100)
        rng_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def batch_data_multiple_iters(data, batch_size, num_iters):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    idx = 0

    for i in range(num_iters):
        if idx+batch_size >= len(data_x):
            idx = 0
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
        batched_x = data_x[idx: idx+batch_size]
        batched_y = data_y[idx: idx+batch_size]
        idx += batch_size
        yield (batched_x, batched_y)

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds'] + 1
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}      
        self.accuracies_on_train = []
        self.accuracies_on_test = []
        self.loss_on_train = []
        self.loss_on_test = []
        self.grad_diff = []
        self.grad_norm = []

    def update_train(self, stats):
        # loss, acc
        num_samples = np.sum(stats[2])
        self.accuracies_on_train.append(np.sum(stats[3])*1.0/num_samples)
        # 注意, loss 输出的 mean 之后的
        self.loss_on_train.append(np.dot(stats[4], stats[2])*1.0 / num_samples)
        return self.accuracies_on_train[-1], self.loss_on_train[-1]

    def update_client(self, rnd, cid, stats):
        # (bytes_w, comp, bytes_r)
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def update_test(self, stats):
        num_samples = np.sum(stats[2])
        # 3 : tot_correct
        # 4: test_loss
        self.accuracies_on_test.append(np.sum(stats[3]) * 1.0 / num_samples)
        # 注意, loss 输出的 mean 之后的, 所以让 loss 乘以对应的样本的数量并求和得到这个客户端所有测试样本的损失再除以总数量
        self.loss_on_test.append(np.dot(stats[4], stats[2]) * 1.0 / num_samples)
        return self.accuracies_on_test[-1], self.loss_on_test[-1]

    def update_grad_info(self, graddiff, gradnorm):
        self.grad_diff.append(graddiff)
        self.grad_norm.append(gradnorm)

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies_on_train'] = self.accuracies_on_train
        metrics['accuracies_on_test'] = self.accuracies_on_test
        metrics['loss_on_train'] = self.loss_on_train
        metrics['loss_on_test'] = self.loss_on_test
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics['drop_percent'] = self.params['drop_percent']
        metrics['grad_diff'] = self.grad_diff
        metrics['grad_norm'] = self.grad_norm
        metrics_dir = os.path.join('out', self.params['dataset'],
                                   'metrics_seed[{}]_opt[{}]_lr[{}]_epochs[{}]_mu[{}]_dp[{}].json'.format(self.params['seed'], self.params['optimizer'],
                                                                        self.params['learning_rate'],
                                                                        self.params['num_epochs'], self.params['mu'], self.params['drop_percent']))

        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)


class Metrics2(object):
    def __init__(self, clients, options, name='', append2suffix=None):
        self.options = options
        num_rounds = options['num_rounds'] + 1
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}

        # Statistics in training procedure
        self.loss_on_train_data = [0] * num_rounds
        self.acc_on_train_data = [0] * num_rounds
        self.gradnorm_on_train_data = [0] * num_rounds
        self.graddiff_on_train_data = [0] * num_rounds

        # Statistics in test procedure
        self.loss_on_eval_data = [0] * num_rounds
        self.acc_on_eval_data = [0] * num_rounds
        # customs
        self.customs_data = dict()
        self.num_rounds = num_rounds
        self.result_path = mkdir(os.path.join('./result', self.options['dataset']))
        # suffix = '{}_sd{}_lr{}_ep{}_bs{}_{}'.format(name,
        #                                             options['seed'],
        #                                             options['lr'],
        #                                             options['num_epoch'],
        #                                             options['batch_size'],
        #                                             'w' if options['noaverage'] else 'a')
        suffix = '{}_sd{}_lr{}_ep{}_bs{}'.format(name,
                                                 options['seed'],
                                                 options['lr'],
                                                 options['num_epochs'],
                                                 options['batch_size'])
        if append2suffix is not None:
            suffix += '_' + append2suffix

        self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                             options['model'], suffix)
        # if options['dis']:
        #     suffix = options['dis']
        #     self.exp_name += '_{}'.format(suffix)
        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)

    def update_commu_stats(self, round_i, stats):
        cid, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']

        self.bytes_written[cid][round_i] += bytes_w
        self.client_computations[cid][round_i] += comp
        self.bytes_read[cid][round_i] += bytes_r

    def extend_commu_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_commu_stats(round_i, stats)

    def update_train_stats(self, round_i, train_stats):
        self.loss_on_train_data[round_i] = train_stats['loss']
        self.acc_on_train_data[round_i] = train_stats['acc']
        self.gradnorm_on_train_data[round_i] = train_stats['gradnorm']
        self.graddiff_on_train_data[round_i] = train_stats['graddiff']

        self.train_writer.add_scalar('train_loss', train_stats['loss'], round_i)
        self.train_writer.add_scalar('train_acc', train_stats['acc'], round_i)
        self.train_writer.add_scalar('gradnorm', train_stats['gradnorm'], round_i)
        self.train_writer.add_scalar('graddiff', train_stats['graddiff'], round_i)

    def update_grads_stats(self, round_i, stats):
        self.gradnorm_on_train_data[round_i] = stats['gradnorm']
        self.graddiff_on_train_data[round_i] = stats['graddiff']
        self.train_writer.add_scalar('gradnorm', stats['gradnorm'], round_i)
        self.train_writer.add_scalar('graddiff', stats['graddiff'], round_i)

    def update_train_stats_only_acc_loss(self, round_i, train_stats):
        self.loss_on_train_data[round_i] = train_stats['loss']
        self.acc_on_train_data[round_i] = train_stats['acc']
        self.train_writer.add_scalar('train_loss', train_stats['loss'], round_i)
        self.train_writer.add_scalar('train_acc', train_stats['acc'], round_i)

    def update_eval_stats(self, round_i, eval_stats):
        self.loss_on_eval_data[round_i] = eval_stats['loss']
        self.acc_on_eval_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def update_custom_scalars(self, round_i, **data):
        for key, scalar in data.items():
            if key not in self.customs_data:
                self.customs_data[key] = [0] * self.num_rounds
            self.customs_data[key][round_i] = scalar
            self.train_writer.add_scalar(key, scalar_value=scalar, global_step=round_i)

    def write(self):
        metrics = dict()

        # String
        metrics['dataset'] = self.options['dataset']
        metrics['num_rounds'] = self.options['num_rounds']
        metrics['eval_every'] = self.options['eval_every']
        # metrics['eval_train_every'] = self.options['eval_train_every']
        metrics['lr'] = self.options['lr']
        metrics['num_epochs'] = self.options['num_epochs']
        metrics['batch_size'] = self.options['batch_size']

        metrics['loss_on_train_data'] = self.loss_on_train_data
        metrics['acc_on_train_data'] = self.acc_on_train_data
        metrics['gradnorm_on_train_data'] = self.gradnorm_on_train_data
        metrics['graddiff_on_train_data'] = self.graddiff_on_train_data

        metrics['loss_on_eval_data'] = self.loss_on_eval_data
        metrics['acc_on_eval_data'] = self.acc_on_eval_data

        # Dict(key=cid, value=list(stats for each round))
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        for key, data in self.customs_data.items():
            metrics[key] = data
        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)
