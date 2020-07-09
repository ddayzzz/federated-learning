import json
import os
import glob
import pickle
import numpy as np
np.random.seed(6)

ROOT = os.path.dirname(__file__)
TEST_DIR = '%s/data/test' % ROOT
TRAIN_DIR = '%s/data/train' % ROOT

for d in ['%s/data' % ROOT, TEST_DIR, TRAIN_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)

TRAIN_TARGET = '{}/train_9_test_1.pkl'.format(TRAIN_DIR)
TEST_TARGET = '{}/train_9_test_1.pkl'.format(TEST_DIR)

GROUND_TRUTH = os.sep.join((ROOT, 'data', 'origin', 'ground_truth'))
FLAIR = os.sep.join((ROOT, 'data', 'origin', 'flair'))


def load_config(json_file, max_size):
    num_sliced = 155
    train_rate = 0.9
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    cfg = json.load(open(json_file))

    for institution, patient_ids in cfg.items():
        # 读取对应的 img, ann
        ips = []
        for pid in patient_ids:
            # 直接保存对用的路径, 这个相当于 index
            input_files = [os.path.split(x)[-1] for x in glob.glob(os.sep.join((FLAIR, f'{pid}*.npy')), recursive=False)[:max_size]]
            input_files = [x[:x.rfind('.')] for x in input_files]
            ips += input_files
        # 分割对应的数据, 这列打乱顺序再分可好?
        np.random.shuffle(ips)
        train_len = int(len(ips) * train_rate)
        test_len = len(ips) - train_len
        train_idx = ips[:train_len]
        test_idx = ips[train_len + 1:]
        train_data['users'].append(institution)
        train_data['user_data'][institution] = {'x_index': train_idx}
        test_data['users'].append(institution)
        test_data['user_data'][institution] = {'x_index': test_idx}
        train_data['num_samples'].append(train_len)
        test_data['num_samples'].append(test_len)
    print('num of institutions', len(cfg))
    print('training data distribution', train_data['num_samples'])
    print('test data distribution', test_data['num_samples'])
    with open(TRAIN_TARGET, 'wb') as fp:
        pickle.dump(train_data, fp)
    with open(TEST_TARGET, 'wb') as fp:
        pickle.dump(test_data, fp)


load_config(json_file=ROOT + '/hgg_config.json', max_size=None)