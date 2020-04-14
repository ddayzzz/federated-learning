import torch
import numpy as np
import pickle
import os
from tqdm import trange
from torchvision import datasets
cpath = os.path.dirname(__file__)

np.random.seed(6)
outdir = cpath


def generate():
    # Setup directory for train/test data
    train_dataset = datasets.MNIST(cpath, train=True, download=True,
                                   transform=None)

    test_dataset = datasets.MNIST(cpath, train=False, download=True,
                                  transform=None)

    # 这里吧 train 和 test 合并构成数据
    # all_train_test_target = np.concatenate((train_dataset.targets.numpy(), test_dataset.targets))
    train_targets = train_dataset.targets.numpy()
    test_targets = test_dataset.targets.numpy()
    train_targets_index = np.array(len(train_targets))
    test_targets_index = np.array(len(test_targets))
    # [index,类]
    # index_labels = np.stack((np.arange(len(all_train_test_target)), all_train_test_target), axis=0)
    train_index_record = os.path.sep.join((outdir, 'data', 'train', 'user1000_niid_0_keep_10_train_9.pkl'))
    test_index_record = os.path.sep.join((outdir, 'data', 'test', 'user1000_niid_0_keep_10_train_9.pkl'))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # mu = np.mean(tr.data.astype(np.float32), 0)
    # sigma = np.std(mnist.data.astype(np.float32), 0)
    # mnist.data = (mnist.data.astype(np.float32) - mu) / (sigma + 0.001)
    # 所有 MNIST 数据的 index, 按照类别排列
    mnist_idx = []

    for i in trange(10):
        idx = index_labels[0, all_train_test_target == i]
        mnist_idx.append(idx)
    print('每个类拥有的数据量:', [len(v) for v in mnist_idx])
    ###### CREATE USER DATA SPLIT #######
    # Assign 10 samples to each user
    # 目前只记录 index, 这个是按照 MNIST 的标签的递增顺序构建的
    X_idx = [[] for _ in range(1000)]
    y_idx = [[] for _ in range(1000)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(1000):
        for j in range(2):
            l = (user + j) % 10
            # X[user] += mnist_data[l][idx[l]:idx[l] + 5].tolist()
            # y[user] += (l * np.ones(5)).tolist()
            X_idx[user] += mnist_idx[l][idx[l]:idx[l] + 5].tolist()
            y_idx[user] += (l * np.ones(5)).tolist()
            idx[l] += 5
    print(idx)
    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(0, 2.0, (10, 100, 2))
    props = np.array([[[len(v) - 1000]] for v in mnist_idx]) * props / np.sum(props, (1, 2), keepdims=True)
    # idx = 1000*np.ones(10, dtype=np.int64)
    for user in trange(1000):
        for j in range(2):
            l = (user + j) % 10
            num_samples = int(props[l, user // 10, j])
            # print(num_samples)
            if idx[l] + num_samples < len(mnist_idx[l]):
                # X[user] += mnist_data[l][idx[l]:idx[l] + num_samples].tolist()
                # y[user] += (l * np.ones(num_samples)).tolist()
                X_idx[user] += mnist_idx[l][idx[l]:idx[l] + num_samples].tolist()
                y_idx[user] += (l * np.ones(num_samples)).tolist()
                idx[l] += num_samples

    print(idx)
    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    # train_test_idx = dict([(user, {'train': None, 'test': None}) for user in range(1000)])
    # Setup 1000 users
    for i in trange(1000, ncols=120):
        # uname = 'f_{0:05d}'.format(i)
        uname = i
        num_samples = len(X_idx[i])
        train_len = int(0.9 * num_samples)
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x_index': X_idx[i][:train_len]}
        train_data['num_samples'].append(train_len)

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x_index': X_idx[i][train_len:]}
        test_data['num_samples'].append(num_samples - train_len)

        # combined = list(zip(X_idx[i], y_idx[i]))
        # random.shuffle(combined)
        # X_idx[i][:], y_idx[i][:] = zip(*combined)
        # num_samples = len(X_idx[i])
        # train_len = int(0.9 * num_samples)
        # test_len = num_samples - train_len
        # train_test_idx[i]['train'] = X_idx[i][:train_len]
        # train_test_idx[i]['test'] = X_idx[i][train_len:]
        # train_data['users'].append(uname)
        # train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        # train_data['num_samples'].append(train_len)
        # test_data['users'].append(uname)
        # test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        # test_data['num_samples'].append(test_len)
        # 客户端拥有的类
        ya = y_idx[i][:train_len] + y_idx[i][train_len:]
        assert len(set(ya)) <= 2
        # 便利所有的 idnex
        for ji, j in enumerate(train_data['user_data'][i]['x_index'] + test_data['user_data'][i]['x_index']):
            # j 是拥有的数据的 index 对应在全局数据上的标签必须和 ya[ji] 对应
            assert all_train_test_target[j] == ya[ji]
    # print(train_data['num_samples'])
    # print(sum(train_data['num_samples']))
    with open(train_index_record, 'wb') as fp:
        pickle.dump(train_data, fp)
    with open(test_index_record, 'wb') as fp:
        pickle.dump(test_data, fp)

if __name__ == '__main__':
    generate()