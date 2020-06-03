import torchvision
import os
import pickle
import numpy as np

MNIST_DOWNLOAD = os.path.dirname(__file__)


def test_equal_noniid():
    trainset = torchvision.datasets.MNIST(MNIST_DOWNLOAD + '/data', download=True, train=True).targets.numpy()
    testset = torchvision.datasets.MNIST(MNIST_DOWNLOAD + '/data', download=True, train=False).targets.numpy()
    train_path = '{}/data/train/all_data_{}_equal_niid.pkl'.format(MNIST_DOWNLOAD, 0)
    test_path = '{}/data/test/all_data_{}_equal_niid.pkl'.format(MNIST_DOWNLOAD, 0)
    with open(train_path, 'rb') as fp:
        train_info = pickle.load(fp)
    with open(test_path, 'rb') as fp:
        test_info = pickle.load(fp)
    train_sizes, test_sizes = [], []
    for user in train_info['users']:
        train_idx = train_info['user_data'][user]['x_index']
        test_idx = test_info['user_data'][user]['x_index']
        train_sizes.append(train_info['num_samples'][user])
        test_sizes.append(test_info['num_samples'][user])
        assert len(set(trainset[train_idx])) == 2
        assert len(set(testset[test_idx])) == 2
    for sizes in [train_sizes]:
        lsz = sizes[0]
        for sz in sizes[1:]:
            assert lsz == sz


def test_original_paper_ver_and_index_ver():
    trainset = torchvision.datasets.MNIST(MNIST_DOWNLOAD + '/data', download=True, train=True)
    # testset = torchvision.datasets.MNIST(MNIST_DOWNLOAD + '/data', download=True, train=False)
    train_x = trainset.data.numpy()
    train_y = trainset.targets.numpy()
    org = '{}/data/train/random_niid_org.pkl'.format(MNIST_DOWNLOAD, 1)
    index = '{}/data/train/all_data_{}_random_niid.pkl'.format(MNIST_DOWNLOAD, 0)
    with open(org, 'rb') as fp:
        org_ver = pickle.load(fp)
    with open(index, 'rb') as fp:
        index_ver = pickle.load(fp)
    for user in index_ver['users']:
        assert org_ver['num_samples'][user] == index_ver['num_samples'][user]
        for i in range(len(org_ver['user_data'][user]['x'])):
            assert np.sum(org_ver['user_data'][user]['x'][i] - train_x[index_ver['user_data'][user]['x_index'][i]])  == 0
            assert org_ver['user_data'][user]['y'][i] == train_y[index_ver['user_data'][user]['x_index'][i]]

if __name__ == '__main__':
    # test_equal_noniid()
    test_original_paper_ver_and_index_ver()