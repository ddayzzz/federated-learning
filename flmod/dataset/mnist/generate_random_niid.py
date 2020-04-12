import torch
import numpy as np
import pickle
import os
import torchvision
cpath = os.path.dirname(__file__)

NUM_USER = 100
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
MNIST_DOWNLOAD = cpath
IMAGE_DATA = False
np.random.seed(6)


def data_split(data, num_split):
    # delta 是 data 的数据按照 num_split 的等分的数量; r 是否有多余的
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            # 如果是等分的, 即 r == 0, 则会一直使用这句话
            data_lst.append(data[i:i+delta])
            # 分割完成
            i += delta
    return data_lst


def choose_two_digit(split_data_lst):
    """

    :param split_data_lst:
    :return:
    """
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            # 如果这个 digit 的可用的样本 > 0
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, 2, replace=False).tolist()
    except:
        print(available_digit)
    return lst


def main():
    # Get MNIST data, normalize, and divide by level
    print('>>> Get MNIST data.')
    trainset = torchvision.datasets.MNIST(MNIST_DOWNLOAD, download=True, train=True)
    testset = torchvision.datasets.MNIST(MNIST_DOWNLOAD, download=True, train=False)
    # 数据变成 numpy 格式, 且 flatten
    # train_mnist = ImageDataset(trainset.train_data, trainset.train_labels)
    # test_mnist = ImageDataset(testset.test_data, testset.test_labels)
    train_mnist_y = trainset.targets.numpy()
    train_mnist_index = np.arange(len(train_mnist_y))
    test_mnist_y = testset.targets.numpy()
    test_mnist_index = np.arange(len(test_mnist_y))
    # 每个类的数据在一起, 按照 0-9的排列. [(类别0的数量, 784), ..., (类别9的数量, 784)]
    mnist_train_data_index = []

    for number in range(10):
        idx = train_mnist_y == number
        # 存储对应的编号
        mnist_train_data_index.append(train_mnist_index[idx])
    # 最小的数据的大小
    # min_number = min([len(dig) for dig in mnist_train_data_index])
    # for number in range(10):
    #     # 采取的数量
    #     mnist_train_data_index[number] = mnist_train_data_index[number][:min_number-1]

    split_mnist_traindata = []
    for digit in mnist_train_data_index:
        # 一种类型的 digit 进行 20 的等分
        split_mnist_traindata.append(data_split(digit, 20))

    mnist_test_data_index = []
    for number in range(10):
        idx = test_mnist_y == number
        # 存储对应的编号
        mnist_test_data_index.append(test_mnist_index[idx])
    # 按照比例分割, 总共有 len(digit) // 20 = 最小的类别的数据 // 20
    split_mnist_testdata = []
    for digit in mnist_test_data_index:
        split_mnist_testdata.append(data_split(digit, 20))
    # 每个类别数据的占比
    data_distribution = np.array([len(v) for v in mnist_train_data_index])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))
    # 训练数据中, 每个类的数据量
    digit_count = np.array([len(v) for v in split_mnist_traindata])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

    # digit_count = np.array([len(v) for v in split_mnist_testdata])
    # print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    # Assign train samples to each user
    # train_X = [[] for _ in range(NUM_USER)]
    # train_y = [[] for _ in range(NUM_USER)]
    # test_X = [[] for _ in range(NUM_USER)]
    # test_y = [[] for _ in range(NUM_USER)]
    train_X_idx = [list() for _ in range(NUM_USER)]
    test_X_idx = [list() for _ in range(NUM_USER)]

    print(">>> Data is non-i.i.d. distributed")
    print(">>> Data is unbalanced")

    for user in range(NUM_USER):
        # 显示当前用户下剩下的可以分配的数据批次(每次递减两个类)
        print(user, np.array([len(v) for v in split_mnist_traindata]))

        for d in choose_two_digit(split_mnist_traindata):
            # d 是选择的两个类
            # 只存储 index
            # l = len(split_mnist_traindata[d][-1])
            train_X_idx[user] += split_mnist_traindata[d].pop().tolist()
            # train_y_idx[user] += (d * np.ones(l)).tolist()

            # l = len(split_mnist_testdata[d][-1])
            test_X_idx[user] += split_mnist_testdata[d].pop().tolist()
            # test_y_idx[user] += (d * np.ones(l)).tolist()

    # Setup directory for train/test data
    print('>>> Set data path for MNIST.')
    image = 1 if IMAGE_DATA else 0
    train_path = '{}/data/train/all_data_{}_random_niid.pkl'.format(cpath, image)
    test_path = '{}/data/test/all_data_{}_random_niid.pkl'.format(cpath, image)

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 1000 users
    for i in range(NUM_USER):
        uname = i
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x_index': train_X_idx[i]}
        train_data['num_samples'].append(len(train_X_idx[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x_index': test_X_idx[i]}
        test_data['num_samples'].append(len(test_X_idx[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile)

        print('>>> Save data.')


if __name__ == '__main__':
    main()

