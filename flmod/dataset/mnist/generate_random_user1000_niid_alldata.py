# from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import pickle
import os
from torchvision import datasets
DATA = os.path.dirname(__file__)
IMAGE_DATA = 0


# Setup directory for train/test data
train_path = './data/train/user1000_niid_0_keep_10_train_9.pkl'
test_path = './data/test/user1000_niid_0_keep_10_train_9.pkl'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
# mnist = fetch_openml('MNIST original', data_home='./data')
train_dataset = datasets.MNIST(DATA, train=True, download=True, transform=None)
test_dataset = datasets.MNIST(DATA, train=False, download=True, transform=None)
x_data = np.concatenate((train_dataset.data.view(-1, 784).numpy() / 255, test_dataset.data.view(-1, 784).numpy() / 255.))
y_data = np.concatenate((train_dataset.targets.numpy(), test_dataset.targets.numpy()))
mu = np.mean(x_data.astype(np.float32), 0)
sigma = np.std(x_data.astype(np.float32), 0)
train_x = (x_data.astype(np.float32) - mu) / (sigma + 0.001)
mnist_data = []
for i in trange(10):
    idx = y_data == i
    mnist_data.append(x_data[idx])

print([len(v) for v in mnist_data])

###### CREATE USER DATA SPLIT #######
# Assign 10 samples to each user
X = [[] for _ in range(1000)]
y = [[] for _ in range(1000)]
idx = np.zeros(10, dtype=np.int64)
for user in range(1000):
    for j in range(2):
        l = (user + j) % 10
        X[user] += mnist_data[l][idx[l]:idx[l] + 5].tolist()
        y[user] += (l * np.ones(5)).tolist()
        idx[l] += 5
print(idx)

# Assign remaining sample by power law
user = 0
props = np.random.lognormal(0, 2.0, (10, 100, 2))
props = np.array([[[len(v) - 1000]] for v in mnist_data]) * props / np.sum(props, (1, 2), keepdims=True)
# idx = 1000*np.ones(10, dtype=np.int64)
for user in trange(1000):
    for j in range(2):
        l = (user + j) % 10
        num_samples = int(props[l, user // 10, j])
        # print(num_samples)
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l] + num_samples].tolist()
            y[user] += (l * np.ones(num_samples)).tolist()
            idx[l] += num_samples

print(idx)

# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

# Setup 1000 users
for i in trange(1000, ncols=120):
    uname = 'f_{0:05d}'.format(i)

    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.9 * num_samples)
    test_len = num_samples - train_len

    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print(train_data['num_samples'])
print(sum(train_data['num_samples']))

with open(train_path, 'wb') as outfile:
    pickle.dump(train_data, outfile)
with open(test_path, 'wb') as outfile:
    pickle.dump(test_data, outfile)