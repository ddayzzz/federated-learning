import torch.utils.data as data
import os
import os.path
import errno
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms

np.random.seed(6)


class Omniglot(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')
        # 图像的信息
        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        # Dict: 图像的ID(由字符类别+对应其中的编号组成)->对应index
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):

        filename = self.all_items[index][0]
        # 完整的文件路径
        img = str.join(os.sep, [self.all_items[index][2], filename])
        # 类型名称: 字符集名称/字符的类型名
        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        """
        下载文件
        :return:
        """
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            # 遍历的文件名
            if f.endswith("png"):
                r = root.split(os.sep)
                lr = len(r)
                # 保存: (文件名, 字符集名称/字符的类型名(这里用character[xx]来表示), 文件存在的路径前缀)
                retour.append((f, r[lr - 2] + os.sep + r[lr - 1], root))
    print(">>> Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print(">>> Found %d classes" % len(idx))
    return idx


class OmniglotNShot:

    def __init__(self, root, imgsz):
        """
        设置 Omniglot 的数据集的格式
        :param root: 保存有 omniglot 数据集的路径 (e.g. dataset/omniglot)
        :param imgsz: 图像大小
        """
        self.imgsz = imgsz
        self.root = root
        # 加载数据
        self.x = self.load_data()

    def load_data(self):
        # if root/data.npy does not exist, just download it
        x = Omniglot(os.path.sep.join((self.root, 'data')), download=True,
                     transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                   lambda x: x.resize((self.imgsz, self.imgsz)),
                                                   lambda x: np.reshape(x, (self.imgsz, self.imgsz, 1)),
                                                   lambda x: np.transpose(x, [2, 0, 1]),
                                                   lambda x: x / 255.])
                     )

        temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
        # 这里按照类排序
        for (img, label) in x:
            if label in temp.keys():
                temp[label].append(img)
            else:
                temp[label] = [img]

        x = []
        for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
            x.append(np.array(imgs))

        # as different class may have different number of imgs
        # 每个类别只有20张图像
        x = np.array(x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
        # each character contains 20 imgs
        print('>>> Data shape:', x.shape)  # [1623, 20, 84, 84, 1]

        return x

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    def generate_dataset_for_fair(self):
        """
        生成 ICLR 2020 fair learning 格式的数据集.
        :return:
        """
        # 对应的 task 拥有的数据的类的索引号: 前300个task数据范围为 [0, 1200); 后100个为 [1200, 1623), 注意这个是类
        # 后100个任务只有后 423 类型的数据, 这样造成了 task 的不同, 目前不清楚为什么
        assert self.x.shape[0] == 1623
        task_to_class = {}
        for i in range(400):  # 400 tasks
            if i < 300:  # first 300 meta-training tasks
                # TODO 看样子是 5 ways
                class_ids = np.random.choice(1200, 5)
                task_to_class[i] = class_ids
            else:
                # 后面的 426
                class_ids = np.random.choice(range(1200, 1623), 5)  # 这样是没有重复的
                task_to_class[i] = class_ids
        # 拥有对应类的 task 的编号
        class_to_task = {}
        for i in range(1643):
            class_to_task[i] = []
        for i in range(400):
            for j in task_to_class[i]:
                class_to_task[j].append(i)

        X_test = {}  # testing test of all tasks (300 meta-train + 100 meta-test)
        y_test = {}
        X_train = {}  # training set of all tasks (300 meta-train + 100 meta-test)
        y_train = {}

        for i in range(400):
            X_test[i] = []
            y_test[i] = []
            X_train[i] = []
            y_train[i] = []

        all_data = []
        for idx in range(self.x.shape[0]):
            # idx 代表的 class id
            # 指定对应的 shot
            for i in range(self.x.shape[1]):
                content = self.x[idx, i, :, :, :]
                content = content.flatten()
                all_data.append(content)
                if i < 10:
                    for device_id in class_to_task[idx]:
                        X_train[device_id].append(content)
                        # np.where 找到对应的位置
                        y_train[device_id].append(int(np.where(task_to_class[device_id] == idx)[0][0].astype(np.int32)))

                else:
                    for device_id in class_to_task[idx]:
                        X_test[device_id].append(content)
                        y_test[device_id].append(int(np.where(task_to_class[device_id] == idx)[0][0].astype(np.int32)))

        all_data = np.asarray(all_data)
        print(">>> original data:", all_data[0])
        print('>>> Y[100]:', y_train[100])
        print('>>> Y[399]:', y_train[399])
        # some simple normalization
        mu = np.mean(all_data.astype(np.float32), 0)
        print(">>> mu:", mu)
        sigma = np.std(all_data.astype(np.float32), 0)

        for device_id in range(400):
            X_train[device_id] = np.array(X_train[device_id])
            X_test[device_id] = np.array(X_test[device_id])

        for device_id in range(400):
            X_train[device_id] = (X_train[device_id].astype(np.float32) - mu) / (sigma + 0.001)
            X_test[device_id] = (X_test[device_id].astype(np.float32) - mu) / (sigma + 0.001)
            X_train[device_id] = X_train[device_id].tolist()
            X_test[device_id] = X_test[device_id].tolist()
        # 生成数据
        train_file = os.sep.join((self.root, 'data', 'train', 'fair_task[400].pkl'))
        test_file = os.sep.join((self.root, 'data', 'test', 'fair_task[400].pkl'))
        num_device = 400  # device is task
        train_data = {'users': [], 'user_data': {}, 'num_samples': []}
        test_data = {'users': [], 'user_data': {}, 'num_samples': []}
        for i in range(num_device):
            uname = "class_" + str(i)
            train_data['users'].append(uname)
            train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
            test_data['users'].append(uname)
            test_data['user_data'][uname] = {'x': X_test[i], 'y': y_test[i]}
        with open(train_file, 'wb') as outfile:
            pickle.dump(train_data, outfile, pickle.HIGHEST_PROTOCOL)
        with open(test_file, 'wb') as outfile:
            pickle.dump(test_data, outfile, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    prefix = os.path.dirname(__file__)
    if len(prefix) <= 0:
        prefix = '.'
    paths = ['{prefix}{sep}data', '{prefix}{sep}data{sep}train', '{prefix}{sep}data{sep}test']
    for path in paths:
        p = path.format(sep=os.sep, prefix=prefix)
        if not os.path.exists(p):
            os.mkdir(p)
    omniglot_gen = OmniglotNShot(root=prefix, imgsz=28)
    omniglot_gen.generate_dataset_for_fair()
