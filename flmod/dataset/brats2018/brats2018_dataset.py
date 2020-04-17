
import numpy as np
import torch
import torch.utils.data as tdata
import os
import glob
from skimage import transform


class BRATS2018Dataset(tdata.Dataset):

    def __init__(self, training_dir, img_dim, max_size=None):
        """
        定义 BRATS2018 的数据集
        :param training_dir: 处理好的数据目录.
        :param img_dim: 输出的图像的维度
        :param max_size: 最大的大小
        """
        self.training_dir = training_dir
        self.img_dim = img_dim
        input_files = glob.glob(os.sep.join((training_dir, 'flair', '*.npy')), recursive=False)[:max_size]
        annotation_files = [os.sep.join((training_dir, 'ground_truth', os.path.split(x)[-1])) for x in input_files]
        self.inputs = input_files
        self.annotations = annotation_files

    @staticmethod
    def preprocess(img, dim):
        # [H, W]
        img = transform.resize(img, (dim, dim))
        # [1, H, W]
        img = np.expand_dims(img, axis=0)
        return img
    
    @staticmethod
    def resize(img, dim):
        img = transform.resize(img, (dim, dim))
        return img

    def _preprocess(self, img):
        # [H, W]
        img = transform.resize(img, (self.img_dim, self.img_dim))
        # [1, H, W]
        img = np.expand_dims(img, axis=0)
        # float64 -> 32
        img = img.astype(np.float32)
        return img

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        imgi = self._preprocess(np.load(self.inputs[item]))  # [H, W]
        imga = self._preprocess(np.load(self.annotations[item]))
        return imgi, imga
        # return {'image': torch.from_numpy(imgi), 'mask': torch.from_numpy(imga)}
        # worker_info = tdata.get_worker_info()
        # if worker_info is None:
        #     start = self.start
        #     end = self.end
        # else:
        #     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     start = self.start + worker_id * per_worker
        #     end = min(start + per_worker, self.end)
        # # 返回一批数据
        # for data, anna in zip(self.inputs[start:end], self.annotations[start:end]):
        #     yield {
        #         'image': self.preprocess(np.load(data)),
        #         'mask': self.preprocess(np.load(anna))
        #     }


class BRATS2018DatasetForFL(BRATS2018Dataset):

    def __init__(self, training_dir, img_dim, max_size=None):
        super(BRATS2018DatasetForFL, self).__init__(training_dir=training_dir, img_dim=img_dim, max_size=max_size)

    def __getitem__(self, item):
        imgi = self._preprocess(np.load(self.inputs[item]))  # [H, W]
        imga = self._preprocess(np.load(self.annotations[item]))
        return torch.from_numpy(imgi), torch.from_numpy(imga)


if __name__ == '__main__':
    training_dir = os.path.sep.join(('data', 'brats2018', 'training'))
    dataset = BRATS2018Dataset(training_dir=training_dir, img_dim=128)
    print(list(torch.utils.data.DataLoader(dataset, num_workers=2)))