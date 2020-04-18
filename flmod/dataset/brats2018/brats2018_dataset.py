
import numpy as np
import torch
import torch.utils.data as tdata
import os
import glob
from skimage import transform


ROOT = os.path.dirname(__file__)
ORIGIN_DATA_ROOT = os.sep.join((ROOT, 'data', 'origin'))
FLAIR_ROOT = os.sep.join((ORIGIN_DATA_ROOT, 'flair'))
GROUND_TRUTH_ROOT = os.sep.join((ORIGIN_DATA_ROOT, 'ground_truth'))


class BRATS2018Dataset(tdata.Dataset):

    def __init__(self, data, options):
        """
        BRATS2018
        :param train:
        :param options:
        """
        self.img_dim = options['input_shape'][1]  # [1,128,128]
        ids = data['x_index']
        # 生成现有的文件
        self.images = [os.sep.join((FLAIR_ROOT, x)) + '.npy' for x in ids]
        self.labels = [os.sep.join((GROUND_TRUTH_ROOT, x)) + '.npy' for x in ids]

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
        return len(self.labels)

    def __getitem__(self, item):
        imgi = self._preprocess(np.load(self.images[item]))  # [H, W]
        imga = self._preprocess(np.load(self.labels[item]))
        return imgi, imga


class InstitutionWiseBRATS2018Dataset(tdata.Dataset):

    def __init__(self, training_dir, img_dim, config_json, max_size=None):
        self.training_dir = training_dir
        self.img_dim = img_dim
        # input_files = glob.glob(os.sep.join((training_dir, 'flair', '*.npy')), recursive=False)[:max_size]
        # annotation_files = [os.sep.join((training_dir, 'ground_truth', os.path.split(x)[-1])) for x in input_files]
        # 添加这些数据所属的机构号, e.g.

        # 查找对应的名称
        import json
        with open(config_json) as f:
            cfg = json.load(f)
        institutions_its_sample_index = dict()
        images = []
        ann = []
        start_index = 0
        for institution, patient_ids in cfg.items():
            # 读取对应的 img, ann
            one_inst_ids = []
            for pid in patient_ids:
                input_files = glob.glob(os.sep.join((training_dir, 'flair', f'{pid}*.npy')), recursive=False)[
                              :max_size]
                annotation_files = [os.sep.join((training_dir, 'ground_truth', os.path.split(x)[-1])) for x in
                                    input_files]
                one_inst_ids.extend(range(start_index, start_index + len(input_files)))
                start_index = start_index + len(input_files)
                # 添加到总体的样本上去
                images.extend(input_files)
                ann.extend(annotation_files)
            institutions_its_sample_index[institution] = set(one_inst_ids)
        self.inputs = images
        self.annotations = ann
        self.institutions = cfg.keys()
        self.institutions_its_sample_index = institutions_its_sample_index

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
        return imgi, imga  # image, mask

if __name__ == '__main__':
    training_dir = os.path.sep.join(('data', 'brats2018', 'training'))
    dataset = BRATS2018Dataset(training_dir=training_dir, img_dim=128)
    print(list(torch.utils.data.DataLoader(dataset, num_workers=2)))