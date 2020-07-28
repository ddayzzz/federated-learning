# 使用全部四种模态以及 HGG 和 LGG 的数据集
import json
import glob
import os
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class _BaseBRATS2018AllModDataset(torch.utils.data.Dataset):

    def __init__(self, images, masks):
        self.images = images
        self.labels = masks
        # for i, l in zip(self.images, self.labels):
        #     assert i[i.rfind(os.sep):] == l[l.rfind(os.sep):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.labels[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        # [H, W, C]
        npimage = npimage.transpose((2, 0, 1))
        # 针对输入数据的不同的模态设计不同的 ground truth
        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.
        WT_Label[npmask == 2] = 1.
        WT_Label[npmask == 4] = 1.
        TC_Label = npmask.copy()
        TC_Label[npmask == 1] = 1.
        TC_Label[npmask == 2] = 0.
        TC_Label[npmask == 4] = 1.
        ET_Label = npmask.copy()
        ET_Label[npmask == 1] = 0.
        ET_Label[npmask == 2] = 0.
        ET_Label[npmask == 4] = 1.
        nplabel = np.empty((160, 160, 3))
        # 输入数据四个模态: flair,t1,t1ce,t2
        nplabel[:, :, 0] = WT_Label
        nplabel[:, :, 1] = TC_Label
        nplabel[:, :, 2] = ET_Label
        nplabel = nplabel.transpose((2, 0, 1))

        nplabel = nplabel.astype(np.float32)
        npimage = npimage.astype(np.float32)

        return npimage, nplabel


class BRATS2018AllModDataset(_BaseBRATS2018AllModDataset):

    def __init__(self, prefix):
        self.prefix = prefix
        #
        images = glob.glob(os.path.join(prefix, 'image') + os.sep + '*.npy')
        labels = glob.glob(os.path.join(prefix, 'mask') + os.sep + '*.npy')
        for i, l in zip(self.images, self.labels):
            assert i[i.rfind(os.sep):] == l[l.rfind(os.sep):]
        super(BRATS2018AllModDataset, self).__init__(images, labels)


class BRATS2018AllModDatasetInsWise(_BaseBRATS2018AllModDataset):

    def __init__(self, cfg, data_key):
        assert data_key in ['train', 'val', 'test']
        with open(cfg, 'r') as fp:
            cfg = json.load(fp)
        assert data_key in cfg.keys()
        super(BRATS2018AllModDatasetInsWise, self).__init__(images=cfg[data_key]['image'], masks=cfg[data_key]['mask'])


if __name__ == '__main__':
    prefix = '/home/liuyuan/shu_codes/brats_seg/data/brats2018_train'
    ds = BRATS2018AllModDataset(prefix)
    for i, m in ds:
        h = 5