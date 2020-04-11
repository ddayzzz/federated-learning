import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        """
        给定原始的数据集和对应的 index, 产生在 index 中存在的子数据集
        :param dataset:
        :param idxs:
        """
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label