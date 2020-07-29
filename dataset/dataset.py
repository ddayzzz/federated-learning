import abc


class BaseDataset(abc.ABC):

    def __init__(self, dataset_name, sub_name=None):
        self.dataset_name = dataset_name
        self.sub_name = sub_name


class KeyValuePairDataset(BaseDataset):

    def __init__(self, dataset_name, sub_name=None):
        super(KeyValuePairDataset, self).__init__(dataset_name=dataset_name, sub_name=sub_name)

