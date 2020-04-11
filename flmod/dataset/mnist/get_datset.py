import os
from torchvision import datasets, transforms


class FlattenInput(object):

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.view(-1)


def get_dataset(flatten_input=False):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    :param dataset:
    :param iid:
    :param num_users:
    :return:
    """
    if flatten_input:
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            FlattenInput()])
    else:
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.dirname(__file__)
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    return train_dataset, test_dataset