


def get_model(dataset_name, net):
    if net == 'unet':
        options = {'brats2018': {'n_classes': 3, 'n_channels': 4, 'bilinear': True, 'input_shape': [4, 160, 160]}}
        from distributed.models.unet.unet_model import UNet
        return UNet(**options[dataset_name]), options[dataset_name]
    else:
        raise NotImplementedError