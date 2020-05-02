import numpy as np
import argparse
import importlib
import torch
import os
import random
from flmod.utils.worker_utils import read_data_pkl
from flmod.config import DATASETS, TRAINERS
from flmod.config import model_settings
from flmod.config import base_options

def read_options():
    parser = base_options()
    parsed = parser.parse_args()
    options = parsed.__dict__
    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(options['seed'])
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    random.seed(1234 + options['seed'])
    if options['device'].startswith('cuda'):
        torch.cuda.manual_seed_all(123 + options['seed'])
        torch.backends.cudnn.deterministic = True  # cudnn


    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # Add model arguments
    options.update(model_settings.config(dataset_name, options['model']))

    # Load selected trainer
    trainer_path = 'flmod.solvers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, trainer_class, dataset_name, sub_data


def main():
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options()

    train_path = os.path.join('./dataset', dataset_name, 'data', 'train')
    test_path = os.path.join('./dataset', dataset_name, 'data', 'test')

    all_data_info, (train_dataset, test_datatset) = read_data_pkl(train_path, test_path, sub_data=sub_data), \
                                                    model_settings.get_entire_dataset(dataset_name, options=options)
    all_data_info = list(all_data_info)
    all_data_info.extend([train_dataset, test_datatset, model_settings.dataset_config(dataset_name, options=options)])
    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info)
    trainer.train()


if __name__ == '__main__':
    main()
