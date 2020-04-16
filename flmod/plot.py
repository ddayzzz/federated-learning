import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def get_info(metric_json):
    with open(metric_json) as fp:
        metrics = json.load(fp)
    return metrics

def get_metrics(dataset):
    prefix = f'result/{dataset}'
    dirs = os.listdir(prefix)
    res = dict()
    for info in dirs:

        res[info] = get_info('{}/{}/metrics.json'.format(prefix, info))
    return res

def plot(dataset, *args):
    infos = get_metrics(dataset)
    for arg in args:
        for exp, jf in infos.items():
            num_rounds = int(jf['num_rounds']) + 1
            metric = np.asarray(jf[arg])
            plt.plot(np.arange(num_rounds), metric, label=exp)
            # plt.plot(np.asarray(rounds1[:len(losses1)]), np.asarray(losses1), '--', linewidth=3.0, label='mu=0, E=20',
            #          color="#17becf")
            plt.legend(loc='best')
        plt.show()



if __name__ == '__main__':
    # dataset = 'mnist_user1000_niid_0_keep_10_train_9'
    dataset = 'synthetic_alpha0_beta0_niid'
    plot(dataset, 'graddiff_on_train_data', 'loss_on_eval_data', 'acc_on_eval_data')