import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re


def get_info(metric_json):
    with open(metric_json) as fp:
        metrics = json.load(fp)
    return metrics

def get_metrics(dataset, exp_filter_re):
    prefix = f'result/{dataset}'
    dirs = os.listdir(prefix)
    res = dict()
    for info in dirs:
        if exp_filter_re is not None and re.search(exp_filter_re, info) is None:
            continue
        exp_name = info[info.find('wn'):]
        res[exp_name] = get_info('{}/{}/metrics.json'.format(prefix, info))
    return res

def plot(dataset, *args, exp_filter_re=None):
    infos = get_metrics(dataset, exp_filter_re)
    for arg in args:
        for exp, jf in infos.items():
            num_rounds = int(jf['num_rounds']) + 1
            metric = np.asarray(jf[arg])
            plt.plot(np.arange(num_rounds), metric, label=exp)
            # plt.plot(np.asarray(rounds1[:len(losses1)]), np.asarray(losses1), '--', linewidth=3.0, label='mu=0, E=20',
            #          color="#17becf")
            plt.legend(loc='best')
            plt.xlabel('Rounds')
            plt.ylabel(arg)
        plt.title(dataset)
        plt.show()



if __name__ == '__main__':
    # dataset = 'mnist_user1000_niid_0_keep_10_train_9'
    # dataset = 'synthetic_alpha0_beta0_iid'
    # dataset = 'brats2018_train_9_test_1'
    dataset = 'shakespeare_all_data_niid_sf0.2_tf0.8_k0'
    plot(dataset, 'graddiff_on_train_data', 'loss_on_eval_data', 'acc_on_eval_data', exp_filter_re=r'dp0.9')
    dataset = 'brats2018_train_9_test_1'
    # plot(dataset, 'loss_on_eval_data', 'acc_on_eval_data', exp_filter_re=r'mu0\.0_')
    # plot(dataset, 'loss_on_eval_data', 'acc_on_eval_data', exp_filter_re=r'dp0\.0')
    # datasets = ['synthetic_alpha0.5_beta0.5_niid', 'synthetic_alpha0_beta0_niid', 'synthetic_alpha1_beta1_niid', 'synthetic_alpha0_beta0_iid']
    # for ds in datasets:
    #     plot(ds, 'graddiff_on_train_data', 'loss_on_eval_data', 'acc_on_eval_data')