import os
import json
import matplotlib.pyplot as plt
import matplotlib
gui_backend = plt.get_backend()
matplotlib.use(gui_backend)
import numpy as np
import re
import pandas as pd
import glob


def get_info(metric_json):
    with open(metric_json) as fp:
        metrics = json.load(fp)
    return metrics


def load_metric(result_prefix, dataset, exp_filter_re):
    prefix = f'{result_prefix}/{dataset}'
    dirs = os.listdir(prefix)
    res = dict()
    for info in dirs:
        if exp_filter_re is not None and re.search(exp_filter_re, info) is None:
            continue
        exp_name = info[info.find('wn'):]
        res[exp_name] = get_info('{}/{}/metrics.json'.format(prefix, info))
    return res


def load_metrics(cfgs: dict) -> dict:
    """
    加载多个 metrics
    :param cfgs:
    :param exp_filter_re:
    :return:
    """
    res = dict()
    for exp_dir in cfgs.keys():
        metric = get_info(os.path.join(exp_dir, 'metrics.json'))
        evals = glob.glob(os.path.join(exp_dir, 'eval*.csv'))
        eval_result = pd.read_csv(evals[0])
        res[exp_dir] = {'metric': metric, 'eval': eval_result}
    return res


def plots_metric(infos, cfgs, title, **kwarg):
    for y_key, y_name in kwarg.items():
        plt.figure()
        for exp, jf in infos.items():
            num_rounds = int(jf['metric']['num_rounds']) + 1
            metric = np.asarray(jf['metric'][y_key])
            plt.plot(np.arange(num_rounds), metric, label=cfgs[exp], linewidth=3.0)
        plt.legend(loc='best', ncol=3)
        leg = plt.gca().get_legend()  # 或leg=ax.get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=12)
        plt.xlabel('通信轮次', fontsize=12)
        plt.ylabel(y_name, fontsize=12)
        plt.title(title)
        plt.show()
        # plt.tight_layout()
        # f.savefig(f'exps/exp_{dataset}_{arg}.png')


def plots_eval_hist(infos, cfgs, title, **kwarg):
    for exp_name, info in infos.items():
        df = info['eval']
        ax = df.test_acc.hist(alpha=0.5, bins=100)
        ax = df.test_acc.plot.density(ax=ax)
    plt.legend(loc='best', ncol=3)
    leg = plt.gca().get_legend()  # 或leg=ax.get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12)
    plt.xlabel('测试集上的准确率', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title)
    # for y_key, y_name in kwarg.items():
    plt.show()


if __name__ == '__main__':
    # dataset = 'mnist_user1000_niid_0_keep_10_train_9'
    # dataset = 'synthetic_alpha0_beta0_iid'
    # dataset = 'brats2018_train_9_test_1'
    # dataset = 'shakespeare_all_data_niid_sf0.2_tf0.8_k0'
    # plot(dataset, 'graddiff_on_train_data', 'loss_on_eval_data', 'acc_on_test_data', exp_filter_re=r'dp0.9')
    # dataset = 'brats2018_train_9_test_1'
    # plot(dataset, 'loss_on_eval_data', 'acc_on_test_data', exp_filter_re=r'mu0\.0_')
    # plot(dataset, 'loss_on_eval_data', 'acc_on_test_data', exp_filter_re=r'dp0\.0')
    # datasets = ['synthetic_alpha0.5_beta0.5_niid', 'synthetic_alpha0_beta0_niid', 'synthetic_alpha1_beta1_niid', 'synthetic_alpha0_beta0_iid']
    # for ds in datasets:
    #     plot(ds, 'graddiff_on_train_data', 'loss_on_eval_data', 'acc_on_test_data')
    result_prefix = '../flearn/result'
    dataset = 'femnist'
    exp_filter_re = None
    cfgs = {
        '../flearn/result/femnist/2020-07-26T15-07-12_fedavg_cnn__wn10_tn173_sd0_lr0.004_ep1_bs5': 'FedAvg',
    }
    infos = load_metrics(cfgs)
    # plots_metric(infos, cfgs, title='femnist', loss_on_eval_data='测试集上的损失函数', acc_on_test_data='测试集上的准确率')
    plots_eval_hist(infos, cfgs, title='femnist', loss_on_eval_data='测试集上的损失函数', acc_on_eval_data='测试集上的准确率')