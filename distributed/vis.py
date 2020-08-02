import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_dfs(result_dir, max_round):
    # 按照回应的名称加载
    res = dict()
    for i in range(max_round):
        res[i] = pd.read_csv(os.path.join(result_dir, 'round_at_round' + str(i) + '.csv'))
    return res

def plot_one_item(infos, item, reduce='mean'):
    df = pd.DataFrame(columns=[item])
    num_rounds = len(infos)
    for i in range(num_rounds):
        # 每一个round 应该 reduce
        metric_per_client = infos[i][item]
        if reduce == 'mean':
            reduced = metric_per_client.mean()
        df = df.append({item: reduced}, ignore_index=True)
    # 可视化变化曲线
    df.plot()
    plt.show()

def plot_one_item_channel_wise(infos, item_prefix, reduce='mean'):
    df = pd.DataFrame(columns=[item_prefix + '_channel{}'.format(i) for i in range(3)])
    num_rounds = len(infos)
    for i in range(num_rounds):
        # 每一个round 应该 reduce
        reduced_dict = dict()
        for c in range(3):
            metric_per_client = infos[i][item_prefix + '_channel' + str(c)]
            if reduce == 'mean':
                reduced = metric_per_client.mean()
            reduced_dict[item_prefix + '_channel' + str(c)] = reduced
        df = df.append(reduced_dict, ignore_index=True)
    # 可视化变化曲线
    df.plot()
    plt.show()


def plot_loss_wise_one_client(infos, client_id_index):
    df = pd.DataFrame(columns=['训练集的损失', '测试集的损失'])
    num_rounds = len(infos)
    for i in range(num_rounds):
        # 每一个round 应该 reduce
        reduced_dict = dict()
        reduced_dict['训练集的损失'] = infos[i].loc[client_id_index, 'train_bce_dice_loss']
        reduced_dict['测试集的损失'] = infos[i].loc[client_id_index, 'test_bce_dice_loss']
        df = df.append(reduced_dict, ignore_index=True)
    # 可视化变化曲线
    df.plot()
    plt.title(infos[0].loc[client_id_index, 'client_id'])
    plt.show()


def plot_item_one_client(infos, client_id, item_prefix):
    df = pd.DataFrame(columns=[item_prefix + '_channel{}'.format(i) for i in range(3)])
    num_rounds = len(infos)
    for i in range(num_rounds):
        # 每一个round 应该 reduce
        reduced_dict = dict()
        for c in range(3):
            d = infos[i].loc[client_id, item_prefix + '_channel' + str(c)]
            reduced_dict[item_prefix + '_channel' + str(c)] = d
        df = df.append(reduced_dict, ignore_index=True)
    # 可视化变化曲线
    df.plot()
    plt.show()

if __name__ == '__main__':
    # res_dir = 'result/2020-07-29T17-03-24_fedavg_sd0_rounds200_ep10_bs32_trainclient3_'
    res_dir = 'result/2020-07-29T14-18-08_fedavg_sd0_rounds200_ep5_bs32_trainclient13_'
    infos = read_dfs(res_dir, 21)
    # plot_one_item(infos, 'test_bce_dice_loss')
    # plot_one_item_channel_wise(infos, 'test_dice_coeff')
    # plot_one_item_channel_wise(infos, 'test_ppv')
    # plot_one_item_channel_wise(infos, 'test_hd95')
    # plot_one_item_channel_wise(infos, 'test_sensitivity')
    # plot_one_item_channel_wise(infos, 'test_iou_score')
    # plot_loss_wise_one_client(infos, 0)
    # plot_loss_wise_one_client(infos, 8)
    # plot_loss_wise_one_client(infos, 12)
    plot_item_one_client(infos, 0, 'test_dice_coeff')
    # plot_item_one_client(infos, 0, 'test_dice_coeff')
    # plot_item_one_client(infos, 0, 'test_ppv')
    # plot_item_one_client(infos, 0, 'test_hd95')
    # plot_item_one_client(infos, 0, 'test_sensitivity')
    # plot_item_one_client(infos, 0, 'test_iou_score')