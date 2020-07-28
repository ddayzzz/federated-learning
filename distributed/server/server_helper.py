import abc
import io
import pickle
import codecs
import torch
import pandas as pd
from torch.nn import Module


class ServerHelper(abc.ABC):

    def __init__(self, model: Module, params):
        self.model = model
        self.latest_weights = model.state_dict()
        self.min_train_client = params['min_clients']
        self.num_rounds = params['num_rounds']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']
        self.save_every = params['save_every']
        self.eval_every = params['eval_every']
        self.test_every = params['test_every']
        self.num_clients_per_round = params['clients_per_round']
        # 保存相关的参数
        self.online_clients = set()
        self.current_round = -1
        # 记录的数据, 行为 rounds, 列为各项参数,
        # TODO 暂时在 BRATS 2018 上记录
        #
        self.received_train_stats_from_client_this_round = []
        self.received_eval_stats_from_client_this_round = None
        # 用于缓存 metric 的记录的 metric
        self.cached_metris_columns = None

    def dump_weights(self):
        buffer = io.BytesIO()
        torch.save(self.latest_weights, buffer)
        buffer.seek(0)
        x = codecs.encode(pickle.dumps(buffer), "base64").decode()
        buffer.close()
        return x

    def load_weights(self):
        """
        :return:
        """
        o = []
        for weight_str in self.received_train_stats_from_client_this_round:
            weight = pickle.loads(codecs.decode(weight_str['new_weight'].encode(), "base64"))
            weight = torch.load(weight)
            o.append(weight)
        return o

    def compute_train_stats(self):
        # received_train_stats_from_client_this_round 包含了各项相关的信息
        if self.cached_metris_columns is None:
            columns = dict()
            one_item = self.received_train_stats_from_client_this_round[0]
            for valid_item in ['train', 'test', 'val']:
                if valid_item in one_item:
                    for key, value in one_item[valid_item].items():
                        columns.update({valid_item + '_' + key: type(value)})
            # 第一级的参数, 有很多, 包括 weight 等都不要
            columns['round_number'] = str
            columns['client_id'] = str
            self.cached_metris_columns = columns
        columns = list(self.cached_metris_columns.keys())
        df = pd.DataFrame(columns=columns)
        for item in self.received_train_stats_from_client_this_round:
            to_write = dict()
            for k, v in item.items():
                if k in ['train', 'test', 'val']:
                    for key, value in v.items():
                        to_write.update({k + '_' + key: value})
                else:
                    if k in columns:
                        to_write[k] = v
            df = df.append(to_write, ignore_index=True)

        return df

    def log_metrics(self, df):
        """
        基于给定的结果输出
        :param df:
        :return:
        """
        # 输出均值, 最小, 最大, 标准查
        print('{:<25}{:^20}{:^20}{:^20}{:^20}'.format('Metric', 'Min', 'Mean', 'Max', 'Std'))
        for column_name, column_type in self.cached_metris_columns.items():
            if column_type == str:
                continue
            std = df[column_name].std()
            min = df[column_name].min()
            max = df[column_name].max()
            mean = df[column_name].mean()
            if column_type == int:
                print(f'{column_name:<25}{min:>20d}{mean:>203d}{max:>20d}{std:>20d}')
            else:
                print(f'{column_name:<25}{min:>20.3f}{mean:>20.3f}{max:>20.3f}{std:>20.3f}')

    @property
    def num_online_clients(self):
        return len(self.online_clients)

    def add_client(self, client):
        self.online_clients.add(client)

    def remove_client(self, client):
        self.online_clients.remove(client)

    @abc.abstractmethod
    def aggregate(self):
        pass


class FedAvgHelper(ServerHelper):

    def __init__(self, model: Module, params):
        super(FedAvgHelper, self).__init__(model, params)

    def aggregate(self):
        """
        聚合模型的参数
        :param state_dicts:
        :return:
        """
        weights = self.load_weights()
        new_weights = dict()
        num_clients = len(weights)
        weight_keys = weights[0].keys()
        for k in weight_keys:
            new_weight = torch.zeros_like(weights[0][k])
            for client in weights:
                new_weight += client[k]
            new_weight /= num_clients
            new_weights[k] = new_weight
        self.latest_weights = new_weights
        print('aggreaged')
