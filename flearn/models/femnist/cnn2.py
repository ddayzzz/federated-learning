import tensorflow as tf
import tqdm
import numpy as np
from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size, process_grad
from flearn.models.base_model import BaseModel


class Model(BaseModel):

    def __init__(self, num_classes, image_size, options, optimizer, seed=1):
        # params
        self.num_classes = num_classes
        self.image_size = image_size
        super(Model, self).__init__(optimizer=optimizer, seed=seed, options=options)

    # 一下为 MAML 的格式
    def create_conv_variables(self, kernel_size, in_dim, out_dim, conv_name, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d):
        """
        创建卷积层的变量
        :param kernel_size:
        :param in_dim:
        :param out_dim:
        :param conv_name:
        :param kernel_initializer:
        :return:
        """
        w = tf.get_variable(conv_name + '_w', [kernel_size, kernel_size, in_dim, out_dim], initializer=kernel_initializer())
        b = tf.get_variable(conv_name + '_b', initializer=tf.zeros([out_dim]))
        return (w, b)

    def create_fc_variables(self, in_dim, out_dim, fc_name,
                            weight_initializer=tf.contrib.layers.xavier_initializer):
        w = tf.get_variable(fc_name + '_w', [in_dim, out_dim], initializer=weight_initializer())
        b = tf.get_variable(fc_name + '_b', initializer=tf.zeros([out_dim]))
        return (w, b)

    def create_params(self):
        """
        创建网路的参数. 网络的参数保存在
        :param input_channel:
        :param kernel_size:
        :return: 参数 dict: Dict[name] -> variable
        """
        weights = {}
        with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
            (weights['conv1w'], weights['conv1b']) = self.create_conv_variables(5, 1, 32, 'conv1')
            (weights['conv2w'], weights['conv2b']) = self.create_conv_variables(5, 32, 64, 'conv2')
            (weights['fc1w'], weights['fc1b']) = self.create_fc_variables(7 * 7 * 64, 2048, 'fc1')
            (weights['fc2w'], weights['fc2b']) = self.create_fc_variables(2048, self.num_classes, 'fc2')
        return weights

    def conv_block(self, x, weight, bias, scope):
        """
        build a block with conv2d->pooling. 暂时删除 batch_norm 的设置
        :param x: 输入的张量
        :param weight: conv2d 的 weight
        :param bias: conv2d 的 bias
        :param scope:
        :return:
        """
        # conv
        x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME', name=scope + '_conv2d') + bias
        x = tf.nn.relu(x, name=scope + '_relu')
        # pooling
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name=scope + '_pool')
        return x

    def fc_block(self, x, weight, bias, name, flatten=False, act=tf.nn.relu):
        if flatten:
            x = tf.reshape(x, [-1, np.prod([int(dim) for dim in x.get_shape()[1:]])], name=name + '_flatten')
        x = tf.add(tf.matmul(x, weight), bias, name=name + '_out')
        if act is not None:
            x = act(x, name=name + '_act')
        return x

    def forward(self, x, weights):
        hidden1 = self.conv_block(x, weights['conv1w'], weights['conv1b'], 'conv1')
        hidden2 = self.conv_block(hidden1, weights['conv2w'], weights['conv2b'], 'conv2')
        # TODO 灰度图像, dim1 = dim2 = 1, 可使用  reduce_mean 去掉
        output = self.fc_block(hidden2, weights['fc1w'], weights['fc1b'], name='fc1', flatten=True)
        output = self.fc_block(output, weights['fc2w'], weights['fc2b'], name='fc2', act=None, flatten=False)
        return output

    def create_model(self):
        """
        创建基本你的模型
        :param optimizer:
        :return:
        """
        support_features = tf.placeholder(tf.float32, shape=[None, self.image_size * self.image_size], name='support_features')
        query_features = tf.placeholder(tf.float32, shape=[None, self.image_size * self.image_size], name='query_features')
        # 转换为张量
        support_input_layer = tf.reshape(support_features, [-1, self.image_size, self.image_size, 1], name='support_features_reshaped')
        query_input_layer = tf.reshape(query_features, [-1, self.image_size, self.image_size, 1], name='query_features_reshaped')
        support_labels = tf.placeholder(tf.int64, shape=[None], name='support_labels')
        query_labels = tf.placeholder(tf.int64, shape=[None], name='query_labels')
        support_labels_onehot = tf.one_hot(support_labels, depth=self.num_classes, name='support_labels_onehot')
        query_labels_onehot = tf.one_hot(query_labels, depth=self.num_classes, name='query_labels_onehot')
        # 基于 support, 计算一次参数
        self.weights = self.create_params()
        support_pred_logitis = self.forward(support_input_layer, self.weights)
        support_correct_count = tf.count_nonzero(
            tf.equal(tf.argmax(support_labels_onehot, axis=1), tf.argmax(tf.nn.softmax(support_pred_logitis, dim=1), axis=1)))
        #
        grads = tf.gradients(support_pred_logitis, list(self.weights.values()))
        # 计算后的梯度保存为 key->variable 字典
        # gvs = dict(zip(self.weights.keys(), grads))

        # theta' = theta - alpha * grads, 这里能否使用 adam?
        # fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.options['lr'] * gvs[key] for key in self.weights.keys()]))
        to_minimize = self.optimizer.apply_gradients(grads)
        with tf.control_dependencies([to_minimize]):
            # 接着是基于 query
            query_pred = self.forward(query_input_layer, self.weights)
            # 计算损失函数 L(f_theta'(D'))
            query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_labels_onehot)
            # 基于这个 query 定义优化器
            gvs = self.optimizer.compute_gradients(query_loss)
            train_op = self.optimizer.apply_gradients(gvs)
            grads, _ = zip(*gvs)
            # eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
            # return features, labels, train_op, grads, eval_metric_ops, loss
            query_correct_count = tf.count_nonzero(
                tf.equal(tf.argmax(query_labels_onehot, axis=1), tf.argmax(tf.nn.softmax(query_pred, dim=1), axis=1)))
        return (support_features, query_features), (support_labels, query_labels), train_op, grads, query_correct_count, tf.reduce_mean(query_loss)

    def solve_sgd_meta(self, sp, qr):
        """
        运行一次 SGD
        :param mini_batch_data:
        :return:
        """
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                           feed_dict={self.features[0]: sp[0],
                                                      self.features[1]: qr[0],
                                                      self.labels[0]: sp[1],
                                                      self.labels[1]: qr[1]})
        sz = len(sp[1]) + len(qr[1])
        comp = sz * self.flops
        weights = self.get_params()
        return grads, loss, weights, comp

    def test_meta(self, sp, qr):
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features[0]: sp[0],
                                                      self.features[1]: qr[0],
                                                      self.labels[0]: sp[1],
                                                      self.labels[1]: qr[1]})
        return tot_correct, loss
