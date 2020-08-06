import tensorflow as tf
import tqdm
import numpy as np
from flearn.models.base_model import BaseModel


class Model(BaseModel):

    def __init__(self, num_classes, image_size, options, optimizer, seed=1):
        # params
        self.num_classes = num_classes
        self.image_size = image_size
        # 使用 mini-batch 的数量
        self.num_inner_steps = options['num_inner_steps']
        self.batch_size = options['batch_size']
        self.inner_lr = options['lr']
        super(Model, self).__init__(optimizer=optimizer, seed=seed, options=options)


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
        """
        创建 dense 层的相关变量
        :param in_dim:
        :param out_dim:
        :param fc_name:
        :param weight_initializer:
        :return:
        """
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
        """
        前向操作
        :param x:
        :param weight:
        :param bias:
        :param name:
        :param flatten: 是否扁平化输入
        :param act: 输出之前的激活函数
        :return:
        """
        if flatten:
            x = tf.reshape(x, [-1, np.prod([int(dim) for dim in x.get_shape()[1:]])], name=name + '_flatten')
        x = tf.add(tf.matmul(x, weight), bias, name=name + '_out')
        if act is not None:
            x = act(x, name=name + '_act')
        return x

    def forward(self, x, weights):
        """
        输入到输出的定义
        :param x:
        :param weights:
        :return:
        """
        hidden1 = self.conv_block(x, weights['conv1w'], weights['conv1b'], 'conv1')
        hidden2 = self.conv_block(hidden1, weights['conv2w'], weights['conv2b'], 'conv2')
        output = self.fc_block(hidden2, weights['fc1w'], weights['fc1b'], name='fc1', flatten=True)
        output = self.fc_block(output, weights['fc2w'], weights['fc2b'], name='fc2', act=None, flatten=False)
        return output

    def create_model(self):
        """
        创建基本你的模型
        :param optimizer:
        :return:
        """
        support_features = tf.placeholder(tf.float32, shape=[self.num_inner_steps, self.batch_size, self.image_size * self.image_size], name='support_features')
        query_features = tf.placeholder(tf.float32, shape=[self.num_inner_steps, self.batch_size, self.image_size * self.image_size], name='query_features')
        # 转换为张量
        support_labels = tf.placeholder(tf.int64, shape=[self.num_inner_steps, self.batch_size], name='support_labels')
        query_labels = tf.placeholder(tf.int64, shape=[self.num_inner_steps, self.batch_size], name='query_labels')
        # 基于 support, 计算一次参数
        self.weights = self.create_params()

        def support_update(inputx):
            # inputx: 第一个维度为 batch_size
            one_support_features_batch, one_support_label_batch = inputx
            one_support_features_batch_reshaped = tf.reshape(one_support_features_batch, [-1, self.image_size, self.image_size, 1])
            one_support_label_batch_onehot = tf.one_hot(one_support_label_batch, depth=self.num_classes)
            # 利用网络进行前向
            support_pred_logitis = self.forward(one_support_features_batch_reshaped, self.weights)
            support_correct_count = tf.count_nonzero(
                tf.equal(tf.argmax(one_support_label_batch_onehot, axis=1),
                         tf.argmax(tf.nn.softmax(support_pred_logitis, dim=1), axis=1)))

            support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred_logitis,
                                                                   labels=one_support_label_batch_onehot)
            support_loss_mean = tf.reduce_mean(support_loss)
            # 这里计算一次梯度
            grads = tf.gradients(support_loss_mean, list(self.weights.values()))
            # 更新当前的网络参数
            gradients = dict(zip(self.weights.keys(), grads))
            fast_weights = dict(
                zip(self.weights.keys(), [self.weights[key] - self.inner_lr * gradients[key] for key in self.weights.keys()]))
            # 将 fast weight 更新到 weights
            self.weights = fast_weights

            return (support_loss_mean, support_correct_count)

        def query_calc_loss(inputx):
            # inputx: 第一个维度为 batch_size
            one_support_features_batch, one_support_label_batch = inputx
            one_support_features_batch_reshaped = tf.reshape(one_support_features_batch, [-1, self.image_size, self.image_size, 1])
            one_support_label_batch_onehot = tf.one_hot(one_support_label_batch, depth=self.num_classes)
            # 利用网络进行前向
            support_pred_logitis = self.forward(one_support_features_batch_reshaped, self.weights)
            support_correct_count = tf.count_nonzero(
                tf.equal(tf.argmax(one_support_label_batch_onehot, axis=1),
                         tf.argmax(tf.nn.softmax(support_pred_logitis, dim=1), axis=1)), dtype=tf.int64)

            support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred_logitis,
                                                                   labels=one_support_label_batch_onehot)
            support_loss_mean = tf.reduce_mean(support_loss)
            # 这里计算一次梯度
            grads = tf.gradients(support_loss_mean, list(self.weights.values()))
            # # 更新当前的网络参数
            # gradients = dict(zip(self.weights.keys(), grads))

            return (support_loss_mean, support_correct_count, grads)
        # TODO 无法在另外一个 loop 中应用先前的变量: https://www.shuzhiduo.com/A/q4zVZejWzK/
        num_weights = len(self.weights)
        output_shape = (tf.float32, tf.int64)
        # 这两个均为向量, 长度为循环的次数
        sprt_losses, sprt_corrects = tf.map_fn(support_update, dtype=output_shape, elems=(support_features, support_labels), parallel_iterations=self.num_inner_steps)
        output_shape = (tf.float32, tf.int64, [tf.float32] * num_weights)
        qry_losses, qry_corrects, grads = tf.map_fn(query_calc_loss, dtype=output_shape,
                                               elems=(query_features, query_labels),
                                               parallel_iterations=self.num_inner_steps)
        # 这里的 loss 需要平均一下, 按照

        return (support_features, query_features), (support_labels, query_labels), None, grads, qry_corrects, qry_losses

    def create_model_bak(self):
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
        # self.adam_optimizer.create_momtems(self.weights)
        ###### 直接定义参数
        ######
        support_pred_logitis = self.forward(support_input_layer, self.weights)
        support_correct_count = tf.count_nonzero(
            tf.equal(tf.argmax(support_labels_onehot, axis=1), tf.argmax(tf.nn.softmax(support_pred_logitis, dim=1), axis=1)))

        support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred_logitis, labels=support_labels_onehot)
        # 这个用来验证是否求了在query阶段求了二阶导数, sparse 没有二阶导数的实现. 如果没有报错误, 说明没有求得二阶导数
        # support_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=support_pred_logitis, labels=support_labels)

        # theta' = theta - alpha * grads, 这里能否使用 adam?
        # fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.options['lr'] * gvs[key] for key in self.weights.keys()]))
        ####
        # 这里的 loss 是向量. 现在就是希望能够模拟一个 Adam 的过程
        support_loss_mean = tf.reduce_mean(support_loss)
        grads = tf.gradients(support_loss_mean, list(self.weights.values()))
        gvs = dict(zip(self.weights.keys(), grads))
        fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.options['lr'] * gvs[key] for key in self.weights.keys()]))
        # train_op = self.optimizer.apply_gradients(adam_gvs)
        ####
        # TODO 这种方式行不通!! 根本没有计算二阶导数
        # support_loss_mean = tf.reduce_mean(support_loss)
        # adam_gvs = self.optimizer.compute_gradients(support_loss_mean)
        # train_op = self.optimizer.apply_gradients(adam_gvs)
        ###
        # # 接着是基于 query
        # query_pred = self.forward(query_input_layer, fast_weights)
        # # 计算损失函数 L(f_theta'(D'))
        # query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_labels_onehot)
        # # 基于这个 query 定义优化器
        # # gvs = self.optimizer.compute_gradients(query_loss)
        # # train_op = self.optimizer.apply_gradients(gvs)
        # # grads, _ = zip(*gvs)
        #
        # # eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        # # return features, labels, train_op, grads, eval_metric_ops, loss
        # second_order_grads = tf.gradients(query_loss, list(self.weights.values()))
        # query_correct_count = tf.count_nonzero(
        #     tf.equal(tf.argmax(query_labels_onehot, axis=1), tf.argmax(tf.nn.softmax(query_pred, dim=1), axis=1)))

        query_pred = self.forward(query_input_layer, fast_weights)
        # 计算损失函数 L(f_theta'(D'))
        query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_labels_onehot)
        # query_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=query_pred, labels=query_labels)
        query_loss_mean = tf.reduce_mean(query_loss)
        second_order_grads = tf.gradients(query_loss_mean, list(self.weights.values()))
        query_correct_count = tf.count_nonzero(
            tf.equal(tf.argmax(query_labels_onehot, axis=1), tf.argmax(tf.nn.softmax(query_pred, dim=1), axis=1)))

        return (support_features, query_features), (support_labels, query_labels), None, second_order_grads, (support_correct_count, query_correct_count), (support_loss_mean, query_loss_mean)

    def solve_sgd_meta_one_batch(self, sp, qr):
        """
        运行一次 SGD
        :param mini_batch_data:
        :return:
        """
        self.adam_optimizer.increase_n()
        with self.graph.as_default():
            grads, loss = self.sess.run([self.grads, self.loss],
                                           feed_dict={self.features[0]: sp[0],
                                                      self.features[1]: qr[0],
                                                      self.labels[0]: sp[1],
                                                      self.labels[1]: qr[1]})
        sz = len(sp[1]) + len(qr[1])
        comp = sz * self.flops
        return grads, loss, comp, sz

    def solve_sgd_meta_full_data(self, sp, qr):
        """
        运行一次 SGD
        :param mini_batch_data:
        :return:
        """
        self.adam_optimizer.increase_n()
        with self.graph.as_default():
            grads, loss = self.sess.run([self.grads, self.loss],
                                           feed_dict={self.features[0]: sp['x'],
                                                      self.features[1]: qr['x'],
                                                      self.labels[0]: sp['y'],
                                                      self.labels[1]: qr['y']})
        sz = len(sp['y']) + len(qr['y'])
        comp = sz * self.flops
        return grads, loss, comp, sz

    def test_meta(self, sp, qr):
        all_x = np.concatenate((sp['x'], qr['x']), axis=0)
        all_y = np.concatenate((sp['y'], qr['y']), axis=0)
        with self.graph.as_default():
            # tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
            #                                   feed_dict={self.features[0]: sp[0],
            #                                           self.features[1]: qr[0],
            #                                           self.labels[0]: sp[1],
            #                                           self.labels[1]: qr[1]})
            tot_correct, loss = self.sess.run([self.eval_metric_ops[0], self.loss[0]],
                                              feed_dict={self.features[0]: all_x,
                                                         self.labels[0]: all_y})
        return tot_correct, loss


