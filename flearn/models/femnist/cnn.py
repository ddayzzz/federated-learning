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

    def create_model(self):
        """
        创建基本你的模型
        :param optimizer:
        :return:
        """
        features = tf.placeholder(tf.float32, shape=[None, self.image_size * self.image_size], name='features')
        input_layer = tf.reshape(features, [-1, self.image_size, self.image_size, 1], name='features_reshaped')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        grads_and_vars = self.optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        self.use_meta_sgd = self.options.get('meta_algo') == 'meta_sgd'
        if self.use_meta_sgd:
            # 这个变量作为被训练的参数
            grad_wise_alpha = []
            grad_num = len(grads)
            for grad_index in range(grad_num):
                alpha = tf.get_variable(name='alpha_for_theta_{}'.format(grad_index), shape=grads[grad_index].get_shape().as_list(), dtype=tf.float32, initializer=tf.constant_initializer(self.options['lr']), trainable=True)
                grad_wise_alpha.append(alpha)
            # 将对应的梯度组合
            alpha_times_grads = []
            for grad_alpha, grad_and_var in zip(grad_wise_alpha, grads_and_vars):
                alpha_times_grad = grad_alpha * grad_and_var[0]
                alpha_times_grads.append((alpha_times_grad, grad_and_var[1]))
            train_op = self.optimizer.apply_gradients(alpha_times_grads, global_step=tf.train.get_global_step())
            # TODO 这里直接设置对应的参数
            self.grad_wise_alpha = grad_wise_alpha
        else:

            train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss

    def solve_sgd(self, mini_batch_data):
        """
        运行一次 SGD
        :param mini_batch_data:
        :return:
        """
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                           feed_dict={self.features: mini_batch_data[0],
                                                      self.labels: mini_batch_data[1]})
        sz = len(mini_batch_data[1])
        comp = sz * self.flops
        weights = self.get_params()
        return grads, loss, weights, comp

    # 由于保存另外一个可以被训练的参数 alpha, 所以需要重新设定
    # def solve_sgd(self, mini_batch_data):
    #     if not self.use_meta_sgd:
    #         return super(Model, self).solve_sgd(mini_batch_data)
    #     with self.graph.as_default():
    #         grads, alpha_grads, loss, _ = self.sess.run([self.grads, self.grad_wise_alpha, self.loss, self.train_op],
    #                                        feed_dict={self.features: mini_batch_data[0],
    #                                                   self.labels: mini_batch_data[1]})
    #     comp = len(mini_batch_data[1]) * self.flops
    #     weights = self.get_params()
    #     return list(grads) + alpha_grads, loss, weights, comp

    def solve_inner_support_query(self, data, client_id, round_i, num_epochs=1, batch_size=32, hide_output=False):
        """
        :param data:
        :param client_id:
        :param round_i:
        :param num_epochs:
        :param batch_size:
        :param hide_output:
        :return:
        """
        grads = []
        num_inter = 0
        with tqdm.trange(num_epochs, disable=hide_output) as t:
            for epoch in t:
                t.set_description(f'Client: {client_id}, Round: {round_i + 1}, Epoch :{epoch + 1}')
                for batch_idx, (X, y) in enumerate(batch_data(data, batch_size)):
                    with self.graph.as_default():
                        iter_grads, _ = self.sess.run([self.grads, self.train_op], feed_dict={self.features: X, self.labels: y})
                    num_inter += 1
                    grads.append(iter_grads)
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        # 这里的 grad 的和必须要求和后除以迭代的次数
        grads_sum = [np.zeros_like(g) for g in grads[0]]
        for i in grads:
            for j, grad in enumerate(i):
                grads_sum[j] += grad
        grads_mean = [g / num_inter for g in grads_sum]
        return grads_mean, comp

    # def test(self, data):
    #     """
    #     基于完整的数据集测试
    #     :param data:
    #     :return:
    #     """
    #     with self.graph.as_default():
    #         tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
    #                                           feed_dict={self.features: data[0], self.labels: data[1]})
    #     return tot_correct, loss

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
            (weights['fc2w'], weights['fc2b']) = self.create_fc_variables(2018, self.num_classes, 'fc2')
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

    def create_model2(self):
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
        support_labels = tf.one_hot(support_labels, depth=self.num_classes, name='support_labels_onehot')
        query_labels = tf.one_hot(query_labels, depth=self.num_classes, name='query_labels_onehot')
        # 基于 support, 计算一次参数
        self.weights = self.create_params()
        support_pred_logitis = self.forward(support_input_layer, self.weights)
        support_correct_count = tf.count_nonzero(
            tf.equal(tf.argmax(support_labels, axis=1), tf.argmax(tf.nn.softmax(support_pred_logitis, dim=1), axis=1)))
        #
        grads = tf.gradients(support_loss, list(self.weights.values()))
        # 计算后的梯度保存为 key->variable 字典
        gvs = dict(zip(self.weights.keys(), grads))

        # theta' = theta - alpha * grads
        fast_weights = dict(
            zip(self.weights.keys(), [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))
        # # 这里的网络使用
        # labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        # conv1 = tf.layers.conv2d(
        #     inputs=input_layer,
        #     filters=32,
        #     kernel_size=[5, 5],
        #     padding="same",
        #     activation=tf.nn.relu)
        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # conv2 = tf.layers.conv2d(
        #     inputs=pool1,
        #     filters=64,
        #     kernel_size=[5, 5],
        #     padding="same",
        #     activation=tf.nn.relu)
        # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        # dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
        # logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        # predictions = {
        #     "classes": tf.argmax(input=logits, axis=1),
        #     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        # }
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # grads_and_vars = self.optimizer.compute_gradients(loss)
        # grads, _ = zip(*grads_and_vars)
        # use_meta_sgd = self.options.get('meta_algo') == 'meta_sgd'
        # if use_meta_sgd:
        #     # 这个变量作为被训练的参数
        #     grad_num = len(grads)
        #     self.alpha = tf.get_variable(name='alpha', shape=[grads], dtype=tf.float32, initializer=tf.initializers.constant(self.options['lr']))
        # train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        #
        # eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        # return features, labels, train_op, grads, eval_metric_ops, loss
