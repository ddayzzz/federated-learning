import numpy as np
import tensorflow as tf
import tqdm


from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size, process_grad


class Model(object):

    def __init__(self, num_classes, optimizer, seed=1):
        """
        定义 Omniglot 的 CNN 模型
        :param num_classes:
        :param optimizer:
        :param seed:
        """
        # params
        self.num_classes = num_classes
        self.train_lr = 1e-2
        self.meta_lr = 1e-3
        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # TODO 参数; 每次运行一个 task/client
            self.train_op = self.build(K=5, task_num=1, query_shots=10, query_ways=5, sprt_shots=10, sprt_ways=5)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def conv_weights(self, input_channel):
        """
        创建网路的参数
        :param input_channel:
        :param kernel_size:
        :return:
        """
        weights = {}

        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        fc_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
            weights['conv1'] = tf.get_variable('conv1w', [3, 3, input_channel, 32], initializer=conv_initializer)
            weights['b1'] = tf.get_variable('conv1b', initializer=tf.zeros([32]))
            weights['conv2'] = tf.get_variable('conv2w', [3, 3, 32, 32], initializer=conv_initializer)
            weights['b2'] = tf.get_variable('conv2b', initializer=tf.zeros([32]))
            weights['conv3'] = tf.get_variable('conv3w', [3, 3, 32,32], initializer=conv_initializer)
            weights['b3'] = tf.get_variable('conv3b', initializer=tf.zeros([32]))
            weights['conv4'] = tf.get_variable('conv4w', [3, 3, 32, 32], initializer=conv_initializer)
            weights['b4'] = tf.get_variable('conv4b', initializer=tf.zeros([32]))
            # 灰度图像输出 32, RGB 32*32*5
            if input_channel == 1:
                weights['w5'] = tf.get_variable('fc1w', [32, self.num_classes], initializer=fc_initializer)
            elif input_channel == 3:
                weights['w5'] = tf.get_variable('fc1w', [32 * 5 * 5, self.num_classes], initializer=fc_initializer)
            weights['b5'] = tf.get_variable('fc1b', initializer=tf.zeros([self.num_classes]))

        return weights

    def conv_block(self, x, weight, bias, scope):
        """
        build a block with conv2d->batch_norm->pooling
        :param x: 输入的张量
        :param weight: conv2d 的 weight
        :param bias: conv2d 的 bias
        :param scope:
        :param training:
        :return:
        """
        # conv
        x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME', name=scope + '_conv2d') + bias
        # batch norm, activation_fn=tf.nn.relu,
        # NOTICE: must have tf.layers.batch_normalization
        # x = tf.contrib.layers.batch_norm(x, activation_fn=tf.nn.relu)
        with tf.variable_scope('MAML'):
            # 这里将不会被写入到计算图的 normalization 添加到 MAML scope 中, 这样才可以被优化到
            # train is set to True ALWAYS, please refer to https://github.com/cbfinn/maml/issues/9
            # when FLAGS.train=True, we still need to build evaluation network
            x = tf.layers.batch_normalization(x, training=True, name=scope + '_bn', reuse=tf.AUTO_REUSE)
        # relu
        x = tf.nn.relu(x, name=scope + '_relu')
        # pooling
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name=scope + '_pool')
        return x

    def forward(self, x, weights):
        """
        前向操作, 组合相关的参数
        :param x: 输入的张量
        :param weights: 定义好的网络参数
        :param training:
        :return: logits
        """
        hidden1 = self.conv_block(x, weights['conv1'], weights['b1'], 'conv0')
        hidden2 = self.conv_block(hidden1, weights['conv2'], weights['b2'], 'conv1')
        hidden3 = self.conv_block(hidden2, weights['conv3'], weights['b3'], 'conv2')
        hidden4 = self.conv_block(hidden3, weights['conv4'], weights['b4'], 'conv3')

        # get_shape is static shape, (5, 5, 5, 32)
        # print('flatten:', hidden4.get_shape())
        # flatten layer
        # TODO 灰度图像, dim1 = dim2 = 1, 可使用  reduce_mean 去掉
        hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])], name='reshape2')

        output = tf.add(tf.matmul(hidden4, weights['w5']), weights['b5'], name='fc1')

        return output

    def build(self, K, task_num, sprt_ways, sprt_shots, query_ways, query_shots, mode='train'):
        """
        构建网络
        :param K: 第一梯度下降的次数
        :param task_num:
        :param mode: 表示是否是训练模式
        :return:
        """
        """
        support_xb: [task_num, support_batch_size, 28 * 28]
        support_yb: [task_num, support_batch_size]
        query_xb: [task_num, query_batch_size, 28 * 28]
        query_yb: [task_num, query_batch_size]
        """
        # 之所以使用 one-hot 是因为 sparse 版本的交叉熵无法求二阶导数
        self.support_xb = tf.placeholder(dtype=tf.float32, name='support_xb', shape=[task_num, sprt_ways * sprt_shots, 28 * 28])
        self.support_yb = tf.placeholder(dtype=tf.int32, name='support_yb', shape=[task_num, sprt_ways * sprt_shots])
        self.query_xb = tf.placeholder(dtype=tf.float32, name='query_xb', shape=[task_num, query_ways * query_shots, 28 * 28])
        self.query_yb = tf.placeholder(dtype=tf.int32, name='query_yb', shape=[task_num, query_ways * query_shots])

        support_xb_reshaped = tf.reshape(self.support_xb, shape=[task_num, sprt_ways * sprt_shots, 28, 28, 1])
        query_xb_reshaped = tf.reshape(self.support_xb, shape=[task_num, query_ways * query_shots, 28, 28, 1])
        support_yb_one_hot = tf.one_hot(self.support_yb, depth=sprt_ways)
        query_yb_one_hot = tf.one_hot(self.query_yb, depth=query_ways)
        # create or reuse network variable, not including batch_norm variable, therefore we need extra reuse mechnism
        # to reuse batch_norm variables.
        self.weights = self.conv_weights(input_channel=1)
        # TODO: meta-test is sort of test stage.
        training = True if mode is 'train' else False

        def meta_task(input):
            """
            在 task_num 的维度上基于
            :param input: support_x, support_y, query_x, query_y
            :return:
            """
            support_x, support_y, query_x, query_y = input
            # to record the op in t update step.
            query_preds, query_losses, query_accs = [], [], []

            # ==================================
            # REUSE       True        False
            # Not exist   Error       Create one
            # Existed     reuse       Error
            # ==================================
            # That's, to create variable, you must turn off reuse
            support_pred = self.forward(support_x, self.weights)

            support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=support_y)
            support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(support_pred, dim=1), axis=1),
                                                      tf.argmax(support_y, axis=1))
            # compute gradients
            grads = tf.gradients(support_loss, list(self.weights.values()))
            # grad and variable dict
            gvs = dict(zip(self.weights.keys(), grads))

            # theta_pi = theta - alpha * grads
            fast_weights = dict(zip(self.weights.keys(),
                                    [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))
            # use theta_pi to forward meta-test
            query_pred = self.forward(query_x, fast_weights)
            # meta-test loss
            query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
            # record T0 pred and loss for meta-test
            query_preds.append(query_pred)
            query_losses.append(query_loss)

            # continue to build T1-TK steps graph
            for _ in range(1, K):
                # T_k loss on meta-train
                # we need meta-train loss to fine-tune the task and meta-test loss to update theta
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.forward(support_x, fast_weights),
                                                               labels=support_y)
                # compute gradients
                grads = tf.gradients(loss, list(fast_weights.values()))
                # compose grad and variable dict
                gvs = dict(zip(fast_weights.keys(), grads))
                # update theta_pi according to varibles
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.train_lr * gvs[key]
                                                              for key in fast_weights.keys()]))
                # forward on theta_pi
                query_pred = self.forward(query_x, fast_weights)
                # we need accumulate all meta-test losses to update theta
                query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
                query_preds.append(query_pred)
                query_losses.append(query_loss)

            # compute every steps' accuracy on query set
            for i in range(K):
                query_accs.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_preds[i], dim=1), axis=1),
                                                              tf.argmax(query_y, axis=1)))
            # we just use the first step support op: support_pred & support_loss, but igonre these support op
            # at step 1:K-1.
            # however, we return all pred&loss&acc op at each time steps.
            result = [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]

            return result

        # return: [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]
        out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K]
        result = tf.map_fn(meta_task, elems=(support_xb_reshaped, support_yb_one_hot, query_xb_reshaped, query_yb_one_hot),
                           dtype=out_dtype, parallel_iterations=task_num, name='map_fn')
        # 输出结果
        support_pred_tasks, support_loss_tasks, support_acc_tasks, \
        query_preds_tasks, query_losses_tasks, query_accs_tasks = result
        # average loss
        self.support_loss = tf.reduce_sum(support_loss_tasks) / task_num
        # [avgloss_t1, avgloss_t2, ..., avgloss_K],
        self.query_losses = tf.stack([tf.reduce_sum(query_losses_tasks[j]) / task_num for j in range(K)], axis=0)
        # average accuracy
        self.support_acc = tf.reduce_sum(support_acc_tasks) / task_num
        # average accuracies
        self.query_accs = tf.stack([tf.reduce_sum(query_accs_tasks[j]) / task_num for j in range(K)], axis=0)

        # # add batch_norm ops before meta_op
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        # 	# TODO: the update_ops must be put before tf.train.AdamOptimizer,
        # 	# otherwise it throws Not in same Frame Error.
        # 	meta_loss = tf.identity(self.query_losses[-1])

        # meta-train optim
        optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
        # meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
        gvs = optimizer.compute_gradients(self.query_losses[-1])
        # meta-train grads clipping
        gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
        # update theta
        return optimizer.apply_gradients(gvs)

    def set_params(self, model_params: dict):
        with self.graph.as_default():
            # all_var_keys, all_vars = self.weights.items()
            # for variable, value in zip(all_vars, model_params):
            #     variable.load(value, self.sess)
            for var_name, var in self.weights.items():
                var.load(model_params[var_name], self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(list(self.weights.values()))
        return dict(zip(self.weights.keys(), model_params))

    def solve_gd(self, sprt_data, query_data):
        sprt_xb, sprt_yb, query_xb, query_yb = sprt_data['x'], sprt_data['y'], query_data['x'], query_data['y']
        feed_dict = dict(zip([self.support_xb, self.support_yb, self.query_xb, self.query_yb], map(lambda x: np.expand_dims(x, 0), [sprt_xb, sprt_yb, query_xb, query_yb])))
        with self.graph.as_default():
            support_loss, support_acc, query_losses, query_accs, _ = self.sess.run([self.support_loss, self.support_acc, self.query_losses, self.query_accs, self.train_op], feed_dict=feed_dict)
        params = self.get_params()
        comp = (len(sprt_yb) + len(query_yb)) * self.flops
        return params, comp, (support_loss, support_acc, query_losses, query_accs)

    def test(self, data):
        """
        这里是基于
        :param data: 这届
        :return:
        """
        xb = data['x']
        yb = data['y']
        feed_dict = dict(zip([self.support_xb, self.support_yb], map(lambda x: np.expand_dims(x, 0), [xb, yb])))
        with self.graph.as_default():
            support_loss, support_acc = self.sess.run(
                [self.support_loss, self.support_acc],
                feed_dict=feed_dict)
        return support_acc, support_loss

    def close(self):
        self.sess.close()
