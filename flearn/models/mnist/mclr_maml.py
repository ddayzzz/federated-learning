import numpy as np
import tensorflow as tf
from flearn.utils.tf_utils import graph_size


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
        self.train_lr = 0.01
        self.meta_lr = 0.001
        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # TODO 参数; 每次运行一个 task/client
            self.train_op = self.build(K=5)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def fc_weights(self):
        """
        创建网路的参数. 网络的参数保存在
        :param input_channel:
        :param kernel_size:
        :return: 参数 dict: Dict[name] -> variable
        """
        weights = {}
        fc_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
            weights['fc1_w'] = tf.get_variable('fc1_w', [28 * 28, self.num_classes], initializer=fc_initializer)
            weights['fc1_b'] = tf.get_variable('fc1_b', initializer=tf.zeros([self.num_classes]))
        return weights


    def forward(self, x, weights):
        """
        前向操作, 组合相关的参数
        :param x: 输入的张量
        :param weights: 定义好的网络参数
        :param training:
        :return: logits
        """
        output = tf.add(tf.matmul(x, weights['fc1_w']), weights['fc1_b'], name='fc1')
        return output

    def meta_task(self, support_x, support_y, query_x, query_y, K):
        """
        运行一个 meta-task 的任务
        :param support_x:
        :param support_y:
        :param query_x:
        :param query_y:
        :return:
        """
        # to record the op in t update step.
        query_preds, query_losses, query_correct_count = [], [], []
        # 基于 support_set 进行 forward 运算
        support_pred = self.forward(support_x, self.weights)
        # support set 的损失函数
        support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=support_y)
        # support set 预测正确的数量
        support_correct_count = tf.count_nonzero(tf.equal(tf.argmax(support_y, axis=1), tf.argmax(tf.nn.softmax(support_pred, dim=1), axis=1)))

        # 计算梯度
        grads = tf.gradients(support_loss, list(self.weights.values()))
        # 计算后的梯度保存为 key->variable 字典
        gvs = dict(zip(self.weights.keys(), grads))

        # theta_pi = theta - alpha * grads
        fast_weights = dict(zip(self.weights.keys(),
                                [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))
        # theta_pi' 在query set 上运行
        query_pred = self.forward(query_x, fast_weights)
        # meta-test loss
        query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
        # record T0 pred and loss for meta-test
        query_preds.append(query_pred)
        query_losses.append(query_loss)

        # continue to build T1-TK steps graph
        for _ in range(1, K):
            # 这里的总体步骤同上, 先计算 fast_weight, 再利用 query set 来学习
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
            # query_correct_count.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_preds[i], dim=1), axis=1),
            #                                               tf.argmax(query_y, axis=1)))
            correct_cnt = tf.count_nonzero(
                tf.equal(tf.argmax(query_y, axis=1), tf.argmax(tf.nn.softmax(query_preds[i], dim=1), axis=1)))
            query_correct_count.append(correct_cnt)

        return support_loss, support_correct_count, query_losses, query_correct_count

    def build(self, K, mode='train'):
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
        is_train = True if mode == 'train' else False
        # 之所以使用 one-hot 是因为 sparse 版本的交叉熵无法求二阶导数
        self.support_xb = tf.placeholder(dtype=tf.float32, name='support_xb', shape=[None, 28 * 28])
        self.support_yb = tf.placeholder(dtype=tf.int32, name='support_yb', shape=[None])
        self.query_xb = tf.placeholder(dtype=tf.float32, name='query_xb', shape=[None, 28 * 28])
        self.query_yb = tf.placeholder(dtype=tf.int32, name='query_yb', shape=[None])

        support_yb_one_hot = tf.one_hot(self.support_yb, depth=self.num_classes)
        query_yb_one_hot = tf.one_hot(self.query_yb, depth=self.num_classes)
        # create or reuse network variable, not including batch_norm variable, therefore we need extra reuse mechnism
        # to reuse batch_norm variables.
        self.weights = self.fc_weights()
        # TODO: meta-test is sort of test stage.
        # 然而没有设用这个代码
        # training = True if mode is 'train' else False

        # return: [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]
        # out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K]
        # result = tf.map_fn(meta_task, elems=(support_xb_reshaped, support_yb_one_hot, query_xb_reshaped, query_yb_one_hot),
        #                    dtype=out_dtype, parallel_iterations=task_num, name='map_fn')
        # 输出结果
        support_loss, support_correct_count, query_losses, query_correct_counts = self.meta_task(support_x=self.support_xb, support_y=support_yb_one_hot, query_x=self.query_xb, query_y=query_yb_one_hot, K=K)
        # support 上的损失, 向量. 需要把 loss 平均一下. scalar
        self.support_loss = tf.reduce_mean(support_loss)
        # [T_1, T_2,...,T_K] 上的平均. -> [K]
        self.query_losses = tf.stack([tf.reduce_mean(query_losses[j]) for j in range(K)], axis=0)
        # support 上的准确的个数 -> scalar
        self.support_correct_count = support_correct_count
        # query 上的准确的个数 -> [K]
        self.query_correct_count = tf.stack(query_correct_counts, axis=0)
        # # add batch_norm ops before meta_op
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        # 	# TODO: the update_ops must be put before tf.train.AdamOptimizer,
        # 	# otherwise it throws Not in same Frame Error.
        # 	meta_loss = tf.identity(self.query_losses[-1])
        # if is_train:
        #     tf.summary.scalar('train/support_loss', self.support_loss)
        #     tf.summary.scalar('train/support_acc', self.support_acc)
        #     for j in range(K):
        #         tf.summary.scalar('train/query_loss_step ' + str(j + 1), self.query_losses[j])
        #         tf.summary.scalar('train/query_acc_step ' + str(j + 1), self.query_accs[j])
        # meta-train optim
        optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
        # meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
        gvs = optimizer.compute_gradients(self.query_losses[-1])
        # meta-train grads clipping
        # gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
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
        # 由于每一个客户端视为一个 task, 所以需要扩张第一维
        # feed_dict = dict(zip([self.support_xb, self.support_yb, self.query_xb, self.query_yb], map(lambda x: np.expand_dims(x, 0), [sprt_xb, sprt_yb, query_xb, query_yb])))
        feed_dict = {self.support_xb: sprt_xb, self.support_yb: sprt_yb, self.query_xb: query_xb,
                     self.query_yb: query_yb}
        with self.graph.as_default():
            support_loss, support_cnt, query_losses, query_cnt, _ = self.sess.run([self.support_loss, self.support_correct_count, self.query_losses, self.query_correct_count, self.train_op], feed_dict=feed_dict)
        params = self.get_params()
        num_samples = len(sprt_yb) + len(query_yb)
        comp = num_samples * self.flops
        return params, comp, num_samples, (support_loss, support_cnt, query_losses, query_cnt)

    def test(self, support_data, query_data):
        """
        同时计算
        :param data: 这届
        :return:
        """
        sprt_xb, sprt_yb, query_xb, query_yb = support_data['x'], support_data['y'], query_data['x'], query_data['y']
        # 由于每一个客户端视为一个 task, 所以需要扩张第一维
        feed_dict = {self.support_xb: sprt_xb, self.support_yb: sprt_yb, self.query_xb: query_xb,
                     self.query_yb: query_yb}
        with self.graph.as_default():
            support_loss, support_cnt, query_losses, query_cnt, = self.sess.run(
                [self.support_loss, self.support_correct_count, self.query_losses, self.query_correct_count],
                feed_dict=feed_dict)
        return support_loss, support_cnt, query_losses, query_cnt

    def close(self):
        self.sess.close()
