import numpy as np
import tqdm
import tensorflow as tf
from tensorflow.contrib import rnn
from flearn.utils.tf_utils import graph_size, process_sparse_grad
from flearn.utils.language_utils import letter_to_vec, word_to_indices
from flearn.utils.model_utils import batch_data, batch_data_multiple_iters


def process_x(raw_x_batch):
    """
    将数据集中的句子
    :param raw_x_batch:
    :return:
    """
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_y(raw_y_batch):
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch


class Model(object):
    def __init__(self, seq_len, num_classes, n_hidden, optimizer, seed):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph, config=config)

        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer):
        features = tf.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.get_variable("embedding", [self.num_classes, 8])
        x = tf.nn.embedding_lookup(embedding, features)
        labels = tf.placeholder(tf.int32, [None, self.num_classes])
        
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        pred = tf.layers.dense(inputs=outputs[:,-1,:], units=self.num_classes)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())


        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, grads, eval_metric_ops, loss


    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        """
        获得梯度, 过大的 batch 会造成 OOM
        :param data:
        :param model_len:
        :return:
        """
        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        processed_samples = 0

        if num_samples < 50:
            input_data = process_x(data['x'])
            target_data = process_y(data['y'])
            with self.graph.as_default():
                model_grads = self.sess.run(self.grads, 
                    feed_dict={self.features: input_data, self.labels: target_data})
            grads = process_sparse_grad(model_grads)
            processed_samples = num_samples

        else:  # in order to fit into memory, compute gradients in a batch of size 50, and subsample a subset of points to approximate
            for i in range(min(int(num_samples / 50), 4)):
                input_data = process_x(data['x'][50*i:50*(i+1)])
                target_data = process_y(data['y'][50*i:50*(i+1)])

                with self.graph.as_default():
                    model_grads = self.sess.run(self.grads,
                    feed_dict={self.features: input_data, self.labels: target_data})
            
                flat_grad = process_sparse_grad(model_grads)
                grads = np.add(grads, flat_grad)

            grads = grads * 1.0 / min(int(num_samples/50), 4)
            processed_samples = min(int(num_samples / 50), 4) * 50

        return processed_samples, grads
    
    def solve_inner(self, data, client_id, round_i, num_epochs=1, batch_size=32, hide_output=False):
        """
        运行若干次 epoch
        :param data:
        :param client_id:
        :param round_i:
        :param num_epochs:
        :param batch_size:
        :param hide_output:
        :return:
        """
        with tqdm.trange(num_epochs, disable=hide_output) as t:
            for epoch in t:
                t.set_description(f'Client: {client_id}, Round: {round_i + 1}, Epoch :{epoch + 1}')
                for batch_idx, (X, y) in enumerate(batch_data(data, batch_size)):
                    input_data = process_x(X)
                    target_data = process_y(y)
                    with self.graph.as_default():
                        self.sess.run(self.train_op,
                                      feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):
        """
        运行指定数量的迭代次数
        :param data:
        :param num_iters: 当 num_iters = 1 时, 则运行的是标准的 SGD
        :param batch_size:
        :return:
        """

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            input_data = process_x(X)
            target_data = process_y(y)
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        comp = 0
        return soln, comp

    def _test_all(self, data):
        """
        基于所有的数据测试
        :param data:
        :return:
        """
        x_vecs = process_x(data['x'])
        labels = process_y(data['y'])
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels})
        return tot_correct, loss

    def test(self, data):
        """
        基于某个数据集得到准确个数和平均的损失
        :param data:
        :return: 准确个数, 平均的损失
        """
        data_size = len(data['y'])
        if data_size <= 1000:
            return self._test_all(data)
        # 分为若干 batch
        tot_correct, tot_loss, num_samples = 0, 0.0, 0
        for X, y in batch_data(data, batch_size=200, shuffle=False):
            # X 转为矩阵
            # y 为 one-hot
            x_vecs = process_x(X)
            labels = process_y(y)
            num_sample = len(labels)
            with self.graph.as_default():
                correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: x_vecs, self.labels: labels})
            tot_correct += correct
            num_samples += num_sample
            # loss -> 这个batch 的损失
            tot_loss += loss * num_sample
        return tot_correct, (tot_loss / num_samples)
    
    def close(self):
        self.sess.close()

