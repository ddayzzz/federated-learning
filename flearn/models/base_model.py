import numpy as np
import tensorflow as tf
import tqdm
import abc
from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class BaseModel(abc.ABC):

    def __init__(self, optimizer, options, seed=1):
        self.optimizer = optimizer
        self.options = options
        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model()
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    @abc.abstractmethod
    def create_model(self):
        pass
        # return features, labels, train_op, grads, eval_metric_ops, loss

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
        num_samples = len(data['y'])
        with self.graph.as_default():
            model_grads = self.sess.run(self.grads, feed_dict={self.features: data['x'], self.labels: data['y']})
            # flatten
            grads = process_grad(model_grads)

        return num_samples, grads

    def get_loss(self, data):
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: data['x'], self.labels: data['y']})
        return loss

    def solve_inner(self, data, client_id, round_i, num_epochs=1, batch_size=32, hide_output=False):
        """

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
                    with self.graph.as_default():
                        self.sess.run(self.train_op,
                                      feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):
        """
        运行指定的迭代次数
        :param data:
        :param num_iters:
        :param batch_size:
        :return:
        """
        raise NotImplementedError
        # 一下的代码未经测试
        # for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
        #     with self.graph.as_default():
        #         self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        # soln = self.get_params()
        # comp = 0
        # return soln, comp

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
        comp = len(mini_batch_data[1]) * self.flops
        weights = self.get_params()
        return grads, loss, weights, comp

    def test(self, data):
        """
        基于完整的数据集测试
        :param data:
        :return:
        """
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()

    def save(self, path):
        return self.saver.save(self.sess, path)

    def load_checkpoint(self, path):
        self.saver.restore(self.sess, path)