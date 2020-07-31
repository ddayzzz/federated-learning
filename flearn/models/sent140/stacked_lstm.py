import tensorflow as tf
from tensorflow.contrib import rnn
import tqdm
import numpy as np
import sys
import os
from flearn.utils.model_utils import batch_data
from flearn.models.base_model import BaseModel
from flearn.models.sent140.language_utils import line_to_indices, get_word_emb_arr, val_to_vec

ROOT = os.path.dirname(os.path.realpath(__file__))
VOCAB_DIR = os.path.join(ROOT, 'embs.json')


class Model(BaseModel):

    def __init__(self, seq_len, num_classes, n_hidden, emb_arr, options, optimizer, seed=1):
        # params
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.word_emb, self.word2id, self.id2word = get_word_emb_arr(VOCAB_DIR)
        self.vocab_size = len(self.word2id)
        # if emb_arr:
        #     self.emb_arr = emb_arr

        super(Model, self).__init__(optimizer=optimizer, seed=seed, options=options)

    def create_model(self):
        features = tf.placeholder(tf.int32, [None, self.seq_len])
        # 400001 的数量, 100
        # embedding = tf.get_variable('embedding', [self.vocab_size + 1, self.n_hidden], dtype=tf.float32, trainable=False)
        # x = tf.cast(tf.nn.embedding_lookup(embedding, features), tf.float32)
        embs = tf.Variable(self.word_emb, dtype=tf.float32, trainable=False)
        x = tf.nn.embedding_lookup(embs, features)
        labels = tf.placeholder(tf.float32, [None, self.num_classes])

        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        # TODO 这里应该要激活才对
        fc1 = tf.layers.dense(inputs=outputs[:, -1, :], units=128, activation=tf.nn.relu)
        pred = tf.layers.dense(inputs=fc1, units=self.num_classes)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
        # train_op = self.optimizer.minimize(
        #     loss=loss,
        #     global_step=tf.train.get_global_step())
        grads_and_vars = self.optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, grads, eval_metric_ops, loss

    def process_x(self, raw_x_batch, max_words=25):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [line_to_indices(e, self.word2id, max_words) for e in x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        y_batch = [val_to_vec(self.num_classes, e) for e in y_batch]
        y_batch = np.array(y_batch)
        return y_batch

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                           feed_dict={self.features: self.process_x(mini_batch_data[0]),
                                                      self.labels: self.process_y(mini_batch_data[1])})
        sz = len(mini_batch_data[1])
        comp = sz * self.flops
        weights = self.get_params()
        return grads, loss, weights, comp

    def test(self, data):
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: self.process_x(data['x']), self.labels: self.process_y(data['y'])})
        return tot_correct, loss