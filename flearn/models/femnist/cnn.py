import tensorflow as tf
import tqdm

from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size, process_grad
from flearn.models.base_model import BaseModel


class Model(BaseModel):

    def __init__(self, num_classes, image_size, optimizer, seed=1):
        # params
        self.num_classes = num_classes
        self.image_size = image_size
        super(Model, self).__init__(optimizer=optimizer, seed=seed)

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
        train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss
