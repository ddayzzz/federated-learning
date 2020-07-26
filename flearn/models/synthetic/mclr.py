import numpy as np
import tensorflow as tf

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad
from flearn.models.base_model import BaseModel


class Model(BaseModel):

    
    def __init__(self, num_classes, optimizer, seed=1):

        # params
        self.num_classes = num_classes
        super(Model, self).__init__(optimizer=optimizer, seed=seed)
    
    def create_model(self):
        """Model function for Logistic Regression."""
        features = tf.placeholder(tf.float32, shape=[None, 60], name='features')
        labels = tf.placeholder(tf.int64, shape=[None,], name='labels')
        logits = tf.layers.dense(inputs=features, units=self.num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = self.optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss, predictions["classes"]