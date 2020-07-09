from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class PerturbedGradientDescent(optimizer.Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer"""
    def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="PGD"):
        super(PerturbedGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu
       
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._mu_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            self._zeros_slot(v, "vstar", self._name)  # 创建 slot, 初始化为 0

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")  # 获得创建得到的 slot

        var_update = state_ops.assign_sub(var, lr_t*(grad + mu_t*(var-vstar)))

        return control_flow_ops.group(*[var_update,])

    
    def _apply_sparse_shared(self, grad, var, indices, scatter_add):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        v_diff = state_ops.assign(vstar, mu_t * (var - vstar), use_locking=self._use_locking)

        with ops.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
            # 对应位置相加 https://www.tensorflow.org/api_docs/python/tf/compat/v1/scatter_add. 这个是针对稀疏向量的版本的
            # indices 估计是梯度的位置
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = state_ops.assign_sub(var, lr_t * scaled_grad)

        return control_flow_ops.group(*[var_update,])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(x, i, v))
    

    def set_params(self, cog, client):
        """
        cog 是 latest model(w_t); client 是模型的参数(w_k^{t+1})
        """
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, cog):
                vstar = self.get_slot(variable, "vstar")  # 创建名为 vstar 的关于 variable 的 slot; 所有的 variable 将会传递到 apply_gradients 或者 minimize
                vstar.load(value, client.sess)
