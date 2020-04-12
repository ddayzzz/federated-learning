import torch
import numpy as np


def zeros(shape, gpu=False, **kwargs):
    # return torch.zeros(*shape, **kwargs).cuda() if use_gpu and gpu else torch.zeros(*shape)
    # TODO 创建 CUDA 的张量
    return torch.zeros(*shape, **kwargs).cuda()

def get_flat_params_from(model):
    """
    flat model parameters(parameters() 和 state_dict 不同, 前者返回一个生成器)
    :param model:
    :return:
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def model_parameters_shape_list(model):
    return [x.size() for x in model.parameters()]


def from_flatten_to_parameter(shape_info, flat_params):
    new_params = []
    prev_ind = 0
    for shape in shape_info:
        # 计算 flat 后的乘积
        flat_size = int(np.prod(list(shape)))
        # 恢复值
        new_params.append(flat_params[prev_ind:prev_ind + flat_size].view(shape))
        prev_ind += flat_size
    return new_params

def set_flat_params_to(model, flat_params):
    """
    set model parameters from flatten parameters
    :param model:
    :param flat_params:
    :return:
    """
    prev_ind = 0
    for param in model.parameters():
        # 计算 flat 后的乘积
        flat_size = int(np.prod(list(param.size())))
        # 恢复值
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_grad_dict(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    """

    :param output:
    :param inputs:
    :param filter_input_ids:
    :param retain_graph:
    :param create_graph:
    :return:
    """
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)
    # 计算梯度: output 关于 params 的梯度;
    # retain_graph 和 create_graph 的值一般相同; 前者为 False 表示计算后销毁计算图
    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = dict()
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads[i] = zeros(param.data.view(-1).shape)
        else:
            out_grads[i] = grads[j]
            j += 1

    for param in params:
        param.grad = None
    return out_grads


def get_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(zeros(param.data.view(-1).shape))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads