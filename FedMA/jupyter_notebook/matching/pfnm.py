from .utils import *
from .gaus_marginal_matching import match_local_atoms
import copy

def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j):

    match_norms = ((weights_j_l + global_weights) ** 2 / (sigma_inv_j + global_sigmas)).sum(axis=1) - (
                global_weights ** 2 / global_sigmas).sum(axis=1)

    return match_norms


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts), 10)
    param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    param_cost += np.log(counts / (J - counts))

    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost


def matching_upd_j(weights_j, global_weights, sigma_inv_j, global_sigmas, prior_mean_norm, prior_inv_sigma,
                   popularity_counts, gamma, J):

    L = global_weights.shape[0]

    full_cost = compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)

    row_ind, col_ind = linear_sum_assignment(-full_cost)
    # please note that this can not run on non-Linux systems
    #row_ind, col_ind = solve_dense(-full_cost)

    assignment_j = []

    new_L = L

    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_weights[i] += weights_j[l]
            global_sigmas[i] += sigma_inv_j
        else:  # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_weights = np.vstack((global_weights, prior_mean_norm + weights_j[l]))
            global_sigmas = np.vstack((global_sigmas, prior_inv_sigma + sigma_inv_j))

    return global_weights, global_sigmas, popularity_counts, assignment_j


def objective(global_weights, global_sigmas):
    obj = ((global_weights) ** 2 / global_sigmas).sum()
    return obj


def patch_weights(w_j, L_next, assignment_j_c):
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    return new_w_j

def block_patching(w_j, L_next, assignment_j_c, layer_index, model_meta_data, 
                                matching_shapes=None, 
                                layer_type="fc", 
                                dataset="cifar10",
                                network_name="lenet"):
    """
    In CNN, weights patching needs to be handled block-wisely
    We handle all conv layers and the first fc layer connected with the output of conv layers here
    """
    if assignment_j_c is None:
        return w_j

    layer_meta_data = model_meta_data[2 * layer_index - 2]
    prev_layer_meta_data = model_meta_data[2 * layer_index - 2 - 2]

    if layer_type == "conv":    
        new_w_j = np.zeros((w_j.shape[0], L_next*(layer_meta_data[-1]**2)))

        # we generate a sequence of block indices
        block_indices = [np.arange(i*layer_meta_data[-1]**2, (i+1)*layer_meta_data[-1]**2) for i in range(L_next)]
        ori_block_indices = [np.arange(i*layer_meta_data[-1]**2, (i+1)*layer_meta_data[-1]**2) for i in range(layer_meta_data[1])]
        for ori_id in range(layer_meta_data[1]):
            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]

    elif layer_type == "fc":
        # we need to estimate the output shape here:
        if network_name == "simple-cnn":
            if dataset in ("cifar10", "cinic10"):
                shape_estimator = SimpleCNNContainerConvBlocks(input_channel=3, num_filters=matching_shapes, kernel_size=5)
            elif dataset == "mnist":
                shape_estimator = SimpleCNNContainerConvBlocks(input_channel=1, num_filters=matching_shapes, kernel_size=5)
        elif network_name == "moderate-cnn":
            if dataset in ("cifar10", "cinic10"):
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=matching_shapes)
            elif dataset == "mnist":
                shape_estimator = ModerateCNNContainerConvBlocksMNIST(num_filters=matching_shapes)

        if dataset in ("cifar10", "cinic10"):
            dummy_input = torch.rand(1, 3, 32, 32)
        elif dataset == "mnist":
            dummy_input = torch.rand(1, 1, 28, 28)
        estimated_output = shape_estimator(dummy_input)
        new_w_j = np.zeros((w_j.shape[0], estimated_output.view(-1).size()[0]))
        logger.info("estimated_output shape : {}".format(estimated_output.size()))
        logger.info("meta data of previous layer: {}".format(prev_layer_meta_data))
        
        block_indices = [np.arange(i*estimated_output.size()[-1]**2, (i+1)*estimated_output.size()[-1]**2) for i in range(L_next)]

        ori_block_indices = [np.arange(i*estimated_output.size()[-1]**2, (i+1)*estimated_output.size()[-1]**2) for i in range(prev_layer_meta_data[0])]

        for ori_id in range(prev_layer_meta_data[0]):
            #logger.info("{} ------------ to ------------ {}".format(block_indices[assignment_j_c[ori_id]], ori_block_indices[ori_id]))
            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]
    return new_w_j


def process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0):
    J = len(batch_weights)
    sigma_bias = sigma
    sigma0_bias = sigma0
    mu0_bias = 0.1
    softmax_bias = [batch_weights[j][-1] for j in range(J)]
    softmax_inv_sigma = [s / sigma_bias for s in last_layer_const]
    softmax_bias = sum([b * s for b, s in zip(softmax_bias, softmax_inv_sigma)]) + mu0_bias / sigma0_bias
    softmax_inv_sigma = 1 / sigma0_bias + sum(softmax_inv_sigma)
    return softmax_bias, softmax_inv_sigma


def match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, it):
    J = len(weights_bias)

    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])

    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior

    global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
    global_sigmas = np.outer(np.ones(global_weights.shape[0]), sigma_inv_prior + sigma_inv_layer[group_order[0]])

    popularity_counts = [1] * global_weights.shape[0]

    assignment = [[] for _ in range(J)]

    assignment[group_order[0]] = list(range(global_weights.shape[0]))

    ## Initialize
    for j in group_order[1:]:
        global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                        global_weights,
                                                                                        sigma_inv_layer[j],
                                                                                        global_sigmas, prior_mean_norm,
                                                                                        sigma_inv_prior,
                                                                                        popularity_counts, gamma, J)
        assignment[j] = assignment_j

    ## Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order:  # random_order:
            to_delete = []
            ## Remove j
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj), assignment[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                logger.info('Warning - weird unmatching')
                else:
                    global_weights[i] = global_weights[i] - batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]

            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)

            ## Match j
            global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                            global_weights,
                                                                                            sigma_inv_layer[j],
                                                                                            global_sigmas,
                                                                                            prior_mean_norm,
                                                                                            sigma_inv_prior,
                                                                                            popularity_counts, gamma, J)
            assignment[j] = assignment_j

    logger.info('Number of global neurons is %d, gamma %f' % (global_weights.shape[0], gamma))
    logger.info("***************Shape of global weights after match: {} ******************".format(global_weights.shape))
    return assignment, global_weights, global_sigmas


def layer_wise_group_descent(batch_weights, layer_index, batch_frequencies, sigma_layers, 
                                sigma0_layers, gamma_layers, it, 
                                model_meta_data, 
                                model_layer_type,
                                n_layers,
                                matching_shapes,
                                args):
    """
    We implement a layer-wise matching here:
    """
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    last_layer_const = []
    total_freq = sum(batch_frequencies)
    for f in batch_frequencies:
        last_layer_const.append(f / total_freq)

    # J: number of workers
    J = len(batch_weights)
    # init_num_kernel: the number of conv filters in the first conv layer 
    init_num_kernel = batch_weights[0][0].shape[0]

    # for saving (#channel * k * k)
    init_channel_kernel_dims = []
    for bw in batch_weights[0]:
        if len(bw.shape) > 1:
            init_channel_kernel_dims.append(bw.shape[1])
    logger.info("init_channel_kernel_dims: {}".format(init_channel_kernel_dims))
    
    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None

    sigma = sigma_layers[layer_index - 1]
    sigma_bias = sigma_bias_layers[layer_index - 1]
    gamma = gamma_layers[layer_index - 1]
    sigma0 = sigma0_layers[layer_index - 1]
    sigma0_bias = sigma0_bias_layers[layer_index - 1]

    if layer_index <= 1:
        weights_bias = [np.hstack((batch_weights[j][0], batch_weights[j][layer_index * 2 - 1].reshape(-1, 1))) for j in range(J)]

        sigma_inv_prior = np.array(
            init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias])
        mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias])

        # handling 2-layer neural network
        if n_layers == 2:
            sigma_inv_layer = [
                np.array(D * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
        else:
            sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias]) for j in range(J)]

    elif layer_index == (n_layers - 1) and n_layers > 2:
        # our assumption is that this branch will consistently handle the last fc layers
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))

        # if first_fc_identifier:
        #     weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, 
        #                                 batch_weights[j][2 * layer_index - 1].reshape(-1, 1),
        #                                 batch_weights[j][2 * layer_index])) for j in range(J)]
        # else:
        #     weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, 
        #                                 batch_weights[j][2 * layer_index - 1].reshape(-1, 1),
        #                                 batch_weights[j][2 * layer_index])) for j in range(J)]

        # we switch to ignore the last layer here:
        if first_fc_identifier:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, 
                                        batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]
        else:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, 
                                        batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]


        sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
        mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
        
        # hwang: this needs to be handled carefully
        #sigma_inv_layer = [np.array([1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
        #sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in range(J)]

        #sigma_inv_layer = [np.array((matching_shapes[layer_index - 2]) * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
        sigma_inv_layer = [np.array((matching_shapes[layer_index - 2]) * [1 / sigma] + [1 / sigma_bias]) for j in range(J)]

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]

        if 'conv' in layer_type or 'features' in layer_type:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2], batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]

        elif 'fc' in layer_type or 'classifier' in layer_type:
            # we need to determine if the type of the current layer is the same as it's previous layer
            # i.e. we need to identify if the fully connected layer we're working on is the first fc layer after the conv block
            #first_fc_identifier = ('fc' in layer_type and 'conv' in prev_layer_type)
            first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))
            #logger.info("first_fc_identifier: {}".format(first_fc_identifier))
            if first_fc_identifier:
                weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]
            else:
                weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]          

        sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
        mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
        sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in range(J)]

    logger.info("Layer index: {}, init_num_kernel: {}".format(layer_index, init_num_kernel))
    logger.info("weights bias: {}".format(weights_bias[0].shape))
    logger.info("sigma_inv_prior shape: {}".format(sigma_inv_prior.shape))
    logger.info("mean_prior shape: {}".format(mean_prior.shape))

    assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                  sigma_inv_prior, gamma, it)

    L_next = global_weights_c.shape[0]

    if layer_index <= 1:
        if n_layers == 2:
            softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
            global_weights_out = [softmax_bias]
            global_inv_sigmas_out = [softmax_inv_sigma]
        
        global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[int(layer_index/2)]], global_weights_c[:, init_channel_kernel_dims[int(layer_index/2)]]]
        global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[int(layer_index/2)]], global_sigmas_c[:, init_channel_kernel_dims[int(layer_index/2)]]]

        logger.info("Branch A, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    elif layer_index == (n_layers - 1) and n_layers > 2:
        softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)

        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        #first_fc_identifier = ('fc' in layer_type and 'conv' in prev_layer_type)
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))

        # if first_fc_identifier:
        #     global_weights_out = [global_weights_c[:, 0:-softmax_bias.shape[0]-1].T, 
        #                             global_weights_c[:, -softmax_bias.shape[0]-1], 
        #                             global_weights_c[:, -softmax_bias.shape[0]:], 
        #                             softmax_bias]

        #     global_inv_sigmas_out = [global_sigmas_c[:, 0:-softmax_bias.shape[0]-1].T, 
        #                                 global_sigmas_c[:, -softmax_bias.shape[0]-1], 
        #                                 global_sigmas_c[:, -softmax_bias.shape[0]:], 
        #                                 softmax_inv_sigma]
        # else:
        #     global_weights_out = [global_weights_c[:, 0:matching_shapes[layer_index - 1 - 1]].T, 
        #                             global_weights_c[:, matching_shapes[layer_index - 1 - 1]], 
        #                             global_weights_c[:, matching_shapes[layer_index - 1 - 1]+1:], 
        #                             softmax_bias]

        #     global_inv_sigmas_out = [global_sigmas_c[:, 0:matching_shapes[layer_index - 1 - 1]].T, 
        #                                 global_sigmas_c[:, matching_shapes[layer_index - 1 - 1]], 
        #                                 global_sigmas_c[:, matching_shapes[layer_index - 1 - 1]+1:], 
        #                                 softmax_inv_sigma]

        # remove fitting the last layer
        if first_fc_identifier:
            global_weights_out = [global_weights_c[:, 0:-softmax_bias.shape[0]-1].T, 
                                    global_weights_c[:, -softmax_bias.shape[0]-1]]

            global_inv_sigmas_out = [global_sigmas_c[:, 0:-softmax_bias.shape[0]-1].T, 
                                        global_sigmas_c[:, -softmax_bias.shape[0]-1]]
        else:
            global_weights_out = [global_weights_c[:, 0:matching_shapes[layer_index - 1 - 1]].T, 
                                    global_weights_c[:, matching_shapes[layer_index - 1 - 1]]]

            global_inv_sigmas_out = [global_sigmas_c[:, 0:matching_shapes[layer_index - 1 - 1]].T, 
                                        global_sigmas_c[:, matching_shapes[layer_index - 1 - 1]]]
        logger.info("Branch B, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape

        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1], global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1], global_sigmas_c[:, gwc_shape[1]-1]]
        elif "fc" in layer_type or 'classifier' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1].T, global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1].T, global_sigmas_c[:, gwc_shape[1]-1]]
        logger.info("Branch layer index, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    logger.info("global inv sigma out shape: {}".format([giso.shape for giso in global_inv_sigmas_out]))
    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
    return map_out, assignment_c, L_next



def layer_wise_group_descent_comm(batch_weights, layer_index, batch_frequencies, sigma_layers, 
                                sigma0_layers, gamma_layers, it, 
                                model_meta_data, 
                                model_layer_type,
                                n_layers,
                                matching_shapes,
                                args):
    """
    We implement a layer-wise matching here:
    """
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    last_layer_const = []
    #total_freq = sum(batch_frequencies)
    #for f in batch_frequencies:
    #    last_layer_const.append(f / total_freq)

    # J: number of workers
    J = len(batch_weights)
    # init_num_kernel: the number of conv filters in the first conv layer 
    init_num_kernel = batch_weights[0][0].shape[0]

    # for saving (#channel * k * k)
    init_channel_kernel_dims = []
    for bw in batch_weights[0]:
        if len(bw.shape) > 1:
            init_channel_kernel_dims.append(bw.shape[1])
    logger.info("init_channel_kernel_dims: {}".format(init_channel_kernel_dims))
    
    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None

    sigma = sigma_layers[layer_index - 1]
    sigma_bias = sigma_bias_layers[layer_index - 1]
    gamma = gamma_layers[layer_index - 1]
    sigma0 = sigma0_layers[layer_index - 1]
    sigma0_bias = sigma0_bias_layers[layer_index - 1]

    def ___trans_next_conv_layer_forward(layer_weight, next_layer_shape):
        reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
        return reshaped

    def ___trans_next_conv_layer_backward(layer_weight, next_layer_shape):
        reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
        reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
        return reshaped

    if layer_index <= 1:
        # _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]

        # weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2], 
        #                            batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
        #                            ___trans_next_conv_layer_forward(batch_weights[j][(layer_index+1) * 2 - 2], _next_layer_shape))) for j in range(J)]  

        # _residual_dim = weights_bias[0].shape[1] - init_channel_kernel_dims[layer_index - 1] - 1

        # sigma_inv_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias] + _residual_dim * [1 / sigma0])
        # mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias] + _residual_dim * [mu0])
        # sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias] +  _residual_dim * [1 / sigma]) for j in range(J)]

        _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]

        weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2], 
                                   batch_weights[j][layer_index * 2 - 1].reshape(-1, 1))) for j in range(J)]  

        _residual_dim = weights_bias[0].shape[1] - init_channel_kernel_dims[layer_index - 1] - 1

        sigma_inv_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias])
        mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias])
        sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias]) for j in range(J)]

    elif layer_index == (n_layers - 1) and n_layers > 2:
        # our assumption is that this branch will consistently handle the last fc layers
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))


        weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, 
                                         batch_weights[j][2 * layer_index - 1].reshape(-1, 1),
                                         batch_weights[j][(layer_index+1) * 2 - 2])) for j in range(J)]

        sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma0] + [1 / sigma0_bias] +  batch_weights[0][(layer_index+1) * 2 - 2].shape[1] * [1 / sigma0])
        mean_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [mu0] + [mu0_bias] + batch_weights[0][(layer_index+1) * 2 - 2].shape[1] * [mu0])
        sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma] + [1 / sigma_bias] + batch_weights[0][(layer_index+1) * 2 - 2].shape[1] * [1 / sigma]) for j in range(J)]

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))

        if 'conv' in layer_type or 'features' in layer_type:
            # hard coded a bit for now:
            if layer_index != 6:
                weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2],
                                           batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                           ___trans_next_conv_layer_forward(batch_weights[j][(layer_index+1) * 2 - 2], _next_layer_shape))) for j in range(J)]

                sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma0] + [1 / sigma0_bias] + (_next_layer_shape[0]*_next_layer_shape[2]*_next_layer_shape[3]) * [1 / sigma0])
                mean_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [mu0] + [mu0_bias] + (_next_layer_shape[0]*_next_layer_shape[2]*_next_layer_shape[3]) * [mu0])
                sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma] + [1 / sigma_bias] + (_next_layer_shape[0]*_next_layer_shape[2]*_next_layer_shape[3]) * [1 / sigma]) for j in range(J)]
            else:

                logger.info("$$$$$$$$$$Part A shape: {}, Part C shape: {}".format(batch_weights[0][layer_index * 2 - 2].shape, batch_weights[0][(layer_index+1) * 2 - 2].shape))
                # we need to reconstruct the shape of the representation that is going to fill into FC blocks
                __num_filters = copy.deepcopy(matching_shapes)
                __num_filters.append(batch_weights[0][layer_index * 2 - 2].shape[0])
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=__num_filters)
                dummy_input = torch.rand(1, 3, 32, 32)
                estimated_output = shape_estimator(dummy_input)
                # Est output shape is something lookjs like: torch.Size([1, 256, 4, 4])
                __estimated_shape = (estimated_output.size()[1], estimated_output.size()[2], estimated_output.size()[3])
                __ori_shape = batch_weights[0][(layer_index+1) * 2 - 2].shape

                weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2],
                                            batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                            batch_weights[j][(layer_index+1) * 2 - 2].reshape((__estimated_shape[0], __estimated_shape[1]*__estimated_shape[2]*__ori_shape[1])))) for j in range(J)]
                
                sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma0] + [1 / sigma0_bias] + (__estimated_shape[1]*__estimated_shape[2]*__ori_shape[1]) * [1 / sigma0])
                mean_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [mu0] + [mu0_bias] + (__estimated_shape[1]*__estimated_shape[2]*__ori_shape[1]) * [mu0])
                sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma] + [1 / sigma_bias] + (__estimated_shape[1]*__estimated_shape[2]*__ori_shape[1]) * [1 / sigma]) for j in range(J)]
            
        elif 'fc' in layer_type or 'classifier' in layer_type:        
            weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2].T,
                                       batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                       batch_weights[j][(layer_index+1) * 2 - 2])) for j in range(J)]

            sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma0] + [1 / sigma0_bias] + (batch_weights[0][(layer_index+1) * 2 - 2].shape[1]) * [1 / sigma0])
            mean_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [mu0] + [mu0_bias] + (batch_weights[0][(layer_index+1) * 2 - 2].shape[1]) * [mu0])
            sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma] + [1 / sigma_bias] + (batch_weights[0][(layer_index+1) * 2 - 2].shape[1]) * [1 / sigma]) for j in range(J)]


    logger.info("Layer index: {}, init_num_kernel: {}".format(layer_index, init_num_kernel))
    logger.info("weights bias: {}".format(weights_bias[0].shape))
    logger.info("sigma_inv_prior shape: {}".format(sigma_inv_prior.shape))
    logger.info("mean_prior shape: {}".format(mean_prior.shape))

    assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                  sigma_inv_prior, gamma, it)

    L_next = global_weights_c.shape[0]

    if layer_index <= 1:
        # global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
        #                         global_weights_c[:, init_channel_kernel_dims[2 * layer_index - 2]], 
        #                         global_weights_c[:, init_channel_kernel_dims[2 * layer_index - 2]+1:]]

        # global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
        #                             global_sigmas_c[:, init_channel_kernel_dims[2 * layer_index - 2]], 
        #                             global_sigmas_c[:, init_channel_kernel_dims[2 * layer_index - 2]+1:]]

        global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
                                global_weights_c[:, init_channel_kernel_dims[2 * layer_index - 2]]]

        global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
                                    global_sigmas_c[:, init_channel_kernel_dims[2 * layer_index - 2]]]

        logger.info("Branch A, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))
        logger.info("global inv sigma out shape: {}".format([giso.shape for giso in global_inv_sigmas_out]))
        map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
        # we need to do a minor fix here:
        _next_layer_shape = list(model_meta_data[(layer_index+1) * 2 - 2])
        _next_layer_shape[1] = map_out[0].shape[0]
        # please note that the reshape/transpose stuff will also raise issue here
        #map_out[-1] = ___trans_next_conv_layer_backward(map_out[-1], _next_layer_shape)   

    elif layer_index == (n_layers - 1) and n_layers > 2:
        softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)

        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))

        # remove fitting the last layer
        __ori_shape = batch_weights[0][layer_index * 2 - 2].T.shape
        global_weights_out = [global_weights_c[:, 0:__ori_shape[1]].T, 
                                global_weights_c[:, __ori_shape[1]],
                                global_weights_c[:, __ori_shape[1]+1:]]

        global_inv_sigmas_out = [global_sigmas_c[:, 0:__ori_shape[1]].T, 
                                   global_sigmas_c[:, __ori_shape[1]],
                                   global_sigmas_c[:, __ori_shape[1]+1:],]
        map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
        logger.info("Branch B, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape

        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[layer_index - 1]], 
                                    global_weights_c[:, init_channel_kernel_dims[layer_index - 1]], 
                                    global_weights_c[:, init_channel_kernel_dims[layer_index - 1]+1:]]

            global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[layer_index - 1]], 
                                        global_sigmas_c[:, init_channel_kernel_dims[layer_index - 1]], 
                                        global_sigmas_c[:, init_channel_kernel_dims[layer_index - 1]+1:]]
            if layer_index != 6:
                map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
                # we need to do a minor fix here:
                _next_layer_shape = list(model_meta_data[(layer_index+1) * 2 - 2])
                _next_layer_shape[1] = map_out[0].shape[0]
                # please note that the reshape/transpose stuff will also raise issue here
                map_out[-1] = ___trans_next_conv_layer_backward(map_out[-1], _next_layer_shape)
            else:
                map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
                # we need to do a minor fix here:
                _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]
                _ori_shape = map_out[-1].shape
                # please note that the reshape/transpose stuff will also raise issue here
                map_out[-1] = map_out[-1].reshape((int(_ori_shape[0]*_ori_shape[1]/_next_layer_shape[0]), _next_layer_shape[0]))

        elif "fc" in layer_type or 'classifier' in layer_type:
            #global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1].T, global_weights_c[:, gwc_shape[1]-1]]
            #global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1].T, global_sigmas_c[:, gwc_shape[1]-1]]
            __ori_shape = batch_weights[0][layer_index * 2 - 2].T.shape
            global_weights_out = [global_weights_c[:, 0:__ori_shape[1]].T, 
                                    global_weights_c[:, __ori_shape[1]],
                                    global_weights_c[:, __ori_shape[1]+1:]]

            global_inv_sigmas_out = [global_sigmas_c[:, 0:__ori_shape[1]].T, 
                                       global_sigmas_c[:, __ori_shape[1]],
                                       global_sigmas_c[:, __ori_shape[1]+1:],]

            map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
        logger.info("Branch layer index, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))
    return map_out, assignment_c, L_next



def layer_wise_group_descent_comm2(batch_weights, layer_index, batch_frequencies, sigma_layers, 
                                sigma0_layers, gamma_layers, it, 
                                model_meta_data, 
                                model_layer_type,
                                n_layers,
                                matching_shapes,
                                args):
    """
    We implement a layer-wise matching here:
    """
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    last_layer_const = []

    # J: number of workers
    J = len(batch_weights)
    # init_num_kernel: the number of conv filters in the first conv layer 
    init_num_kernel = batch_weights[0][0].shape[0]

    init_channel_kernel_dims = []
    for bw in batch_weights[0]:
        if len(bw.shape) > 1:
            init_channel_kernel_dims.append(bw.shape[1])
    logger.info("init_channel_kernel_dims: {}".format(init_channel_kernel_dims))
    
    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    #mu0_bias = 0.1
    mu0_bias = 0.0
    
    assignment_c = [None for j in range(J)]
    L_next = None

    sigma = sigma_layers[layer_index - 1]
    sigma_bias = sigma_bias_layers[layer_index - 1]
    gamma = gamma_layers[layer_index - 1]
    sigma0 = sigma0_layers[layer_index - 1]
    sigma0_bias = sigma0_bias_layers[layer_index - 1]

    def ___trans_next_conv_layer_forward(layer_weight, next_layer_shape):
        reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
        #reshaped = layer_weight.reshape((next_layer_shape[1], -1))
        return reshaped

    def ___trans_next_conv_layer_backward(layer_weight, next_layer_shape):
        reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
        reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
        #reshaped = layer_weight.reshape(next_layer_shape[0], -1)
        return reshaped

    if layer_index <= 1:
        _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]

        weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2], 
                                   batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                   ___trans_next_conv_layer_forward(batch_weights[j][(layer_index+1) * 2 - 2], _next_layer_shape))) for j in range(J)]  

        _residual_dim = weights_bias[0].shape[1] - init_channel_kernel_dims[layer_index - 1] - 1

        sigma_inv_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias] + _residual_dim * [1 / sigma0])
        mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias] + _residual_dim * [mu0])
        sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias] +  _residual_dim * [1 / sigma]) for j in range(J)]

        # _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]

        # weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2], 
        #                            batch_weights[j][layer_index * 2 - 1].reshape(-1, 1))) for j in range(J)]  

        # _residual_dim = weights_bias[0].shape[1] - init_channel_kernel_dims[layer_index - 1] - 1

        # sigma_inv_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias])
        # mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias])
        # sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias]) for j in range(J)]

    elif layer_index == (n_layers - 1) and n_layers > 2:
        # our assumption is that this branch will consistently handle the last fc layers
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))


        weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, 
                                         batch_weights[j][2 * layer_index - 1].reshape(-1, 1),
                                         batch_weights[j][(layer_index+1) * 2 - 2])) for j in range(J)]

        sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma0] + [1 / sigma0_bias] +  batch_weights[0][(layer_index+1) * 2 - 2].shape[1] * [1 / sigma0])
        mean_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [mu0] + [mu0_bias] + batch_weights[0][(layer_index+1) * 2 - 2].shape[1] * [mu0])
        sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma] + [1 / sigma_bias] + batch_weights[0][(layer_index+1) * 2 - 2].shape[1] * [1 / sigma]) for j in range(J)]

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))

        if 'conv' in layer_type or 'features' in layer_type:
            # hard coded a bit for now:
            if layer_index != 6:
                weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2],
                                           batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                           ___trans_next_conv_layer_forward(batch_weights[j][(layer_index+1) * 2 - 2], _next_layer_shape))) for j in range(J)]

                sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma0] + [1 / sigma0_bias] + (_next_layer_shape[0]*_next_layer_shape[2]*_next_layer_shape[3]) * [1 / sigma0])
                mean_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [mu0] + [mu0_bias] + (_next_layer_shape[0]*_next_layer_shape[2]*_next_layer_shape[3]) * [mu0])
                sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma] + [1 / sigma_bias] + (_next_layer_shape[0]*_next_layer_shape[2]*_next_layer_shape[3]) * [1 / sigma]) for j in range(J)]
            else:

                logger.info("$$$$$$$$$$Part A shape: {}, Part C shape: {}".format(batch_weights[0][layer_index * 2 - 2].shape, batch_weights[0][(layer_index+1) * 2 - 2].shape))
                # we need to reconstruct the shape of the representation that is going to fill into FC blocks
                __num_filters = copy.deepcopy(matching_shapes)
                __num_filters.append(batch_weights[0][layer_index * 2 - 2].shape[0])
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=__num_filters)
                dummy_input = torch.rand(1, 3, 32, 32)
                estimated_output = shape_estimator(dummy_input)
                # Est output shape is something lookjs like: torch.Size([1, 256, 4, 4])
                __estimated_shape = (estimated_output.size()[1], estimated_output.size()[2], estimated_output.size()[3])
                __ori_shape = batch_weights[0][(layer_index+1) * 2 - 2].shape

                weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2],
                                            batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                            batch_weights[j][(layer_index+1) * 2 - 2].reshape((__estimated_shape[0], __estimated_shape[1]*__estimated_shape[2]*__ori_shape[1])))) for j in range(J)]
                
                sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma0] + [1 / sigma0_bias] + (__estimated_shape[1]*__estimated_shape[2]*__ori_shape[1]) * [1 / sigma0])
                mean_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [mu0] + [mu0_bias] + (__estimated_shape[1]*__estimated_shape[2]*__ori_shape[1]) * [mu0])
                sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma] + [1 / sigma_bias] + (__estimated_shape[1]*__estimated_shape[2]*__ori_shape[1]) * [1 / sigma]) for j in range(J)]
            
        elif 'fc' in layer_type or 'classifier' in layer_type:        
            weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2].T,
                                       batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                       batch_weights[j][(layer_index+1) * 2 - 2])) for j in range(J)]

            sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma0] + [1 / sigma0_bias] + (batch_weights[0][(layer_index+1) * 2 - 2].shape[1]) * [1 / sigma0])
            mean_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [mu0] + [mu0_bias] + (batch_weights[0][(layer_index+1) * 2 - 2].shape[1]) * [mu0])
            sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma] + [1 / sigma_bias] + (batch_weights[0][(layer_index+1) * 2 - 2].shape[1]) * [1 / sigma]) for j in range(J)]


    logger.info("Layer index: {}, init_num_kernel: {}".format(layer_index, init_num_kernel))
    logger.info("weights bias: {}".format(weights_bias[0].shape))
    logger.info("sigma_inv_prior shape: {}".format(sigma_inv_prior.shape))
    logger.info("mean_prior shape: {}".format(mean_prior.shape))

    assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                  sigma_inv_prior, gamma, it)

    L_next = global_weights_c.shape[0]

    if layer_index <= 1:
        global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
                                global_weights_c[:, init_channel_kernel_dims[2 * layer_index - 2]], 
                                global_weights_c[:, init_channel_kernel_dims[2 * layer_index - 2]+1:]]

        global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
                                    global_sigmas_c[:, init_channel_kernel_dims[2 * layer_index - 2]], 
                                    global_sigmas_c[:, init_channel_kernel_dims[2 * layer_index - 2]+1:]]

        # global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
        #                         global_weights_c[:, init_channel_kernel_dims[2 * layer_index - 2]]]

        # global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
        #                             global_sigmas_c[:, init_channel_kernel_dims[2 * layer_index - 2]]]

        logger.info("Branch A, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))
        logger.info("global inv sigma out shape: {}".format([giso.shape for giso in global_inv_sigmas_out]))
        map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
        # we need to do a minor fix here:
        _next_layer_shape = list(model_meta_data[(layer_index+1) * 2 - 2])
        _next_layer_shape[1] = map_out[0].shape[0]
        # please note that the reshape/transpose stuff will also raise issue here
        #map_out[-1] = ___trans_next_conv_layer_backward(map_out[-1], _next_layer_shape)   

    elif layer_index == (n_layers - 1) and n_layers > 2:
        softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)

        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))

        # remove fitting the last layer
        __ori_shape = batch_weights[0][layer_index * 2 - 2].T.shape
        global_weights_out = [global_weights_c[:, 0:__ori_shape[1]].T, 
                                global_weights_c[:, __ori_shape[1]],
                                global_weights_c[:, __ori_shape[1]+1:]]

        global_inv_sigmas_out = [global_sigmas_c[:, 0:__ori_shape[1]].T, 
                                   global_sigmas_c[:, __ori_shape[1]],
                                   global_sigmas_c[:, __ori_shape[1]+1:],]
        map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
        logger.info("Branch B, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape

        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[layer_index - 1]], 
                                    global_weights_c[:, init_channel_kernel_dims[layer_index - 1]], 
                                    global_weights_c[:, init_channel_kernel_dims[layer_index - 1]+1:]]

            global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[layer_index - 1]], 
                                        global_sigmas_c[:, init_channel_kernel_dims[layer_index - 1]], 
                                        global_sigmas_c[:, init_channel_kernel_dims[layer_index - 1]+1:]]
            if layer_index != 6:
                map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
                # we need to do a minor fix here:
                _next_layer_shape = list(model_meta_data[(layer_index+1) * 2 - 2])
                _next_layer_shape[1] = map_out[0].shape[0]
                # please note that the reshape/transpose stuff will also raise issue here
                map_out[-1] = ___trans_next_conv_layer_backward(map_out[-1], _next_layer_shape)
            else:
                map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
                # we need to do a minor fix here:
                _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]
                _ori_shape = map_out[-1].shape
                # please note that the reshape/transpose stuff will also raise issue here
                map_out[-1] = map_out[-1].reshape((int(_ori_shape[0]*_ori_shape[1]/_next_layer_shape[0]), _next_layer_shape[0]))

        elif "fc" in layer_type or 'classifier' in layer_type:
            #global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1].T, global_weights_c[:, gwc_shape[1]-1]]
            #global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1].T, global_sigmas_c[:, gwc_shape[1]-1]]
            __ori_shape = batch_weights[0][layer_index * 2 - 2].T.shape
            global_weights_out = [global_weights_c[:, 0:__ori_shape[1]].T, 
                                    global_weights_c[:, __ori_shape[1]],
                                    global_weights_c[:, __ori_shape[1]+1:]]

            global_inv_sigmas_out = [global_sigmas_c[:, 0:__ori_shape[1]].T, 
                                       global_sigmas_c[:, __ori_shape[1]],
                                       global_sigmas_c[:, __ori_shape[1]+1:],]

            map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
        logger.info("Branch layer index, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))
    return map_out, assignment_c, L_next


def layer_wise_group_descent_comm3(batch_weights, layer_index, batch_frequencies, sigma_layers, 
                                sigma0_layers, gamma_layers, it, 
                                model_meta_data, 
                                model_layer_type,
                                n_layers,
                                matching_shapes,
                                args):
    """
    We implement a layer-wise matching here:
    """
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    last_layer_const = []

    # J: number of workers
    J = len(batch_weights)
    # init_num_kernel: the number of conv filters in the first conv layer 
    init_num_kernel = batch_weights[0][0].shape[0]

    init_channel_kernel_dims = []
    for bw in batch_weights[0]:
        if len(bw.shape) > 1:
            init_channel_kernel_dims.append(bw.shape[1])
    logger.info("init_channel_kernel_dims: {}".format(init_channel_kernel_dims))
    
    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    #mu0_bias = 0.1
    mu0_bias = 0.0
    
    assignment_c = [None for j in range(J)]
    L_next = None

    sigma = sigma_layers[layer_index - 1]
    sigma_bias = sigma_bias_layers[layer_index - 1]
    gamma = gamma_layers[layer_index - 1]
    sigma0 = sigma0_layers[layer_index - 1]
    sigma0_bias = sigma0_bias_layers[layer_index - 1]

    def ___trans_next_conv_layer_forward(layer_weight, next_layer_shape):
        reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
        return reshaped

    def ___trans_next_conv_layer_backward(layer_weight, next_layer_shape):
        reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
        reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
        return reshaped

    if layer_index <= 1:
        # _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]

        # weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2], 
        #                            batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
        #                            ___trans_next_conv_layer_forward(batch_weights[j][(layer_index+1) * 2 - 2], _next_layer_shape))) for j in range(J)]  

        # _residual_dim = weights_bias[0].shape[1] - init_channel_kernel_dims[layer_index - 1] - 1

        # sigma_inv_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias] + _residual_dim * [1 / sigma0])
        # mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias] + _residual_dim * [mu0])
        # sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias] +  _residual_dim * [1 / sigma]) for j in range(J)]

        # _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]

        weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2], 
                                   batch_weights[j][layer_index * 2 - 1].reshape(-1, 1))) for j in range(J)]  

        sigma_inv_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias])
        mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias])
        sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias]) for j in range(J)]

    elif layer_index == (n_layers - 1) and n_layers > 2:
        # our assumption is that this branch will consistently handle the last fc layers
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))


        weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, 
                                         batch_weights[j][2 * layer_index - 1].reshape(-1, 1),
                                         batch_weights[j][(layer_index+1) * 2 - 2])) for j in range(J)]

        sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma0] + [1 / sigma0_bias] +  batch_weights[0][(layer_index+1) * 2 - 2].shape[1] * [1 / sigma0])
        mean_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [mu0] + [mu0_bias] + batch_weights[0][(layer_index+1) * 2 - 2].shape[1] * [mu0])
        sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma] + [1 / sigma_bias] + batch_weights[0][(layer_index+1) * 2 - 2].shape[1] * [1 / sigma]) for j in range(J)]

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))

        if 'conv' in layer_type or 'features' in layer_type:
            # hard coded a bit for now:
            if layer_index != 6:
                weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2],
                                           batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                           ___trans_next_conv_layer_forward(batch_weights[j][(layer_index+1) * 2 - 2], _next_layer_shape))) for j in range(J)]

                sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma0] + [1 / sigma0_bias] + (_next_layer_shape[0]*_next_layer_shape[2]*_next_layer_shape[3]) * [1 / sigma0])
                mean_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [mu0] + [mu0_bias] + (_next_layer_shape[0]*_next_layer_shape[2]*_next_layer_shape[3]) * [mu0])
                sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma] + [1 / sigma_bias] + (_next_layer_shape[0]*_next_layer_shape[2]*_next_layer_shape[3]) * [1 / sigma]) for j in range(J)]
            else:

                logger.info("$$$$$$$$$$Part A shape: {}, Part C shape: {}".format(batch_weights[0][layer_index * 2 - 2].shape, batch_weights[0][(layer_index+1) * 2 - 2].shape))
                # we need to reconstruct the shape of the representation that is going to fill into FC blocks
                __num_filters = copy.deepcopy(matching_shapes)
                __num_filters.append(batch_weights[0][layer_index * 2 - 2].shape[0])
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=__num_filters)
                dummy_input = torch.rand(1, 3, 32, 32)
                estimated_output = shape_estimator(dummy_input)
                # Est output shape is something lookjs like: torch.Size([1, 256, 4, 4])
                __estimated_shape = (estimated_output.size()[1], estimated_output.size()[2], estimated_output.size()[3])
                __ori_shape = batch_weights[0][(layer_index+1) * 2 - 2].shape

                weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2],
                                            batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                            batch_weights[j][(layer_index+1) * 2 - 2].reshape((__estimated_shape[0], __estimated_shape[1]*__estimated_shape[2]*__ori_shape[1])))) for j in range(J)]
                
                sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma0] + [1 / sigma0_bias] + (__estimated_shape[1]*__estimated_shape[2]*__ori_shape[1]) * [1 / sigma0])
                mean_prior = np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [mu0] + [mu0_bias] + (__estimated_shape[1]*__estimated_shape[2]*__ori_shape[1]) * [mu0])
                sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].shape[1]) * [1 / sigma] + [1 / sigma_bias] + (__estimated_shape[1]*__estimated_shape[2]*__ori_shape[1]) * [1 / sigma]) for j in range(J)]
            
        elif 'fc' in layer_type or 'classifier' in layer_type:        
            weights_bias = [np.hstack((batch_weights[j][layer_index * 2 - 2].T,
                                       batch_weights[j][layer_index * 2 - 1].reshape(-1, 1),
                                       batch_weights[j][(layer_index+1) * 2 - 2])) for j in range(J)]

            sigma_inv_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma0] + [1 / sigma0_bias] + (batch_weights[0][(layer_index+1) * 2 - 2].shape[1]) * [1 / sigma0])
            mean_prior = np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [mu0] + [mu0_bias] + (batch_weights[0][(layer_index+1) * 2 - 2].shape[1]) * [mu0])
            sigma_inv_layer = [np.array((batch_weights[0][2 * layer_index - 2].T.shape[1]) * [1 / sigma] + [1 / sigma_bias] + (batch_weights[0][(layer_index+1) * 2 - 2].shape[1]) * [1 / sigma]) for j in range(J)]


    logger.info("Layer index: {}, init_num_kernel: {}".format(layer_index, init_num_kernel))
    logger.info("weights bias: {}".format(weights_bias[0].shape))
    logger.info("sigma_inv_prior shape: {}".format(sigma_inv_prior.shape))
    logger.info("mean_prior shape: {}".format(mean_prior.shape))

    assignment_c, global_weights_c, popularity_counts, hyper_params = match_local_atoms(local_atoms=weights_bias, 
                                                                        sigma=sigma, 
                                                                        sigma0=sigma0,
                                                                        gamma=gamma, 
                                                                        it=it)
    L_next = global_weights_c.shape[0]

    if layer_index <= 1:
        # global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
        #                         global_weights_c[:, init_channel_kernel_dims[2 * layer_index - 2]], 
        #                         global_weights_c[:, init_channel_kernel_dims[2 * layer_index - 2]+1:]]
        global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[2 * layer_index - 2]], 
                                global_weights_c[:, init_channel_kernel_dims[2 * layer_index - 2]]]

        logger.info("Branch A, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))
        #map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
        map_out = global_weights_out

    elif layer_index == (n_layers - 1) and n_layers > 2:
        softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)

        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))

        # remove fitting the last layer
        __ori_shape = batch_weights[0][layer_index * 2 - 2].T.shape
        global_weights_out = [global_weights_c[:, 0:__ori_shape[1]].T, 
                                global_weights_c[:, __ori_shape[1]],
                                global_weights_c[:, __ori_shape[1]+1:]]

        global_inv_sigmas_out = [global_sigmas_c[:, 0:__ori_shape[1]].T, 
                                   global_sigmas_c[:, __ori_shape[1]],
                                   global_sigmas_c[:, __ori_shape[1]+1:],]
        map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
        logger.info("Branch B, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape

        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[layer_index - 1]], 
                                    global_weights_c[:, init_channel_kernel_dims[layer_index - 1]], 
                                    global_weights_c[:, init_channel_kernel_dims[layer_index - 1]+1:]]

            global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[layer_index - 1]], 
                                        global_sigmas_c[:, init_channel_kernel_dims[layer_index - 1]], 
                                        global_sigmas_c[:, init_channel_kernel_dims[layer_index - 1]+1:]]
            if layer_index != 6:
                map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
                # we need to do a minor fix here:
                _next_layer_shape = list(model_meta_data[(layer_index+1) * 2 - 2])
                _next_layer_shape[1] = map_out[0].shape[0]
                # please note that the reshape/transpose stuff will also raise issue here
                map_out[-1] = ___trans_next_conv_layer_backward(map_out[-1], _next_layer_shape)
            else:
                map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
                # we need to do a minor fix here:
                _next_layer_shape = model_meta_data[(layer_index+1) * 2 - 2]
                _ori_shape = map_out[-1].shape
                # please note that the reshape/transpose stuff will also raise issue here
                map_out[-1] = map_out[-1].reshape((int(_ori_shape[0]*_ori_shape[1]/_next_layer_shape[0]), _next_layer_shape[0]))

        elif "fc" in layer_type or 'classifier' in layer_type:
            __ori_shape = batch_weights[0][layer_index * 2 - 2].T.shape
            global_weights_out = [global_weights_c[:, 0:__ori_shape[1]].T, 
                                    global_weights_c[:, __ori_shape[1]],
                                    global_weights_c[:, __ori_shape[1]+1:]]

            global_inv_sigmas_out = [global_sigmas_c[:, 0:__ori_shape[1]].T, 
                                       global_sigmas_c[:, __ori_shape[1]],
                                       global_sigmas_c[:, __ori_shape[1]+1:],]

            map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
        logger.info("Branch layer index, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))
    return map_out, assignment_c, L_next