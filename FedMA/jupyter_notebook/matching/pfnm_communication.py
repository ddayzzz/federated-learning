import numpy as np
from scipy.optimize import linear_sum_assignment


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


def init_from_assignments(batch_weights_norm, sigma_inv_layer, prior_mean_norm, sigma_inv_prior, assignment):
    L = int(max([max(a_j) for a_j in assignment])) + 1
    popularity_counts = [0] * L

    global_weights = np.outer(np.ones(L), prior_mean_norm)
    global_sigmas = np.outer(np.ones(L), sigma_inv_prior)

    for j, a_j in enumerate(assignment):
        for l, i in enumerate(a_j):
            popularity_counts[i] += 1
            global_weights[i] += batch_weights_norm[j][l]
            global_sigmas[i] += sigma_inv_layer[j]

    return popularity_counts, global_weights, global_sigmas


def match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, it, assignment=None):
    J = len(weights_bias)

    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])

    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior

    if assignment is None:
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
                                                                                            global_sigmas,
                                                                                            prior_mean_norm,
                                                                                            sigma_inv_prior,
                                                                                            popularity_counts, gamma, J)
            assignment[j] = assignment_j
    else:
        popularity_counts, global_weights, global_sigmas = init_from_assignments(batch_weights_norm, sigma_inv_layer,
                                                                                 mean_prior, sigma_inv_prior,
                                                                                 assignment)

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
                                print('Warning - weird unmatching')
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

    print('Number of global neurons is %d, gamma %f' % (global_weights.shape[0], gamma))

    return assignment, global_weights, global_sigmas


def layer_group_descent(batch_weights, batch_frequencies, sigma_layers, sigma0_layers, gamma_layers, it,
                        assignments_old=None):

    n_layers = int(len(batch_weights[0]) / 2)
    J = len(batch_weights)
    D = batch_weights[0][0].shape[0]
    K = batch_weights[0][-1].shape[0]

    if assignments_old is None:
        assignments_old = (n_layers - 1) * [None]
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    if batch_frequencies is None:
        last_layer_const = [np.ones(K) for _ in range(J)]
    else:
        last_layer_const = []
        total_freq = sum(batch_frequencies)
        for f in batch_frequencies:
            last_layer_const.append(f / total_freq)

    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None
    assignment_all = []

    ## Group descent for layer
    for c in range(1, n_layers)[::-1]:
        sigma = sigma_layers[c - 1]
        sigma_bias = sigma_bias_layers[c - 1]
        gamma = gamma_layers[c - 1]
        sigma0 = sigma0_layers[c - 1]
        sigma0_bias = sigma0_bias_layers[c - 1]
        if c == (n_layers - 1) and n_layers > 2:
            weights_bias = [np.hstack((batch_weights[j][c * 2 - 1].reshape(-1, 1), batch_weights[j][c * 2])) for j in
                            range(J)]
            sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
            mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            sigma_inv_layer = [np.array([1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
        elif c > 1:
            weights_bias = [np.hstack((batch_weights[j][c * 2 - 1].reshape(-1, 1),
                                       patch_weights(batch_weights[j][c * 2], L_next, assignment_c[j]))) for j in
                            range(J)]
            sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
            mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in
                               range(J)]
        else:
            weights_bias = [np.hstack((batch_weights[j][0].T, batch_weights[j][c * 2 - 1].reshape(-1, 1),
                                       patch_weights(batch_weights[j][c * 2], L_next, assignment_c[j]))) for j in
                            range(J)]
            sigma_inv_prior = np.array(
                D * [1 / sigma0] + [1 / sigma0_bias] + (weights_bias[0].shape[1] - 1 - D) * [1 / sigma0])
            mean_prior = np.array(D * [mu0] + [mu0_bias] + (weights_bias[0].shape[1] - 1 - D) * [mu0])
            if n_layers == 2:
                sigma_inv_layer = [
                    np.array(D * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in
                    range(J)]
            else:
                sigma_inv_layer = [
                    np.array(D * [1 / sigma] + [1 / sigma_bias] + (weights_bias[j].shape[1] - 1 - D) * [1 / sigma]) for
                    j in range(J)]

        assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                      sigma_inv_prior, gamma, it,
                                                                      assignment=assignments_old[c - 1])
        L_next = global_weights_c.shape[0]
        assignment_all = [assignment_c] + assignment_all

        if c == (n_layers - 1) and n_layers > 2:
            softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
            global_weights_out = [global_weights_c[:, 0], global_weights_c[:, 1:], softmax_bias]
            global_inv_sigmas_out = [global_sigmas_c[:, 0], global_sigmas_c[:, 1:], softmax_inv_sigma]
        elif c > 1:
            global_weights_out = [global_weights_c[:, 0], global_weights_c[:, 1:]] + global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, 0], global_sigmas_c[:, 1:]] + global_inv_sigmas_out
        else:
            if n_layers == 2:
                softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
                global_weights_out = [softmax_bias]
                global_inv_sigmas_out = [softmax_inv_sigma]
            global_weights_out = [global_weights_c[:, :D].T, global_weights_c[:, D],
                                  global_weights_c[:, (D + 1):]] + global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, :D].T, global_sigmas_c[:, D],
                                     global_sigmas_c[:, (D + 1):]] + global_inv_sigmas_out

    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]

    return map_out, assignment_all


def build_init(hungarian_weights, assignments, j):
    batch_init = []
    C = len(assignments)
    if len(hungarian_weights) == 4:
        batch_init.append(hungarian_weights[0][:, assignments[0][j]])
        batch_init.append(hungarian_weights[1][assignments[0][j]])
        batch_init.append(hungarian_weights[2][assignments[0][j]])
        batch_init.append(hungarian_weights[3])
        return batch_init
    for c in range(C):
        if c == 0:
            batch_init.append(hungarian_weights[c][:, assignments[c][j]])
            batch_init.append(hungarian_weights[c + 1][assignments[c][j]])
        else:
            batch_init.append(hungarian_weights[2 * c][assignments[c - 1][j]][:, assignments[c][j]])
            batch_init.append(hungarian_weights[2 * c + 1][assignments[c][j]])
            if c == C - 1:
                batch_init.append(hungarian_weights[2 * c + 2][assignments[c][j]])
                batch_init.append(hungarian_weights[-1])
                return batch_init


def gaus_init(n_units, D, K, seed=None, mu=0, sd=0.1, bias=0.1):
    if seed is not None:
        np.random.seed(seed)

    batch_init = []
    C = len(n_units)
    for c in range(C):
        if c == 0:
            batch_init.append(np.random.normal(mu, sd, (D, n_units[c])))
            batch_init.append(np.repeat(bias, n_units[c]))
        else:
            if C != 1:
                batch_init.append(np.random.normal(mu, sd, (n_units[c - 1], n_units[c])))
                batch_init.append(np.repeat(bias, n_units[c]))
        if c == C - 1:
            batch_init.append(np.random.normal(mu, sd, (n_units[c], K)))
            batch_init.append(np.repeat(bias, K))
            return batch_init
