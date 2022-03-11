import numpy as np
from casadi import *
import numpy.linalg as la
import matplotlib.pyplot as plt


def statiModes(lam_batch, tol=1e-5):
    # dimension of the lambda
    n_lam = lam_batch.shape[1]
    # total number of modes
    total_n_mode = float(2 ** n_lam)

    # do the statistics for the modes
    lam_batch_mode = np.where(lam_batch < tol, 0, 1)
    unique_mode_list, mode_count_list = np.unique(lam_batch_mode, axis=0, return_counts=True)
    mode_frequency_list = mode_count_list / lam_batch.shape[0]

    return unique_mode_list, mode_frequency_list


# do the plot of differnet mode
def plotModes(lam_batch, tol=1e-5):
    # do the statistics for the modes
    lam_batch_mode = np.where(lam_batch < tol, 0, 1)
    unique_mode_list, mode_indices = np.unique(lam_batch_mode, axis=0, return_inverse=True)

    return unique_mode_list, mode_indices


# generate the random control sequence
def randomControlTraj(traj_count, horizon, n_control):
    res = []
    for i in range(traj_count):
        single_traj = np.random.randn(horizon, n_control)
        res += [single_traj]

    return res


def find_closest(candidate_rows, query):
    n = len(candidate_rows)
    min_distance = inf
    closest_index = 0
    for i in range(n):
        candidate = candidate_rows[i]
        current_distance = norm_2(candidate - query)
        if current_distance < min_distance:
            min_distance = current_distance
            closest_index = i

    return closest_index


def dataReorgnize(batch_traj):
    batch_size = len(batch_traj[0])
    traj_horizon = len(batch_traj)

    traj_batch = []
    for batch_i in range(batch_size):
        traj = []
        for t in range(traj_horizon):
            traj += [batch_traj[t][batch_i]]
        traj = np.array(traj)
        traj_batch += [traj]
    return traj_batch


def simulate_mpc_on_lcs(mpc_controller, mpc_lcs, lcs, init_state_batch, control_horizon, mpc_lcs_theta=None):
    traj_count = init_state_batch.shape[0]
    state_traj_batch = []
    control_traj_batch = []
    for i in range(traj_count):
        state_traj = [init_state_batch[i]]
        control_traj = []
        for t in range(control_horizon):
            # compute the current control input
            if mpc_lcs_theta is None:
                control_traj += [mpc_controller.mpc_step(mpc_lcs, state_traj[-1])]
            else:
                control_traj += [mpc_controller.mpc_step(mpc_lcs, state_traj[-1], lcs_theta=mpc_lcs_theta)]
            # compute the true state
            next_state, curr_lam = lcs.dynamics_step(state_traj[-1], control_traj[-1])
            state_traj += [next_state]
        # combine them together
        state_traj_batch += [np.array(state_traj)]
        control_traj_batch += [np.array(control_traj)]
    # compute the current control cost
    cost_batch = mpc_controller.compute_cost(control_traj_batch, state_traj_batch)

    return state_traj_batch, control_traj_batch, cost_batch


def histogram(state_traj_batch, control_traj_batch):
    n_state = state_traj_batch[0].shape[1]
    n_control = control_traj_batch[0].shape[1]

    x_batch = np.vstack(state_traj_batch)
    u_batch = np.vstack(control_traj_batch)

    x_hist = []
    for i in range(n_state):
        x_i = x_batch[:, i]
        hist_i, _, = np.histogram(x_i, range=[-10, 10], bins=100, density=True)
        x_hist += [hist_i]

    u_hist = []
    for i in range(n_control):
        u_i = u_batch[:, i]
        hist_i, _, = np.histogram(u_i, range=[-5, 5], bins=100, density=True)
        u_hist += [hist_i]

    return x_batch, x_hist, u_batch, u_hist


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / (q + 0.00001)), 0))


def kl_hist(x_hist, prev_hist):
    if len(prev_hist) is not 0:
        n_state = len(x_hist)
        kl_error = []
        for i in range(n_state):
            xi_hist = x_hist[i]
            xi_hist_prev = prev_hist[i]
            kl_error += [kl_divergence(xi_hist, xi_hist_prev)]

        return np.mean(kl_error)
    else:
        return 0
