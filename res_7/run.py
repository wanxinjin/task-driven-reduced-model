import matplotlib.pyplot as plt

import lcs.optim as opt
import lcs.task_driven as TD
import numpy as np
from casadi import *
import time


def print(*args):
    __builtins__.print(*("%.5f" % a if isinstance(a, float) else a
                         for a in args))


# ==============================   load the true LCS system
lcs_mats = np.load('random_lcs.npy', allow_pickle=True).item()
n_state = lcs_mats['n_state']
n_control = lcs_mats['n_control']
n_lam = lcs_mats['n_lam']
A = lcs_mats['A']
B = lcs_mats['B']
C = lcs_mats['C']
D = lcs_mats['D']
E = lcs_mats['E']
G = lcs_mats['G']
H = lcs_mats['H']
F = lcs_mats['F']
lcp_offset = lcs_mats['lcp_offset']

# control cost function
control_horizon = 10
Q = 2 * np.eye(n_state)
QN = Q
R = np.eye(n_control)

# print
print('the true system n_dim:', n_lam)

# define the true lcs system
true_sys = TD.LCS_learner_regression(n_state, n_control, n_lam, A, B, C, D, E, G, H, lcp_offset, stiffness=0)

# # ============================= define the ture mpc controller for the true system
mpc_horizon = 5
true_mpc = TD.MPC_Controller(true_sys)
true_mpc.setCostFunction(Q, R, QN)
true_mpc.initializeMPC(mpc_horizon)

# =========================== use the true mpc to simulate on the true lcs system
traj_count = 30
init_state_batch = np.random.randn(traj_count, n_state)
true_state_traj_batch = []
true_control_traj_batch = []
for i in range(traj_count):
    true_state_traj = [init_state_batch[i]]
    true_control_traj = []
    for t in range(control_horizon):
        # compute the current control input
        true_control_traj += [true_mpc.mpc_step(true_sys, true_state_traj[-1])]
        # compute the true state
        true_next_state, true_curr_lam = true_sys.dynamics_step(true_state_traj[-1], true_control_traj[-1])
        true_state_traj += [true_next_state]
    # combine them together
    true_state_traj_batch += [np.array(true_state_traj)]
    true_control_traj_batch += [np.array(true_control_traj)]
# compute the current control cost
true_opt_cost_batch = true_mpc.computeCost(true_control_traj_batch, true_state_traj_batch)
print('True_sys optimal cost (for reference):', np.mean(true_opt_cost_batch))

# ============================= establish the lcs learner (automatically initialized)
reduced_n_lam = 2
lcs_learner = TD.LCS_learner_regression(n_state, n_control, n_lam=reduced_n_lam, stiffness=1)
lcs_learner.val_lcp_theta = 0.1 * np.random.randn(lcs_learner.n_lcp_theta)

# ============================= initialize the optimizer
optimizier = opt.Adam()
optimizier.learning_rate = 5e-3

# ============================= initialize an evaluator for the learned system
learner_mpc = TD.MPC_Controller(lcs_learner)
learner_mpc.setCostFunction(Q, R, QN)
learner_mpc.initializeMPC(mpc_horizon)

# ================= starting the data-driven model learning process
control_cost_trace = []
prev_control_traj_batch = []
prev_state_traj_batch = []
for control_iter in range(20):

    # # ============================= random the initial condition
    traj_count = 50
    np.random.seed(10)
    init_state_batch = np.random.randn(traj_count, n_state)

    # # ============ use the mpc controller (based on the lcs model) to simulate the true system
    state_traj_batch = []
    control_traj_batch = []
    for i in range(traj_count):
        state_traj = [init_state_batch[i]]
        control_traj = []
        for t in range(control_horizon):
            # compute the current control input
            control_traj += [learner_mpc.mpc_step(lcs_learner, state_traj[-1])]
            # compute the true state
            next_state, curr_lam = true_sys.dynamics_step(state_traj[-1], control_traj[-1])
            state_traj += [next_state]
        # combine them together
        state_traj_batch += [np.array(state_traj)]
        control_traj_batch += [np.array(control_traj)]

    # compute the current control cost
    cost_batch = learner_mpc.computeCost(control_traj_batch, state_traj_batch)

    # ============================= print
    print(
        '\n======================================================================'
        '\n|****** Control Iter:', control_iter,
        '| current control cost:', np.mean(cost_batch),
        '| true control cost:', np.mean(true_opt_cost_batch),
        '\n'
    )
    control_cost_trace += [np.mean(cost_batch)]

    # ============================= learn the true lcs system from the true data
    print('lcs learning ')
    TD.LCSRegressionBuffer(lcs_learner, optimizier,
                           control_traj_batch, state_traj_batch,
                           prev_control_traj_batch, prev_state_traj_batch,
                           buffer_ratio=0.5,
                           max_iter=2000,
                           minibatch_size=200,
                           print_level=1)

    # ============================= update the history data
    prev_control_traj_batch = control_traj_batch
    prev_state_traj_batch = state_traj_batch

# save the learned model
learned_lcs_mats = lcs_learner.computeLCSMats(compact=False)
learned_lcs_theta = lcs_learner.computeLCSMats()
np.save('results',
        {
            'control_cost_trace': control_cost_trace,
            'learned_lcs_mats': learned_lcs_mats,
            'learned_lcs_theta': learned_lcs_theta,
            'Q': Q,
            'QN': QN,
            'R': R,
            'reduced_n_lam': reduced_n_lam,
            'control_horizon': control_horizon,
            'mpc_horizon': mpc_horizon,
        })
