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

true_sys = TD.LCS_learner_regression(n_state, n_control, n_lam, A, B, C, D, E, G, H, lcp_offset, stiffness=0)
true_lcp_theta = vertcat(vec(D), vec(E), vec(G), vec(H), vec(lcp_offset)).full().flatten()
true_dyn_theta = vertcat(vec(A), vec(B), vec(C)).full().flatten()
true_theta = vertcat(true_dyn_theta, true_lcp_theta).full().flatten()

# control cost function
control_horizon = 20
Q = 2 * np.eye(n_state)
QN = Q
R = np.eye(n_control)

# ============================= establish the lcs learner (automatically initialized)
reduced_n_lam = 3
lcs_learner = TD.LCS_learner_regression(n_state, n_control, n_lam=reduced_n_lam, stiffness=0)
lcs_learner.val_lcp_theta = 1. * np.random.randn(lcs_learner.n_lcp_theta)

# ============================= initialize the optimizer
optimizier = opt.Adam()
optimizier.learning_rate = 1e-2

# ============================= initialize an evaluator for the learned system
mpc_horizon = 5
evaluator = TD.LCS_evaluation_MPC(lcs_learner)
evaluator.setCostFunction(Q, R, QN)
evaluator.initializeMPC(mpc_horizon)

# # ============================= initialize a random control sequence for random initial condition
#
# init_state_batch = np.random.randn(traj_count, n_state)
# control_traj_batch = TD.randomControlTraj(traj_count, control_horizon, n_control)

# # =============================== compute the true optimal cost (for reference)
# true_sys_oc_solver = TD.LCS_MPC(A, B, C, D, E, F, lcp_offset)
# true_sys_oc_solver.oc_setup(control_horizon)
# true_sys_opt_control_traj_batch = []
# for i in range(traj_count):
#     true_sys_sol = true_sys_oc_solver.mpc(init_state_batch[i], Q, R, QN)
#     true_sys_opt_control_traj_batch += [true_sys_sol['control_traj_opt']]
# true_sys_opt_state_traj_batch, true_sys_opt_lam_traj_batch = true_sys.sim_dyn(init_state_batch,
#                                                                               true_sys_opt_control_traj_batch)
# true_sys_opt_cost_batch = evaluator.computeCost(true_sys_opt_control_traj_batch, true_sys_opt_state_traj_batch)

# ================= starting the data-driven model learning process
control_cost_trace = []
for control_iter in range(1):

    # # ============================= random the initial condition
    traj_count = 30
    init_state_batch = np.random.randn(traj_count, n_state)

    # # ============ use the mpc controller (based on the lcs model) to simulate the true system
    state_batch_traj = [list(init_state_batch)]
    control_batch_traj = []
    for t in range(control_horizon):
        # compute the control input using the current lcs learner
        st=time.time()
        control_batch_traj += [evaluator.mpc(lcs_learner, state_batch_traj[-1])]
        # simulate to the next step
        next_state_batch, lam_batch = true_sys.dyn_step(state_batch_traj[-1], control_batch_traj[-1])
        state_batch_traj += [next_state_batch]
    # reorganize the data format
    control_traj_batch = TD.dataReorgnize(control_batch_traj)
    state_traj_batch = TD.dataReorgnize(state_batch_traj)

    # print
    print(len(control_traj_batch))
    print(len(state_traj_batch))
    print(control_traj_batch[0].shape)
    print(state_traj_batch[0].shape)

    print('hahaha')
