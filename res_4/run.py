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
control_horizon = 6
Q = 2 * np.eye(n_state)
QN = Q
R = np.eye(n_control)

# ============================= establish the lcs learner (automatically initialized)
reduced_n_lam = n_lam
lcs_learner = TD.LCS_learner_regression(n_state, n_control, n_lam=reduced_n_lam, stiffness=0)
lcs_learner.val_lcp_theta = 0 * true_lcp_theta + 1 * np.random.randn(true_lcp_theta.size)

# ============================= initialize the optimizer
optimizier = opt.Adam()
optimizier.learning_rate = 1e-2

# ============================= initialize an evaluator for the learned system
mpc_horizon = 5
controller_evaluator = TD.MPC_Controller(lcs_learner)
controller_evaluator.setCostFunction(Q, R, QN)
controller_evaluator.initializeMPC(mpc_horizon)

# # ============================= compute the true optimal cost (for reference, this is not always correct)
traj_count = 30
init_state_batch = np.random.randn(traj_count, n_state)
state_batch_traj = [list(init_state_batch)]
control_batch_traj = []
for t in range(control_horizon):
    # compute the control input using the current lcs learner
    st = time.time()
    control_batch_traj += [controller_evaluator.mpc(true_sys, state_batch_traj[-1])]
    # simulate to the next step
    next_state_batch, lam_batch = true_sys.dyn_step(state_batch_traj[-1], control_batch_traj[-1])
    state_batch_traj += [next_state_batch]
# reorganize the data format
control_traj_batch = TD.dataReorgnize(control_batch_traj)
state_traj_batch = TD.dataReorgnize(state_batch_traj)
true_sys_opt_cost_batch = controller_evaluator.computeCost(control_traj_batch, state_traj_batch)
print('True_sys optimal cost (for reference):', np.mean(true_sys_opt_cost_batch))

# ================= starting the data-driven model learning process
control_cost_trace = []
prev_control_traj_batch = []
prev_state_traj_batch = []
for control_iter in range(20):

    # # ============================= random the initial condition
    traj_count = 30
    init_state_batch = np.random.randn(traj_count, n_state)

    # # ============ use the mpc controller (based on the lcs model) to simulate the true system
    state_batch_traj = [list(init_state_batch)]
    control_batch_traj = []
    for t in range(control_horizon):
        # compute the control input using the current lcs learner
        st = time.time()
        control_batch_traj += [controller_evaluator.mpc(lcs_learner, state_batch_traj[-1])]
        # simulate to the next step
        next_state_batch, lam_batch = true_sys.dyn_step(state_batch_traj[-1], control_batch_traj[-1])
        state_batch_traj += [next_state_batch]
    # reorganize the data format
    control_traj_batch = TD.dataReorgnize(control_batch_traj)
    state_traj_batch = TD.dataReorgnize(state_batch_traj)

    # # plot for each iteration
    # for sys_trajectory in state_traj_batch:
    #     plt.plot(sys_trajectory)
    # plt.ylabel('state of true system')
    # plt.xlabel('time')
    # plt.show()

    # ============================= print
    true_sys_cost_batch = controller_evaluator.computeCost(control_traj_batch, state_traj_batch)
    print(
        '\n======================================================================'
        '\n|****** Control Iter:', control_iter,
        '| true_sys current cost:', np.mean(true_sys_cost_batch),
        '| true_sys optimal cost (for reference):', np.mean(true_sys_opt_cost_batch),
        '\n'
    )
    control_cost_trace += [np.mean(true_sys_cost_batch)]

    # ============================= learn the true lcs system from the true data
    print('lcs learning ')
    TD.LCSRegressionBuffer(lcs_learner, optimizier,
                           control_traj_batch, state_traj_batch,
                           prev_control_traj_batch, prev_state_traj_batch,
                           buffer_ratio=0.8,
                           max_iter=2000,
                           minibatch_size=100,
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
