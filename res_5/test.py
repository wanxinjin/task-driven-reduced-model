import matplotlib.pyplot as plt

import lcs.optim as opt
import lcs.task_driven as TD
import numpy as np
from casadi import *


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

# ==============================    load the learned results
load = np.load('results.npy', allow_pickle=True).item()
control_cost_trace = load['control_cost_trace']
learned_lcs_mats = load['learned_lcs_mats']
learned_lcs_theta = load['learned_lcs_theta']
learned_A = learned_lcs_mats['A']
learned_B = learned_lcs_mats['B']
learned_C = learned_lcs_mats['C']
learned_D = learned_lcs_mats['D']
learned_E = learned_lcs_mats['E']
learned_F = learned_lcs_mats['F']
learned_lcp_offset = learned_lcs_mats['lcp_offset']

control_horizon = load['control_horizon']
Q = load['Q']
R = load['R']
QN = load['QN']

# plot the loss
plt.plot(control_cost_trace)
plt.xlabel('iteration')
plt.ylabel('control cost of true system')
plt.show()

# ============================= establish the lcs learner (automatically initialized)
reduced_n_lam = load['reduced_n_lam']
print('n_lam for the learner:', reduced_n_lam)

lcs_learner = TD.LCS_learner_regression(n_state, n_control, n_lam=reduced_n_lam, stiffness=1)

# ============================= initialize an evaluator for the learned system
mpc_horizon = load['mpc_horizon']
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
    control_batch_traj += [controller_evaluator.mpc(None, state_batch_traj[-1], learned_lcs_theta)]
    # simulate to the next step
    next_state_batch, lam_batch = true_sys.dyn_step(state_batch_traj[-1], control_batch_traj[-1])
    state_batch_traj += [next_state_batch]
# reorganize the data format
control_traj_batch = TD.dataReorgnize(control_batch_traj)
state_traj_batch = TD.dataReorgnize(state_batch_traj)
true_sys_opt_cost_batch = controller_evaluator.computeCost(control_traj_batch, state_traj_batch)
print('control cost on real system', np.mean(true_sys_opt_cost_batch))


# plot
for sys_trajectory in state_traj_batch:
    plt.plot(sys_trajectory)
plt.ylabel('state of true system')
plt.xlabel('time')
plt.show()

