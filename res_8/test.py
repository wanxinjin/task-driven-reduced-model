import matplotlib.pyplot as plt
import lcs.utility as utility
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
lcs_learner.val_lcs_theta = learned_lcs_theta

# ============================= initialize an evaluator for the learned system
mpc_horizon = load['mpc_horizon']
learner_mpc = TD.MPC_Controller(lcs_learner)
learner_mpc.set_cost_function(Q, R, QN)
learner_mpc.initialize_mpc(mpc_horizon)

# # ============================= compute the true optimal cost (for reference, this is not always correct)
traj_count = 30
init_state_batch = np.random.randn(traj_count, n_state)
state_traj_batch, control_traj_batch, cost_batch = utility.simulate_mpc_on_lcs(learner_mpc,
                                                                               lcs_learner,
                                                                               true_sys,
                                                                               init_state_batch,
                                                                               control_horizon)

# plot
for sys_trajectory in state_traj_batch:
    plt.plot(sys_trajectory)
plt.ylabel('state of true system')
plt.xlabel('time')
plt.show()
