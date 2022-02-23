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

# ============================= establish the lcs learner (automatically initialized)
lcs_learner = TD.LCS_learner_regression(n_state, n_control, n_lam, stiffness=0)
lcs_learner.val_lcp_theta = 0.0 * true_lcp_theta + 1. * np.random.randn(true_lcp_theta.size)

# ==============================    load the learned results
load = np.load('results.npy', allow_pickle=True).item()
control_cost_trace = load['control_cost_trace']
learned_lcs_mats = load['learned_lcs_mats']
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

# ============================= initialize an evaluator for the learned system
evaluator = TD.LCS_evaluation(lcs_learner)
evaluator.setCostFunction(Q, R, QN, control_horizon)
evaluator.differentiable()

# =============================== compute the optimal control inputs using learned lcs
traj_count = 50
init_state_batch = np.random.randn(traj_count, n_state)
learned_lcs_oc_solver = TD.LCS_MPC(learned_A, learned_B, learned_C, learned_D, learned_E, learned_F, learned_lcp_offset)
learned_lcs_oc_solver.oc_setup(control_horizon)
learned_lcs_opt_control_traj_batch = []
for i in range(traj_count):
    sol = learned_lcs_oc_solver.mpc(init_state_batch[i], Q, R, QN)
    learned_lcs_opt_control_traj_batch += [sol['control_traj_opt']]

true_sys_state_traj_batch, true_sys_lam_traj_batch = true_sys.sim_dyn(init_state_batch,
                                                                      learned_lcs_opt_control_traj_batch)
true_sys_cost_batch = evaluator.computeCost(learned_lcs_opt_control_traj_batch, true_sys_state_traj_batch)

for true_sys_trajectory in true_sys_state_traj_batch:
    plt.plot(true_sys_trajectory)

plt.ylabel('state of true system')
plt.xlabel('time')
plt.show()
print('learned control cost:', np.mean(true_sys_cost_batch))

plt.plot(control_cost_trace)
plt.ylabel('iteration')
plt.xlabel('control cost of true system')
plt.show()
