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

# control cost function
control_horizon = 20
Q = 2 * np.eye(n_state)
QN = Q
R = np.eye(n_control)

# ============================= establish the lcs learner (automatically initialized)
lcs_learner = TD.LCS_learner_regression(n_state, n_control, n_lam, stiffness=0)
lcs_learner.val_lcp_theta = true_lcp_theta + 1. * np.random.randn(true_lcp_theta.size)
# lcs_learner = TD.LCS_learner(n_state, n_control, n_lam, stiffness=0)
# lcs_learner.val_theta = true_theta
# ============================= starting from 10 various initial conditions
traj_count = 50
init_state_batch = np.random.randn(traj_count, n_state)
optimizier = opt.Adam()
optimizier.learning_rate = 1e-2

# ============================= initialize a random control sequence for each initial condition
control_traj_batch = TD.randomControlTraj(traj_count, control_horizon, n_control)

# ============================= initialize an evaluator for the learned system
evaluator = TD.LCS_evaluation(lcs_learner)
evaluator.setCostFunction(Q, R, QN, control_horizon)
evaluator.differentiable()

# =============================== compute the true optimal cost (for comparison)
true_sys_oc_solver = TD.LCS_MPC(A, B, C, D, E, F, lcp_offset)
true_sys_oc_solver.oc_setup(control_horizon)
true_sys_opt_control_traj_batch = []
for i in range(traj_count):
    true_sys_sol = true_sys_oc_solver.mpc(init_state_batch[i], Q, R, QN)
    true_sys_opt_control_traj_batch += [true_sys_sol['control_traj_opt']]

true_sys_opt_state_traj_batch, _ = true_sys.sim_dyn(init_state_batch, true_sys_opt_control_traj_batch)
true_sys_opt_cost_batch = evaluator.computeCost(true_sys_opt_control_traj_batch, true_sys_opt_state_traj_batch)

# ================= starting the learning process
control_cost_trace = []
for i in range(5000):
    # ============================= sample from the true system to obtain the state trajectory
    true_state_traj_batch, true_lam_traj_batch = true_sys.sim_dyn(init_state_batch, control_traj_batch)

    # ============================= compute the control cost for the true system
    true_sys_cost_batch = evaluator.computeCost(control_traj_batch, true_state_traj_batch)

    # ============================= learn the true lcs system from the true data
    TD.LCSLearningRegression(lcs_learner, optimizier, control_traj_batch, true_state_traj_batch, max_iter=1000,
                             print_level=1)
    # TD.LCSLearning(lcs_learner, optimizier, control_traj_batch, true_state_traj_batch)

    # =========================== evaluate the current lcs and compute the gradient
    control_traj_batch, model_cost_batch, updated_model_cost_batch = evaluator.EvaluateMS(lcs_learner,
                                                                                          init_state_batch,
                                                                                          control_traj_batch,
                                                                                          true_state_traj_batch)

    # =========================== print
    updated_true_state_traj_batch, _ = true_sys.sim_dyn(init_state_batch, control_traj_batch)
    updated_true_sys_cost_batch = evaluator.computeCost(control_traj_batch, updated_true_state_traj_batch)
    print(
        '==================================================\n'
        '| Control Iter:', i,
        # '| model_cost:', np.mean(model_cost_batch),
        '| learned lcs model current cost:', np.mean(updated_model_cost_batch),
        # '| true_sys_cost:', np.mean(true_sys_cost_batch),
        '| true_sys current cost:', np.mean(updated_true_sys_cost_batch),
        '| true_sys optimal cost:', np.mean(true_sys_opt_cost_batch),
        '\n=================================================='
    )
    control_cost_trace += [np.mean(updated_true_sys_cost_batch)]

