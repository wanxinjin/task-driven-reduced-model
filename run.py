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
control_horizon = 4
Q = 2 * np.eye(n_state)
QN = Q
R = np.eye(n_control)

# ============================= establish the lcs learner (automatically initialized)
lcs_learner = TD.LCS_learner_regression(n_state, n_control, n_lam, stiffness=0)
lcs_learner.val_lcp_theta = 0.0 * true_lcp_theta + 1. * np.random.randn(true_lcp_theta.size)

# ============================= starting from 10 various initial conditions
traj_count = 60
init_state_batch = np.random.randn(traj_count, n_state)
optimizier = opt.Adam()
optimizier.learning_rate = 1e-2

# ============================= initialize a random control sequence for each initial condition
control_traj_batch = TD.randomControlTraj(traj_count, control_horizon, n_control)

# ============================= initialize an evaluator for the learned system
evaluator = TD.LCS_evaluation(lcs_learner)
evaluator.setCostFunction(Q, R, QN, control_horizon)
evaluator.differentiable()

# =============================== compute the true optimal cost (for reference, this is not always correct)
true_sys_oc_solver = TD.LCS_MPC(A, B, C, D, E, F, lcp_offset)
true_sys_oc_solver.oc_setup(control_horizon)
true_sys_opt_control_traj_batch = []
for i in range(traj_count):
    true_sys_sol = true_sys_oc_solver.mpc(init_state_batch[i], Q, R, QN)
    true_sys_opt_control_traj_batch += [true_sys_sol['control_traj_opt']]
true_sys_opt_state_traj_batch, true_sys_opt_lam_traj_batch = true_sys.sim_dyn(init_state_batch,
                                                                              true_sys_opt_control_traj_batch)
true_sys_opt_cost_batch = evaluator.computeCost(true_sys_opt_control_traj_batch, true_sys_opt_state_traj_batch)

# ================= starting the data-driven model learning process
control_cost_trace = []
for control_iter in range(500):
    # ============================= sample from the true system to obtain the state trajectory
    true_state_traj_batch, true_lam_traj_batch = true_sys.sim_dyn(init_state_batch, control_traj_batch)

    # ============================= compute the control cost for the true system
    true_sys_cost_batch = evaluator.computeCost(control_traj_batch, true_state_traj_batch)

    # ============================= learn the true lcs system from the true data
    print('======================================================================')
    print('| Control Iter:', control_iter)
    if control_iter < 2:
        lcs_max_iter = 1000
    else:
        lcs_max_iter = 500  # why: because the lcs_learner is always warm started with the previous control iteration
    TD.LCSLearningRegression(lcs_learner, optimizier, control_traj_batch, true_state_traj_batch, max_iter=lcs_max_iter,
                             print_level=1)

    # =========================== evaluate the current lcs and compute the gradient
    new_control_traj_batch = evaluator.EvaluateMS(lcs_learner, init_state_batch, control_traj_batch)

    # =========================== update the control trajectories
    control_traj_batch = new_control_traj_batch

    # print
    updated_true_sys_state_traj_batch, _ = true_sys.sim_dyn(init_state_batch, control_traj_batch)
    updated_true_sys_cost_batch = evaluator.computeCost(control_traj_batch, updated_true_sys_state_traj_batch)
    updated_model_state_traj_batch, _ = lcs_learner.sim_dyn(init_state_batch, control_traj_batch)
    updated_model_cost_batch = evaluator.computeCost(control_traj_batch, updated_model_state_traj_batch)

    print(
        '\n|****** Control Iter:', control_iter,
        '| learned lcs model current cost:', np.mean(updated_model_cost_batch),
        '| true_sys current cost:', np.mean(updated_true_sys_cost_batch),
        '| true_sys optimal cost (could be wrong):', np.mean(true_sys_opt_cost_batch),
        '\n'
    )
    control_cost_trace += [np.mean(updated_true_sys_cost_batch)]

# save the learned model
learned_lcs_mats = lcs_learner.computeLCSMats(compact=False)
np.save('results',
        {
            'control_cost_trace': control_cost_trace,
            'control_traj_batch': control_traj_batch,
            'init_state_batch': init_state_batch,
            'learned_lcs_mats': learned_lcs_mats,
            'Q': Q,
            'QN': QN,
            'R': R
        })
