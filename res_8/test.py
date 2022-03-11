import matplotlib.pyplot as plt
import lcs.utility as utility
import lcs.optim as opt
import lcs.task_driven as TD
import numpy as np
from casadi import *
import numpy.linalg as la


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
learned_lcs_theta = load['learned_lcs_theta']
lcs_theta_trace = load['lcs_theta_trace']
mpc_horizon = load['mpc_horizon']
control_horizon = load['control_horizon']
kl_distance_trace=load['kl_dist_trace']
Q = load['Q']
R = load['R']
QN = load['QN']

# plot the loss
diff_theta = []
max_iter = len(lcs_theta_trace)
for i in range(max_iter - 1):
    diff_theta += [la.norm(lcs_theta_trace[i + 1] - lcs_theta_trace[i])]

plt.figure(1)
plt.plot(control_cost_trace)
plt.xlabel('iteration')
plt.ylabel('control cost of true system')

plt.figure(2)
plt.plot(diff_theta)
plt.xlabel('iteration')
plt.ylabel('update of lcs')


plt.figure(3)
plt.plot(kl_distance_trace)
plt.xlabel('iteration')
plt.ylabel('kl_distance_trace')


plt.show()

# # ============================= define the ture mpc controller for the true system
true_mpc = TD.MPC_Controller(true_sys)
true_mpc.set_cost_function(Q, R, QN)
true_mpc.initialize_mpc(mpc_horizon)

# ============================= establish the lcs learner (automatically initialized)
reduced_n_lam = load['reduced_n_lam']
print('n_lam for the learner:', reduced_n_lam)
lcs_learner = TD.LCS_learner_regression(n_state, n_control, n_lam=reduced_n_lam, stiffness=1)
# lcs_learner.val_lcs_theta = learned_lcs_theta
lcs_learner.val_lcs_theta = learned_lcs_theta

# ============================= initialize an mpc for the learned system

learner_mpc = TD.MPC_Controller(lcs_learner)
learner_mpc.set_cost_function(Q, R, QN)
learner_mpc.initialize_mpc(mpc_horizon)

# # ============================= compute the true optimal cost (for reference, this is not always correct)
traj_count = 30
init_state_batch = np.random.randn(traj_count, n_state)
state_traj_batch, control_traj_batch, cost_batch = utility.simulate_mpc_on_lcs(
    # true_mpc,
    # true_sys,
    learner_mpc,
    None,
    true_sys,
    init_state_batch,
    control_horizon,
    lcs_theta_trace[-2]
)

# plot
for sys_trajectory in state_traj_batch:
    plt.plot(sys_trajectory)
plt.ylabel('state of true system')
plt.xlabel('time')
plt.show()
