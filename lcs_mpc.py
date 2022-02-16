import lcs.task_driven as TD
import lcs.optim as opt
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import time
import numpy.linalg as la

# load true lcs system
load_lcs = np.load('random_lcs.npy', allow_pickle=True).item()
n_state = load_lcs['n_state']
n_control = load_lcs['n_control']
n_lam = load_lcs['n_lam']
A = load_lcs['A']
B = load_lcs['B']
C = load_lcs['C']
D = load_lcs['D']
E = load_lcs['E']
F = load_lcs['F']
lcp_offset = load_lcs['lcp_offset']

true_oc = TD.LCS_MPC(A=A, B=B, C=C, D=D, E=E, F=F, lcp_offset=lcp_offset)
true_oc.oc_setup(mpc_horizon=20)
Q = np.array([[10, 0, 0], [0, 10, 0], [0, 0, .1]])
R = np.diag([0.05, 0.05])
QN = scipy.linalg.solve_discrete_are(A, B, Q, R)

init_state = 1 * np.random.randn(n_state)
# init_state = np.array([0.1, 0.3, 0])
true_sol = true_oc.mpc(init_state, Q, R, QN)
true_x_traj = true_sol['state_traj_opt']
true_u_traj = true_sol['control_traj_opt']
true_lam_traj = true_sol['lam_traj_opt']

# load the learned lcs system
learned_lcs = np.load('learned.npy', allow_pickle=True).item()
learned_A = learned_lcs['learned_A']
learned_B = learned_lcs['learned_B']
learned_C = learned_lcs['learned_C']
learned_D = learned_lcs['learned_D']
learned_E = learned_lcs['learned_E']
learned_F = learned_lcs['learned_F']
learned_lcp_offset = learned_lcs['learned_lcp_offset']
learned_oc = TD.LCS_MPC(A=learned_A, B=learned_B, C=learned_C, D=learned_D, E=learned_E, F=learned_F,
                        lcp_offset=learned_lcp_offset)
learned_oc.oc_setup(mpc_horizon=20)

# init_state = np.array([0.1, 0.3, 0])
learned_sol = learned_oc.mpc(init_state, Q, R, QN)
learned_x_traj = learned_sol['state_traj_opt']
learned_u_traj = learned_sol['control_traj_opt']
learned_lam_traj = learned_sol['lam_traj_opt']

plt.figure(1)
plt.plot(true_x_traj[:,0])
plt.plot(learned_x_traj[:,0])
plt.figure(2)
plt.plot(true_x_traj[:,1])
plt.plot(learned_x_traj[:,1])
plt.figure(3)
plt.plot(true_x_traj[:,2])
plt.plot(learned_x_traj[:,2])

#
# plt.figure(4)
# plt.plot(true_lam_traj[:,0])
# plt.plot(learned_lam_traj[:,0])
# plt.figure(5)
# plt.plot(true_lam_traj[:,1])
# plt.plot(learned_lam_traj[:,1])




plt.show()