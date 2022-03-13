import matplotlib.pyplot as plt
import lcs.optim as opt
import lcs.task_driven as TD
from casadi import *
import time
import lcs.utility as utility


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

# define the true lcs system
true_lcs = TD.LCS_learner_regression(n_state, n_control, n_lam, A, B, C, D, E, G, H, lcp_offset, stiffness=0)
print('n_lam for the true system:', true_lcs.n_lam)

# # ============================= define the ture mpc controller for the true system
true_mpc_kl = TD.MPC_Controller_KL(true_lcs)
true_mpc_kl.set_cost_function(Q, R, QN)
mpc_horizon = 2
true_mpc_kl.initialize_mpc_KL(mpc_horizon)

# =========================== use the true mpc to simulate on the true lcs system
traj_count = 10
init_state_batch = np.random.randn(traj_count, n_state)
true_lcs_theta = true_lcs.computeLCSMats()
true_state_traj_batch, true_control_traj_batch, true_cost_batch = utility.simulate_mpc_on_lcs_kl(true_mpc_kl,
                                                                                                 true_lcs_theta,
                                                                                                 true_lcs,
                                                                                                 init_state_batch,
                                                                                                 control_horizon)
print('True_sys optimal cost (for reference):', np.mean(true_cost_batch))

# ============================= establish the lcs learner (automatically initialized)
reduced_n_lam = 2
lcs_learner = TD.LCS_learner_regression(n_state, n_control, n_lam=reduced_n_lam, stiffness=1)
lcs_learner.val_lcp_theta = 0.1 * np.random.randn(lcs_learner.n_lcp_theta)

# ============================= initialize an evaluator for the learned system
learner_mpc = TD.MPC_Controller_KL(lcs_learner)
learner_mpc.set_cost_function(Q, R, QN)
learner_mpc.initialize_mpc_KL(mpc_horizon)

# ============================= initialize the optimizer
optimizier = opt.Adam()
optimizier.learning_rate = 1e-3

# ================= starting the data-driven model learning process
control_cost_trace = []
lcs_theta_trace = []
prev_control_traj_batch = []
prev_state_traj_batch = []
prev_x_hist = []
prev_u_hist = []
kl_dist_trace = []
fig, axs = plt.subplots(3, 1)
for control_iter in range(20):
    # current lcs_learner parameter
    learner_lcs_theta = lcs_learner.computeLCSMats()

    # =========== use the mpc controller (based on the lcs model) to simulate the true system
    traj_count = 10
    init_state_batch = np.random.randn(traj_count, n_state)
    state_traj_batch, \
    control_traj_batch, \
    cost_batch = utility.simulate_mpc_on_lcs_kl(learner_mpc,
                                                learner_lcs_theta,
                                                true_lcs,
                                                init_state_batch,
                                                control_horizon,
                                                prev_state_traj_batch,
                                                prev_control_traj_batch)

    # compute histgram and plot
    x_batch, x_hist, u_batch, u_hist = utility.histogram(state_traj_batch, control_traj_batch)
    kl_distance = utility.kl_hist(x_hist, prev_x_hist)
    kl_dist_trace += [kl_distance]
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    axs[0].hist(x_batch[:, 0], bins=10, range=(-5, 5), density=True, facecolor='r')
    axs[1].hist(x_batch[:, 1], bins=10, range=(-5, 5), density=True, facecolor='g')
    axs[2].hist(x_batch[:, 2], bins=10, range=(-5, 5), density=True, facecolor='b')

    plt.pause(0.05)

    print(
        '\n======================================================================'
        '\n|****** Control Iter:', control_iter,
        '| current control cost:', np.mean(cost_batch),
        '| true control cost:', np.mean(true_cost_batch),
        '| kl distance:', kl_distance,
        '\n'
    )
    control_cost_trace += [np.mean(cost_batch)]
    lcs_theta_trace += [lcs_learner.computeLCSMats()]
    # ============================= learn the true lcs system from the true data
    print('lcs learning ')
    TD.LCSRegressionBuffer(lcs_learner, optimizier,
                           control_traj_batch, state_traj_batch,
                           prev_control_traj_batch, prev_state_traj_batch,
                           buffer_ratio=0.5,
                           max_iter=500,
                           minibatch_size=300,
                           print_level=1)

    # ============================= update the history data
    prev_control_traj_batch = control_traj_batch
    prev_state_traj_batch = state_traj_batch
    prev_x_hist = x_hist
    prev_u_hist = u_hist

# save the learned model
learned_lcs_mats = lcs_learner.computeLCSMats(compact=False)
learned_lcs_theta = lcs_learner.computeLCSMats()
np.save('results',
        {
            'control_cost_trace': control_cost_trace,
            'learned_lcs_mats': learned_lcs_mats,
            'learned_lcs_theta': learned_lcs_theta,
            'lcs_theta_trace': lcs_theta_trace,
            'Q': Q,
            'QN': QN,
            'R': R,
            'reduced_n_lam': reduced_n_lam,
            'control_horizon': control_horizon,
            'mpc_horizon': mpc_horizon,
            'kl_dist_trace': kl_dist_trace,
        })
