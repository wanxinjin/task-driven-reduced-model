import lcs.optim as opt
import lcs.task_driven as TD
import numpy.linalg as la
from casadi import *
import matplotlib.pyplot as plt


def print(*args):
    __builtins__.print(*("%.5f" % a if isinstance(a, float) else a
                         for a in args))


# color list
color_list = np.linspace(0, 1, 10)

# ==============================   load the generated LCS system   ==================================
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

# ==============================   generate the training data    ========================================
# create the data generator
data_generator = TD.LCS_learner(n_state, n_control, n_lam, A, B, C, D, E, G, H, lcp_offset, stiffness=0)
train_data_size = 1000
train_x_batch = 1 * np.random.uniform(-1, 1, size=(train_data_size, n_state))
train_u_batch = 5 * np.random.uniform(-1, 1, size=(train_data_size, n_control))
train_x_next_batch, train_lam_opt_batch = data_generator.dyn_prediction(train_x_batch, train_u_batch, theta_val=[])
train_mode_list, train_mode_frequency_list = TD.statiModes(train_lam_opt_batch)
print('number of modes in the training data:', train_mode_frequency_list.size)
print('mode frequency in the training data: ', train_mode_frequency_list)
# check the mode index
train_mode_list, train_mode_indices = TD.plotModes(train_lam_opt_batch)

# =============== plot the training data, each color for each mode  ======================================
# plot dimension index
plot_x_indx = 0
plot_y_indx = 0

plt.figure()
plt.title('True modes marked in (o)')
train_x = train_x_batch[:, plot_x_indx]
train_y = train_x_next_batch[:, plot_y_indx]
plt.scatter(train_x, train_y, c=color_list[train_mode_indices], s=30)

# initialize the plot of the learned learned results
plt.ion()
fig, ax = plt.subplots()
ax.set_title('Learned modes marked in (+) \n True modes marked in (o)')
train_x = train_x_batch[:, plot_x_indx]
train_y = train_x_next_batch[:, plot_y_indx]
plt.scatter(train_x, train_y, c=color_list[train_mode_indices], s=80, alpha=0.3)
pred_x, pred_y = [], []
sc = ax.scatter(pred_x, pred_y, s=30, marker="+", cmap='paried')

fig2, ax = plt.subplots()
ax.set_title('Learned modes marked in (+)')
train_x = train_x_batch[:, plot_x_indx]
train_y = train_x_next_batch[:, plot_y_indx]
plt.scatter(train_x, train_y, c=color_list[train_mode_indices], s=0, alpha=0.3)
pred_x, pred_y = [], []
sc2 = ax.scatter(pred_x, pred_y, s=30, marker="+", cmap='paried')
plt.draw()

# ==============================   create the learner object    ========================================
learner = TD.LCS_learner(n_state, n_control, n_lam=2, stiffness=10)
# ================================   beginning the training process    ======================================
# doing learning process
curr_theta = 1. * np.random.randn(learner.n_theta)
# curr_theta = true_theta + 0.5 * np.random.randn(learner.n_theta)
mini_batch_size = 100
loss_trace = []
theta_trace = []
optimizier = opt.Adam()
optimizier.learning_rate = 1e-2
# epsilon = np.random.random(5000)
epsilon = np.logspace(3,-2,5000)
for k in range(5000):
    # mini batch dataset
    shuffle_index = np.random.permutation(train_data_size)[0:mini_batch_size]
    x_mini_batch = train_x_batch[shuffle_index]
    u_mini_batch = train_u_batch[shuffle_index]
    x_next_mini_batch = train_x_next_batch[shuffle_index]
    lam_mini_batch = train_lam_opt_batch[shuffle_index]

    # compute the lambda batch
    learner.differetiable(gamma=1, epsilon=epsilon[k])
    lam_phi_opt_mini_batch, loss_opt_batch = learner.compute_lambda(x_mini_batch, u_mini_batch, x_next_mini_batch,
                                                                    curr_theta)

    # compute the gradient
    dtheta, loss, dyn_loss, lcp_loss, dtheta_hessian = \
        learner.gradient_step(x_mini_batch, u_mini_batch, x_next_mini_batch, curr_theta, lam_phi_opt_mini_batch,
                              second_order=False)

    # store and update
    loss_trace += [loss]
    theta_trace += [curr_theta]
    curr_theta = optimizier.step(curr_theta, dtheta)
    # curr_theta = optimizier.step(curr_theta, dtheta_hessian)

    if k % 100 == 0:
        # on the prediction using the current learned lcs
        pred_x_next_batch, pred_lam_batch = learner.dyn_prediction(train_x_batch, train_u_batch, curr_theta)

        # compute the prediction error
        error_x_next_batch = pred_x_next_batch - train_x_next_batch
        relative_error = (la.norm(error_x_next_batch, axis=1) / (la.norm(train_x_next_batch, axis=1) + 0.0001)).mean()

        # compute the predicted mode statistics
        pred_mode_list, pred_mode_indices = TD.plotModes(pred_lam_batch)

        # plot the learned mode
        pred_x = train_x_batch[:, plot_x_indx]
        pred_y = pred_x_next_batch[:, plot_y_indx]
        sc.set_offsets(np.c_[pred_x, pred_y])
        sc.set_array(color_list[pred_mode_indices])
        sc2.set_offsets(np.c_[pred_x, pred_y])
        sc2.set_array(color_list[pred_mode_indices])
        fig.canvas.draw_idle()
        plt.pause(0.1)

        print(
            k,
            '| loss:', loss,
            '| loss:', dyn_loss + lcp_loss / epsilon[k],
            '| grad:', norm_2(dtheta),
            '| dyn:', dyn_loss,
            '| lcp:', lcp_loss,
            '| RPE:', relative_error,
            '| PMC:', len(pred_mode_list),
            '| Epsilon:', epsilon[k],
        )

# save


# ================================   do some anlaysis for the learned    ======================================
# on the prediction using the current learned lcs
pred_x_next_batch, pred_lam_batch = learner.dyn_prediction(train_x_batch, train_u_batch, curr_theta)
# compute the overall relative prediction error
error_x_next_batch = pred_x_next_batch - train_x_next_batch
relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(train_x_next_batch, axis=1)).mean()
# compute the predicted mode statistics
pred_mode_list0, pred_mode_frequency_list = TD.statiModes(pred_lam_batch)
pred_mode_list1, pred_mode_indices = TD.plotModes(pred_lam_batch)
pred_error_per_mode_list = []
for i in range(len(pred_mode_list0)):
    mode_i_index = np.where(pred_mode_indices == i)
    mode_i_error = error_x_next_batch[mode_i_index]
    mode_i_relative_error = (la.norm(mode_i_error, axis=1) / la.norm(train_x_next_batch[mode_i_index], axis=1)).mean()
    pred_error_per_mode_list += [mode_i_relative_error]

print(pred_mode_list0)
print(pred_mode_list1)
print(pred_mode_frequency_list)
print(pred_error_per_mode_list)

# take out the plot dimension
pred_x = train_x_batch[:, plot_x_indx]
pred_y = pred_x_next_batch[:, plot_y_indx]

learned_A = learner.A_fn(curr_theta).full()
learned_B = learner.B_fn(curr_theta).full()
learned_C = learner.C_fn(curr_theta).full()
learned_D = learner.D_fn(curr_theta).full()
learned_E = learner.E_fn(curr_theta).full()
learned_G = learner.G_fn(curr_theta).full()
learned_H = learner.H_fn(curr_theta).full()
learned_F = learner.F_fn(curr_theta).full()
learned_lcp_offset = learner.lcp_offset_fn(curr_theta).full()

np.save('learned', {
    'theta_trace': theta_trace,
    'loss_trace': loss_trace,
    'color_list': color_list,
    'train_x': train_x,
    'train_y': train_y,
    'train_mode_indices': train_mode_indices,
    'train_mode_list': train_mode_list,
    'train_mode_count': train_mode_frequency_list.size,
    'train_mode_frequency': train_mode_frequency_list,
    'pred_y': pred_y,
    'pred_mode_indices': pred_mode_indices,
    'pred_mode_list': pred_mode_list1,
    'pred_mode_frequency': pred_mode_frequency_list,
    'pred_error_per_mode_list': pred_error_per_mode_list,
    'pred_mode_count': len(pred_mode_list1),
    'relative_error': relative_error,
    'plot_x_index': plot_x_indx,
    'plot_y_index': plot_x_indx,
    'learned_A': learned_A,
    'learned_B': learned_B,
    'learned_C': learned_C,
    'learned_D': learned_D,
    'learned_E': learned_E,
    'learned_G': learned_G,
    'learned_H': learned_H,
    'learned_F': learned_F,
    'learned_lcp_offset': learned_lcp_offset,
})
