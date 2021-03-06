import numpy as np
from casadi import *
import numpy.linalg as la
import matplotlib.pyplot as plt


# class for learning LCS from the hybrid data
class LCS_learner:
    def __init__(self, n_state, n_control, n_lam,
                 A=None, B=None, C=None, D=None, E=None, G=None, H=None, lcp_offset=None,
                 stiffness=0.):
        self.n_lam = n_lam
        self.n_state = n_state
        self.n_control = n_control

        self.lam = SX.sym('lam', self.n_lam)
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)

        self.theta = []

        if A is None:
            self.A = SX.sym('A', self.n_state, self.n_state)
            self.theta += [vec(self.A)]
        else:
            self.A = DM(A)

        if B is None:
            self.B = SX.sym('B', self.n_state, self.n_control)
            self.theta += [vec(self.B)]
        else:
            self.B = DM(B)

        if C is None:
            self.C = SX.sym('C', self.n_state, self.n_lam)
            self.theta += [vec(self.C)]
        else:
            self.C = DM(C)

        if D is None:
            self.D = SX.sym('D', self.n_lam, self.n_state)
            self.theta += [vec(self.D)]
        else:
            self.D = DM(D)

        if E is None:
            self.E = SX.sym('E', self.n_lam, self.n_control)
            self.theta += [vec(self.E)]
        else:
            self.E = DM(E)

        if G is None:
            self.G = SX.sym('G', self.n_lam, self.n_lam)
            self.theta += [vec(self.G)]
        else:
            self.G = DM(G)

        if H is None:
            self.H = SX.sym('H', self.n_lam, self.n_lam)
            self.theta += [vec(self.H)]
        else:
            self.H = DM(H)

        if lcp_offset is None:
            self.lcp_offset = SX.sym('lcp_offset', self.n_lam)
            self.theta += [vec(self.lcp_offset)]
        else:
            self.lcp_offset = DM(lcp_offset)

        self.theta = vcat(self.theta)
        self.n_theta = self.theta.numel()

        self.F = stiffness * np.eye(self.n_lam) + self.G @ self.G.T + self.H - self.H.T
        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.G_fn = Function('G_fn', [self.theta], [self.G])
        self.H_fn = Function('H_fn', [self.theta], [self.H])
        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.B_fn = Function('B_fn', [self.theta], [self.B])
        self.C_fn = Function('C_fn', [self.theta], [self.C])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])
        # self.dyn_offset_fn = Function('dyn_offset_fn', [self.theta], [self.dyn_offset])

        # initialize the parameter
        self.val_theta = 0.1 * np.random.randn(self.n_theta)

    def differetiable(self, gamma=1e-3, epsilon=0.5):

        # define the dynamics loss
        self.x_next = SX.sym('x_next', self.n_state)
        data = vertcat(self.x, self.u, self.x_next)
        # self.dyn = self.A @ self.x + self.B @ self.u + self.C @ self.lam + self.dyn_offset
        self.dyn = self.A @ self.x + self.B @ self.u + self.C @ self.lam
        dyn_loss = dot(self.dyn - self.x_next, self.dyn - self.x_next)

        # lcp loss
        self.dist = self.D @ self.x + self.E @ self.u + self.F @ self.lam + self.lcp_offset
        self.phi = SX.sym('phi', self.n_lam)
        lcp_loss = dot(self.lam, self.phi) + 1 / gamma * dot(self.phi - self.dist,
                                                             self.phi - self.dist)

        # total loss
        # loss = (1 - epsilon) * dyn_loss + epsilon * lcp_loss
        loss = dyn_loss + lcp_loss / epsilon
        # loss = dot(self.dyn[2:4] - self.x_next[2:4], self.dyn[2:4] - self.x_next[2:4]) + lcp_loss / epsilon
        # loss = (dyn_loss + lcp_loss / epsilon) / (0.5+dot(self.x_next, self.x_next))

        # establish the qp solver
        lam_phi = vertcat(self.lam, self.phi)
        data_theta = vertcat(self.x, self.u, self.x_next, self.theta)
        quadprog = {'x': vertcat(self.lam, self.phi), 'f': loss, 'p': data_theta}
        opts = {'printLevel': 'none', }
        self.inner_QPSolver = qpsol('inner_QPSolver', 'qpoases', quadprog, opts)

        # compute the jacobian from lam to theta
        self.loss_fn = Function('loss_fn', [data, self.theta, lam_phi], [loss])
        self.dloss_fn = Function('dloss_fn', [data, self.theta, lam_phi], [jacobian(loss, self.theta).T])
        self.dyn_loss_fn = Function('dyn_loss_fn', [data, self.theta, lam_phi], [dyn_loss])
        self.lcp_loss_fn = Function('lcp_loss_fn', [data, self.theta, lam_phi], [lcp_loss])

        # compute the second order derivative
        grad_loss = jacobian(loss, lam_phi).T
        L = diag(lam_phi) @ grad_loss
        self.L_fn = Function('L_fn', [data, self.theta, lam_phi], [L])  # this is just for testing
        # compute the gradient of lam_phi_opt with respect to theta
        dL_dsol = jacobian(L, lam_phi)
        dL_dtheta = jacobian(L, self.theta)
        dsol_dtheta = -inv(dL_dsol) @ dL_dtheta
        self.dsol_dtheta_fn = Function('dsol_dtheta_fn', [data, self.theta, lam_phi], [dsol_dtheta])
        # this is just for testing
        dloss2 = jacobian(loss, self.theta) + jacobian(loss, lam_phi) @ dsol_dtheta
        self.dloss2_fn = Function('dloss2_fn', [data, self.theta, lam_phi], [dloss2.T])
        # compute the second order derivative
        dloss_dtheta = jacobian(loss, self.theta).T
        ddloss = jacobian(dloss_dtheta, self.theta) + jacobian(dloss_dtheta, lam_phi) @ dsol_dtheta
        self.ddloss_fn = Function('ddloss_fn', [data, self.theta, lam_phi], [ddloss])

    def compute_lambda(self, x_batch, u_batch, x_next_batch):

        # prepare the data
        batch_size = x_batch.shape[0]
        data_batch = np.hstack((x_batch, u_batch, x_next_batch))
        theta_val_batch = np.tile(self.val_theta, (batch_size, 1))
        data_theta_batch = np.hstack((data_batch, theta_val_batch))

        # compute the lam_phi solution
        sol_batch = self.inner_QPSolver(lbx=0.0, p=data_theta_batch.T)
        loss_opt_batch = sol_batch['f'].full().flatten()
        lam_phi_opt_batch = sol_batch['x'].full().T

        return lam_phi_opt_batch, loss_opt_batch

    def gradient_step(self, x_batch, u_batch, x_next_batch, lam_phi_opt_batch, second_order=False):
        batch_size = x_batch.shape[0]
        data_batch = np.hstack((x_batch, u_batch, x_next_batch))
        theta_val_batch = np.tile(self.val_theta, (batch_size, 1))

        # compute the gradient value
        dtheta_batch = self.dloss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
        dtheta_mean = dtheta_batch.full().mean(axis=1)

        # compute the losses
        loss_batch = self.loss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
        dyn_loss_batch = self.dyn_loss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
        lcp_loss_batch = self.lcp_loss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
        loss_mean = loss_batch.full().mean()
        dyn_loss_mean = dyn_loss_batch.full().mean()
        lcp_loss_mean = lcp_loss_batch.full().mean()

        dtheta_hessian = dtheta_mean

        if second_order is True:
            hessian_batch = self.ddloss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
            # compute the mean hessian
            hessian_sum = 0
            for i in range(batch_size):
                hessian_i = hessian_batch[:, i * self.n_theta:(i + 1) * self.n_theta]
                hessian_sum += hessian_i
            hessian_mean = hessian_sum / batch_size
            damping_factor = 1
            u, s, vh = np.linalg.svd(hessian_mean)
            s = s + damping_factor
            damped_hessian = u @ np.diag(s) @ vh
            dtheta_hessian = (inv(damped_hessian) @ DM(dtheta_mean)).full().flatten()

        return dtheta_mean, loss_mean, dyn_loss_mean, lcp_loss_mean, dtheta_hessian

    def dyn_prediction(self, x_batch, u_batch):
        self.differetiable()

        batch_size = x_batch.shape[0]
        theta_val_batch = np.tile(self.val_theta, (batch_size, 1))
        xu_theta_batch = np.hstack((x_batch, u_batch, theta_val_batch))

        # establish the lcp solver
        lcp_loss = dot(self.dist, self.lam)
        xu_theta = vertcat(self.x, self.u, self.theta)
        quadprog = {'x': self.lam, 'f': lcp_loss, 'g': self.dist, 'p': xu_theta}
        opts = {'printLevel': 'none'}
        lcp_Solver = qpsol('lcp_solver', 'qpoases', quadprog, opts)
        self.lcp_fn = Function('dist_fn', [self.x, self.u, self.lam, self.theta], [self.dist, dot(self.dist, self.lam)])
        self.lcp_dist_fn = Function('dist_fn', [self.x, self.u, self.lam, self.theta], [self.dist])

        # establish the dynamics equation
        dyn_fn = Function('dyn_fn', [self.x, self.u, self.lam, self.theta], [self.dyn])

        # compute the lam_batch
        sol_batch = lcp_Solver(lbx=0., lbg=0., p=xu_theta_batch.T)
        lam_opt_batch = sol_batch['x'].full().T

        # compute the next state batch
        x_next_batch = dyn_fn(x_batch.T, u_batch.T, lam_opt_batch.T, theta_val_batch.T).full().T

        return x_next_batch, lam_opt_batch


# learn a lcs model from the sampled data using l4dc paper
def LCSLearning(lcs_learner, optimizer, control_traj_batch, true_state_traj_batch,
                max_iter=5000, minibatch_size=100):
    # converting the data form
    batch_size = len(control_traj_batch)
    train_u_batch = []
    train_x_batch = []
    train_x_next_batch = []
    for i in range(batch_size):
        train_u_batch += [control_traj_batch[i]]
        train_x_batch += [true_state_traj_batch[i][0:-1]]
        train_x_next_batch += [true_state_traj_batch[i][1:]]
    train_u_batch = np.vstack(train_u_batch)
    train_x_next_batch = np.vstack(train_x_next_batch)
    train_x_batch = np.vstack(train_x_batch)

    # do the learning iteration
    train_data_size = train_u_batch.shape[0]
    epsilon = np.logspace(3, -2, max_iter)
    for k in range(max_iter):
        # mini batch dataset
        shuffle_index = np.random.permutation(train_data_size)[0:minibatch_size]
        x_minibatch = train_x_batch[shuffle_index]
        u_minibatch = train_u_batch[shuffle_index]
        x_next_minibatch = train_x_next_batch[shuffle_index]

        # compute the lambda batch
        lcs_learner.differetiable(gamma=1e-3, epsilon=epsilon[k])
        lam_phi_opt_mini_batch, loss_opt_batch = lcs_learner.compute_lambda(x_minibatch, u_minibatch, x_next_minibatch)

        # compute the gradient
        dtheta, loss, dyn_loss, lcp_loss, dtheta_hessian = \
            lcs_learner.gradient_step(x_minibatch, u_minibatch, x_next_minibatch, lam_phi_opt_mini_batch,
                                      second_order=False)

        # store and update
        lcs_learner.val_theta = optimizer.step(lcs_learner.val_theta, dtheta)

        if k % 100 == 0:
            # on the prediction using the current learned lcs
            pred_x_next_batch, pred_lam_batch = lcs_learner.dyn_prediction(train_x_batch, train_u_batch)

            # compute the prediction error
            error_x_next_batch = pred_x_next_batch - train_x_next_batch
            relative_error = (
                    la.norm(error_x_next_batch, axis=1) / (la.norm(train_x_next_batch, axis=1) + 0.0001)).mean()

            print(
                k,
                '| loss:', loss,
                '| dyn_loss:', dyn_loss,
                '| lcp_loss:', lcp_loss,
                '| grad:', norm_2(dtheta),
                '| PRE:', relative_error,
                '| epsilon:', epsilon[k],
            )


# class for learning LCS from the hybrid data
class LCS_learner_regression:
    def __init__(self, n_state, n_control, n_lam,
                 A=None, B=None, C=None, D=None, E=None, G=None, H=None, lcp_offset=None,
                 stiffness=0.):
        self.n_lam = n_lam
        self.n_state = n_state
        self.n_control = n_control

        self.lam = SX.sym('lam', self.n_lam)
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)

        self.theta = []
        self.lcp_theta = []
        self.dyn_theta = []
        if A is None:
            self.A = SX.sym('A', self.n_state, self.n_state)
            self.theta += [vec(self.A)]
            self.dyn_theta += [vec(self.A)]
        else:
            self.A = DM(A)

        if B is None:
            self.B = SX.sym('B', self.n_state, self.n_control)
            self.theta += [vec(self.B)]
            self.dyn_theta += [vec(self.B)]
        else:
            self.B = DM(B)

        if C is None:
            self.C = SX.sym('C', self.n_state, self.n_lam)
            self.theta += [vec(self.C)]
            self.dyn_theta += [vec(self.C)]
        else:
            self.C = DM(C)

        if D is None:
            self.D = SX.sym('D', self.n_lam, self.n_state)
            self.theta += [vec(self.D)]
            self.lcp_theta += [vec(self.D)]
        else:
            self.D = DM(D)

        if E is None:
            self.E = SX.sym('E', self.n_lam, self.n_control)
            self.theta += [vec(self.E)]
            self.lcp_theta += [vec(self.E)]
        else:
            self.E = DM(E)

        if G is None:
            self.G = SX.sym('G', self.n_lam, self.n_lam)
            self.theta += [vec(self.G)]
            self.lcp_theta += [vec(self.G)]
        else:
            self.G = DM(G)

        if H is None:
            self.H = SX.sym('H', self.n_lam, self.n_lam)
            self.theta += [vec(self.H)]
            self.lcp_theta += [vec(self.H)]
        else:
            self.H = DM(H)

        if lcp_offset is None:
            self.lcp_offset = SX.sym('lcp_offset', self.n_lam)
            self.theta += [vec(self.lcp_offset)]
            self.lcp_theta += [vec(self.lcp_offset)]

        else:
            self.lcp_offset = DM(lcp_offset)

        self.theta = vcat(self.theta)
        self.lcp_theta = vcat(self.lcp_theta)
        self.dyn_theta = vcat(self.dyn_theta)

        self.n_theta = self.theta.numel()
        self.n_lcp_theta = self.lcp_theta.numel()
        self.n_dyn_theta = self.dyn_theta.numel()

        self.F = stiffness * np.eye(self.n_lam) + self.G @ self.G.T + self.H - self.H.T
        self.A_fn = Function('A_fn', [self.dyn_theta], [self.A])
        self.B_fn = Function('B_fn', [self.dyn_theta], [self.B])
        self.C_fn = Function('C_fn', [self.dyn_theta], [self.C])
        self.D_fn = Function('D_fn', [self.lcp_theta], [self.D])
        self.E_fn = Function('E_fn', [self.lcp_theta], [self.E])
        self.G_fn = Function('G_fn', [self.lcp_theta], [self.G])
        self.H_fn = Function('H_fn', [self.lcp_theta], [self.H])
        self.F_fn = Function('F_fn', [self.lcp_theta], [self.F])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.lcp_theta], [self.lcp_offset])

        self.val_lcp_theta = 0.1 * np.random.randn(self.n_lcp_theta)
        self.val_dyn_theta = 0.1 * np.random.randn(self.n_dyn_theta)
        self.val_lcs_theta = None

    def differetiable(self):

        # lcp loss
        self.dist = self.D @ self.x + self.E @ self.u + self.F @ self.lam + self.lcp_offset
        data_lcp_theta = vertcat(self.x, self.u, self.lcp_theta)
        self.lcp_loss = dot(self.dist, self.lam)
        quadprog = {'x': self.lam, 'f': self.lcp_loss, 'g': self.dist, 'p': data_lcp_theta}
        opts = {'printLevel': 'none', }
        self.lcp_Solver = qpsol('lcp_Solver', 'qpoases', quadprog, opts)

        # define the dynamics loss
        self.x_next = SX.sym('x_next', self.n_state)
        self.dyn = self.A @ self.x + self.B @ self.u + self.C @ self.lam
        self.dyn_loss = dot(self.dyn - self.x_next, self.dyn - self.x_next)
        self.dyn_fn = Function('dyn_fn', [self.x, self.u, self.lam, self.dyn_theta], [self.dyn])

        # define the dynamics loss with respect to the lam variable
        self.dloss_dlam = jacobian(self.dyn_loss, self.lam)

        # define the gradient of lam with respect to lcp_theta
        g = diag(self.lam) @ self.dist
        dg_dlam = jacobian(g, self.lam)
        dg_dlcp = jacobian(g, self.lcp_theta)
        self.dlam_dlcp = -inv(dg_dlam) @ dg_dlcp
        self.dloss_dlcp = (self.dloss_dlam @ self.dlam_dlcp).T

        # assemble functions
        data = vertcat(self.x, self.u, self.x_next)
        self.dloss_dlcp_fn = Function('dloss_dlcp_fn', [data, self.lam, self.dyn_theta, self.lcp_theta],
                                      [self.dloss_dlcp])

    def compute_lambda(self, x_batch, u_batch):
        self.differetiable()
        # prepare the data
        batch_size = x_batch.shape[0]
        lcp_data_batch = np.hstack((x_batch, u_batch))
        lcp_theta_batch = np.tile(self.val_lcp_theta, (batch_size, 1))
        data_lcp_theta_batch = np.hstack((lcp_data_batch, lcp_theta_batch))

        # compute the lam solution
        sol_batch = self.lcp_Solver(lbx=0.0, lbg=0.0, p=data_lcp_theta_batch.T)
        lcp_loss_opt_batch = sol_batch['f'].full().flatten()
        lam_opt_batch = sol_batch['x'].full().T

        return lam_opt_batch, lcp_loss_opt_batch

    def dyn_regression(self, x_batch, u_batch, lam_opt_batch, x_next_batch):

        # prepare the data
        I = np.eye(self.n_state)
        kron_x = np.kron(x_batch, I)
        kron_u = np.kron(u_batch, I)
        kron_lam_opt = np.kron(lam_opt_batch, I)
        kron_x_next = x_next_batch.flatten()

        mat_A = np.hstack((kron_x, kron_u, kron_lam_opt))
        vec_b = kron_x_next

        # least square
        dyn_theta_opt, dyn_loss_opt, _, _ = numpy.linalg.lstsq(mat_A, vec_b, rcond=None)
        dyn_loss_opt = dyn_loss_opt.tolist()
        self.val_dyn_theta = dyn_theta_opt

        return dyn_loss_opt

    def gradient_step(self, x_batch, u_batch, x_next_batch, lam_opt_batch):
        batch_size = x_batch.shape[0]
        data_batch = np.hstack((x_batch, u_batch, x_next_batch))
        dyn_theta_opt_batch = np.tile(self.val_dyn_theta, (batch_size, 1))
        lcp_theta_batch = np.tile(self.val_lcp_theta, (batch_size, 1))

        # compute the gradient value
        dlcp = self.dloss_dlcp_fn(data_batch.T, lam_opt_batch.T, dyn_theta_opt_batch.T, lcp_theta_batch.T)
        dlcp_mean = dlcp.full().mean(axis=1)
        return dlcp_mean.flatten()

    def computeLCSMats(self, compact=True):
        A = self.A_fn(self.val_dyn_theta).full()
        B = self.B_fn(self.val_dyn_theta).full()
        C = self.C_fn(self.val_dyn_theta).full()

        D = self.D_fn(self.val_lcp_theta).full()
        E = self.E_fn(self.val_lcp_theta).full()
        F = self.F_fn(self.val_lcp_theta).full()
        G = self.G_fn(self.val_lcp_theta).full()
        H = self.H_fn(self.val_lcp_theta).full()
        lcp_offset = self.lcp_offset_fn(self.val_lcp_theta).full()

        lcs_theta = vertcat(vec(A), vec(B), vec(C),
                            vec(D), vec(E), vec(F),
                            vec(lcp_offset))

        if compact is True:
            return lcs_theta
        else:
            return {'A': A,
                    'B': B,
                    'C': C,
                    'D': D,
                    'E': E,
                    'F': F,
                    'G': G,
                    'H': H,
                    'lcp_offset': lcp_offset,
                    }

    # the following are utility functions

    def dyn_prediction(self, x_batch, u_batch):
        self.differetiable()
        # prepare the data
        batch_size = x_batch.shape[0]
        lcp_data_batch = np.hstack((x_batch, u_batch))
        lcp_theta_batch = np.tile(self.val_lcp_theta, (batch_size, 1))
        lcp_data_theta_batch = np.hstack((lcp_data_batch, lcp_theta_batch))

        # compute the lam solution
        sol_batch = self.lcp_Solver(lbx=0.0, lbg=0.0, p=lcp_data_theta_batch.T)
        lam_opt_batch = sol_batch['x'].full().T

        # compute the next state batch
        dyn_theta_op_batch = np.tile(self.val_dyn_theta, (batch_size, 1))
        x_next_batch = self.dyn_fn(x_batch.T, u_batch.T, lam_opt_batch.T, dyn_theta_op_batch.T).full().T

        return x_next_batch, lam_opt_batch

    def sim_dyn(self, init_x_batch, control_traj_batch):
        self.differetiable()

        batch_size = init_x_batch.shape[0]
        state_traj_batch = []
        lam_traj_batch = []
        for i in range(batch_size):
            init_x = init_x_batch[i]
            control_traj = control_traj_batch[i]
            state_traj = [init_x]
            lam_traj = []
            control_horizon = control_traj.shape[0]
            for t in range(control_horizon):
                curr_x = state_traj[-1]
                curr_u = control_traj[t]
                curr_lcp_data_theta = np.hstack((curr_x, curr_u, self.val_lcp_theta))
                curr_sol = self.lcp_Solver(lbx=0.0, lbg=0.0, p=curr_lcp_data_theta)
                curr_lam = curr_sol['x'].full().flatten()
                next_x = self.dyn_fn(curr_x, curr_u, curr_lam, self.val_dyn_theta).full().flatten()
                state_traj += [next_x]
                lam_traj += [curr_lam]
            state_traj = np.array(state_traj)
            lam_traj = np.array(lam_traj)
            state_traj_batch += [state_traj]
            lam_traj_batch += [lam_traj]

        return state_traj_batch, lam_traj_batch

    def dyn_step(self, x_batch, u_batch):
        self.differetiable()

        if type(x_batch) is np.ndarray:
            x_batch = [x_batch]
            u_batch = [u_batch]

        batch_size = len(x_batch)
        next_x_batch = []
        lam_batch = []
        for i in range(batch_size):
            x = x_batch[i]
            u = u_batch[i]
            lcp_data_theta = np.hstack((x, u, self.val_lcp_theta))

            # compute the lam value
            sol = self.lcp_Solver(lbx=0.0, lbg=0.0, p=lcp_data_theta)
            lam = sol['x'].full().flatten()
            # compute the next state
            next_x = self.dyn_fn(x, u, lam, self.val_dyn_theta).full().flatten()

            # store
            next_x_batch += [next_x]
            lam_batch += [lam]

        return next_x_batch, lam_batch

    def dynamics_step(self, curr_x, curr_u):
        self.differetiable()

        lcp_data_theta = np.hstack((curr_x, curr_u, self.val_lcp_theta))
        # compute the lam value
        sol = self.lcp_Solver(lbx=0.0, lbg=0.0, p=lcp_data_theta)
        curr_lam = sol['x'].full().flatten()
        # compute the next state
        next_x = self.dyn_fn(curr_x, curr_u, curr_lam, self.val_dyn_theta).full().flatten()

        return next_x, curr_lam


# learn a lcs model from the sampled data using regression algorithms
def LCSRegressionBuffer(lcs_learner, optimizier,
                        curr_control_traj_batch, curr_true_state_traj_batch,
                        prev_control_traj_batch, prev_true_state_traj_batch,
                        buffer_ratio=0.5,
                        minibatch_size=200, max_iter=5000, print_level=0):
    # converting the data form
    curr_batch_size = len(curr_control_traj_batch)
    curr_train_u_batch = []
    curr_train_x_batch = []
    curr_train_x_next_batch = []
    for i in range(curr_batch_size):
        curr_train_u_batch += [curr_control_traj_batch[i]]
        curr_train_x_batch += [curr_true_state_traj_batch[i][0:-1]]
        curr_train_x_next_batch += [curr_true_state_traj_batch[i][1:]]
    curr_train_u_batch = np.vstack(curr_train_u_batch)
    curr_train_x_next_batch = np.vstack(curr_train_x_next_batch)
    curr_train_x_batch = np.vstack(curr_train_x_batch)
    curr_train_data_size = curr_train_u_batch.shape[0]
    curr_minibatch_size = int((1 - buffer_ratio) * minibatch_size)

    # converting the data form
    prev_batch_size = len(prev_control_traj_batch)
    if prev_batch_size is not 0:
        prev_train_u_batch = []
        prev_train_x_batch = []
        prev_train_x_next_batch = []
        for i in range(prev_batch_size):
            prev_train_u_batch += [prev_control_traj_batch[i]]
            prev_train_x_batch += [prev_true_state_traj_batch[i][0:-1]]
            prev_train_x_next_batch += [prev_true_state_traj_batch[i][1:]]
        prev_train_u_batch = np.vstack(prev_train_u_batch)
        prev_train_x_next_batch = np.vstack(prev_train_x_next_batch)
        prev_train_x_batch = np.vstack(prev_train_x_batch)
        prev_train_data_size = prev_train_u_batch.shape[0]
        prev_minibatch_size = minibatch_size - curr_minibatch_size
    else:
        curr_minibatch_size = minibatch_size
        prev_minibatch_size = 0

    print('current_data points:', curr_minibatch_size, '| history data points:', prev_minibatch_size)

    for k in range(max_iter):
        # mini batch dataset for current training data set
        curr_shuffle_index = np.random.permutation(curr_train_data_size)[0:curr_minibatch_size]
        curr_x_minibatch = curr_train_x_batch[curr_shuffle_index]
        curr_u_minibatch = curr_train_u_batch[curr_shuffle_index]
        curr_x_next_minibatch = curr_train_x_next_batch[curr_shuffle_index]
        # mini batch dataset for the previous training data set
        if prev_batch_size is not 0:
            prev_shuffle_index = np.random.permutation(prev_train_data_size)[0:prev_minibatch_size]
            prev_x_minibatch = prev_train_x_batch[prev_shuffle_index]
            prev_u_minibatch = prev_train_u_batch[prev_shuffle_index]
            prev_x_next_minibatch = prev_train_x_next_batch[prev_shuffle_index]
            x_minibatch = np.vstack((curr_x_minibatch, prev_x_minibatch))
            u_minibatch = np.vstack((curr_u_minibatch, prev_u_minibatch))
            x_next_minibatch = np.vstack((curr_x_next_minibatch, prev_x_next_minibatch))
        else:
            x_minibatch = curr_x_minibatch
            u_minibatch = curr_u_minibatch
            x_next_minibatch = curr_x_next_minibatch

        # compute the lambda batch
        lam_opt_mini_batch, loss_opt_mini_batch = lcs_learner.compute_lambda(x_minibatch, u_minibatch)

        # regression for the dynamics
        dyn_loss_opt = lcs_learner.dyn_regression(x_minibatch, u_minibatch, lam_opt_mini_batch, x_next_minibatch)

        # compute the gradient
        dlcp_theta = lcs_learner.gradient_step(x_minibatch, u_minibatch, x_next_minibatch, lam_opt_mini_batch)

        # store and update
        lcs_learner.val_lcp_theta = optimizier.step(lcs_learner.val_lcp_theta, dlcp_theta)

        if print_level is not 0:
            if k % 100 == 0:
                # on the prediction using the current learned lcs
                pred_x_next_batch, pred_lam_batch = lcs_learner.dyn_prediction(curr_train_x_batch, curr_train_u_batch)

                # compute the prediction error
                error_x_next_batch = pred_x_next_batch - curr_train_x_next_batch
                relative_error = (
                        la.norm(error_x_next_batch, axis=1) / (
                        la.norm(curr_train_x_next_batch, axis=1) + 0.0001)).mean()

                print(
                    'lcs learning iter', k,
                    '| loss:', dyn_loss_opt,
                    '| grad:', norm_2(dlcp_theta),
                    '| PRE:', relative_error,
                )


# learn a lcs model from the sampled data using regression algorithms without previous
def LCSLearningRegression(lcs_learner, optimizier, control_traj_batch, true_state_traj_batch,
                          max_iter=5000, minibatch_size=100, print_level=0):
    # converting the data form
    batch_size = len(control_traj_batch)
    train_u_batch = []
    train_x_batch = []
    train_x_next_batch = []
    for i in range(batch_size):
        train_u_batch += [control_traj_batch[i]]
        train_x_batch += [true_state_traj_batch[i][0:-1]]
        train_x_next_batch += [true_state_traj_batch[i][1:]]
    train_u_batch = np.vstack(train_u_batch)
    train_x_next_batch = np.vstack(train_x_next_batch)
    train_x_batch = np.vstack(train_x_batch)

    # do the learning iteration
    train_data_size = train_u_batch.shape[0]

    for k in range(max_iter):
        # mini batch dataset
        shuffle_index = np.random.permutation(train_data_size)[0:minibatch_size]
        x_minibatch = train_x_batch[shuffle_index]
        u_minibatch = train_u_batch[shuffle_index]
        x_next_minibatch = train_x_next_batch[shuffle_index]

        # compute the lambda batch
        lam_opt_mini_batch, loss_opt_mini_batch = lcs_learner.compute_lambda(x_minibatch, u_minibatch)
        # regression for the dynamics
        dyn_loss_opt = lcs_learner.dyn_regression(x_minibatch, u_minibatch, lam_opt_mini_batch, x_next_minibatch)
        # compute the gradient
        dlcp_theta = lcs_learner.gradient_step(x_minibatch, u_minibatch, x_next_minibatch, lam_opt_mini_batch)

        # store and update
        lcs_learner.val_lcp_theta = optimizier.step(lcs_learner.val_lcp_theta, dlcp_theta)

        if print_level is not 0:
            if k % 100 == 0:
                # on the prediction using the current learned lcs
                pred_x_next_batch, pred_lam_batch = lcs_learner.dyn_prediction(train_x_batch, train_u_batch)

                # compute the prediction error
                error_x_next_batch = pred_x_next_batch - train_x_next_batch
                relative_error = (
                        la.norm(error_x_next_batch, axis=1) / (la.norm(train_x_next_batch, axis=1) + 0.0001)).mean()

                print(
                    'lcs learning iter', k,
                    '| loss:', dyn_loss_opt,
                    '| grad:', norm_2(dlcp_theta),
                    '| PRE:', relative_error,
                )


# evaluation object to evaluate the learned lcs model using a control cost function
class LCS_evaluation:
    def __init__(self, lcs_learner):
        self.name = 'lcs evaluation'

        # define the system variables
        self.n_state = lcs_learner.n_state
        self.n_control = lcs_learner.n_control
        self.n_lam = lcs_learner.n_lam

        # define the system variables
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)
        self.lam = SX.sym('lam', self.n_lam)

    def setCostFunction(self, Q, R, QN, control_horizon):
        self.Q = DM(Q)
        self.R = DM(R)
        self.QN = DM(QN)
        self.control_horizon = control_horizon

        # define the control cost function
        self.path_cost = dot(self.x, self.Q @ self.x) + dot(self.u, self.R @ self.u)
        self.final_cost = dot(self.x, self.QN @ self.x)

        self.path_cost_fn = Function('path_cost_fn', [self.x, self.u], [self.path_cost])
        self.final_cost_fn = Function('final_cost_fn', [self.x], [self.final_cost])

    def computeCost(self, control_traj_batch, state_traj_batch):
        cost_batch = []
        batch_size = len(control_traj_batch)
        for i in range(batch_size):
            u_traj = control_traj_batch[i]
            x_traj = state_traj_batch[i]

            cost = 0.0
            for t in range(self.control_horizon):
                curr_x = x_traj[t]
                curr_u = u_traj[t]
                cost += self.path_cost_fn(curr_x, curr_u)

            cost += self.final_cost_fn(x_traj[-1])

            cost_batch += [cost]

        return cost_batch

    def differentiable(self):
        self.A = SX.sym('A', self.n_state, self.n_state)
        self.B = SX.sym('B', self.n_state, self.n_control)
        self.C = SX.sym('C', self.n_state, self.n_lam)

        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.F = SX.sym('F', self.n_lam, self.n_lam)
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.lcs_theta = vertcat(vec(self.A), vec(self.B), vec(self.C),
                                 vec(self.D), vec(self.E), vec(self.F),
                                 vec(self.lcp_offset))

        # define the dynamics
        self.f = self.A @ self.x + self.B @ self.u + self.C @ self.lam

        # define the gradient of lam with respect to lcp_theta
        self.dist = self.D @ self.x + self.E @ self.u + self.F @ self.lam + self.lcp_offset
        g = diag(self.lam) @ self.dist
        dg_dlam = jacobian(g, self.lam)
        dg_dx = jacobian(g, self.x)
        dg_du = jacobian(g, self.u)
        dlam_dx = -inv(dg_dlam) @ dg_dx
        dlam_du = -inv(dg_dlam) @ dg_du

        # differentiate
        df_dx = jacobian(self.f, self.x) + jacobian(self.f, self.lam) @ dlam_dx
        df_du = jacobian(self.f, self.u) + jacobian(self.f, self.lam) @ dlam_du

        self.dfdx_fn = Function('dfdx_fn', [self.x, self.u, self.lam, self.lcs_theta], [df_dx])
        self.dfdu_fn = Function('dfdx_fn', [self.x, self.u, self.lam, self.lcs_theta], [df_du])

        # compute the gradient of the cost function
        self.dcdx = jacobian(self.path_cost, self.x).T
        self.dcdu = jacobian(self.path_cost, self.u).T
        self.dhdx = jacobian(self.final_cost, self.x).T

        # establish the functions for the above gradient
        self.dcdx_fn = Function('dcdx_fn', [self.x, self.u], [self.dcdx])
        self.dcdu_fn = Function('dcdu_fn', [self.x, self.u], [self.dcdu])
        self.dhdx_fn = Function('dhdx_fn', [self.x], [self.dhdx])

    # I use the single shooting method to update the control sequence but it does not work
    def Evaluate(self, lcs_learner, init_state_batch, control_traj_batch, true_state_traj_batch):
        lcs_theta = lcs_learner.computeLCSMats()

        # compute the state batch_trajectory
        pred_state_traj_batch, pred_lam_traj_batch = lcs_learner.sim_dyn(init_state_batch, control_traj_batch)

        # # debug for plotting
        # plt.plot(true_state_traj_batch[0])
        # plt.plot(pred_state_traj_batch[0])
        # plt.show()

        # control_update trajectory
        control_update_traj_batch = []
        model_cost_batch = []
        true_sys_cost_batch = []

        batch_size = init_state_batch.shape[0]
        for i in range(batch_size):
            u_traj = control_traj_batch[i]
            pred_x_traj = pred_state_traj_batch[i]
            pred_lam_traj = pred_lam_traj_batch[i]
            true_x_traj = true_state_traj_batch[i]
            control_horizon = u_traj.shape[0]

            # compute the recover matrix (see Jin et al. IJRR for details)
            curr_x = pred_x_traj[0]
            curr_u = u_traj[0]
            curr_lam = pred_lam_traj[0]
            next_x = pred_x_traj[1]
            next_u = u_traj[1]
            next_lam = pred_lam_traj[1]

            curr_dfdu = self.dfdu_fn(curr_x, curr_u, curr_lam, lcs_theta)
            curr_dcdu = self.dcdu_fn(curr_x, curr_u)

            next_dfdx = self.dfdx_fn(next_x, next_u, next_lam, lcs_theta)
            next_dcdx = self.dcdx_fn(next_x, next_u)

            H1 = curr_dfdu.T @ next_dcdx + curr_dcdu
            H2 = curr_dfdu.T @ next_dfdx.T

            model_cost = self.path_cost_fn(curr_x, curr_u)
            true_sys_cost = self.path_cost_fn(true_x_traj[0], curr_u)
            for t in range(1, control_horizon - 1):
                curr_x = pred_x_traj[t]
                curr_u = u_traj[t]
                next_x = pred_x_traj[t + 1]
                next_u = u_traj[t + 1]

                curr_dfdu = self.dfdu_fn(curr_x, curr_u, curr_lam, lcs_theta)
                curr_dcdu = self.dcdu_fn(curr_x, curr_u)

                next_dfdx = self.dfdx_fn(next_x, next_u, next_lam, lcs_theta)
                next_dcdx = self.dcdx_fn(next_x, next_u)

                H1 = vertcat(H1 + H2 @ next_dcdx,
                             curr_dfdu.T @ next_dcdx + curr_dcdu)
                H2 = vertcat(H2 @ next_dfdx.T,
                             curr_dfdu.T @ next_dfdx.T)

                model_cost += self.path_cost_fn(curr_x, curr_u)
                true_sys_cost += self.path_cost_fn(true_x_traj[t], curr_u)

            curr_x = pred_x_traj[control_horizon - 1]
            curr_u = u_traj[control_horizon - 1]
            curr_dfdu = self.dfdu_fn(curr_x, curr_u, curr_lam, lcs_theta)
            curr_dcdu = self.dcdu_fn(curr_x, curr_u)
            next_x = pred_x_traj[control_horizon]
            next_dhdx = self.dhdx_fn(next_x)
            H1 = vertcat(H1 + H2 @ next_dhdx,
                         curr_dfdu.T @ next_dhdx + curr_dcdu).full().flatten()

            model_cost += self.path_cost_fn(curr_x, curr_u)
            true_sys_cost += self.path_cost_fn(true_x_traj[control_horizon - 1], curr_u)
            model_cost += self.final_cost_fn(next_x)
            true_sys_cost += self.final_cost_fn(true_x_traj[control_horizon])

            control_update_traj_batch += [H1.reshape((-1, self.n_control))]
            model_cost_batch += [model_cost]
            true_sys_cost_batch += [true_sys_cost]

        return control_update_traj_batch, model_cost_batch, true_sys_cost_batch

    def Update(self, control_traj_batch, control_update_batch, step_size=1e-4):

        batch_size = len(control_traj_batch)
        new_control_traj_batch = []
        for i in range(batch_size):
            control_traj = control_traj_batch[i]
            update_traj = control_update_batch[i]
            new_control_traj = control_traj - step_size * update_traj
            new_control_traj_batch += [new_control_traj]

        return new_control_traj_batch

    # Then I think about to update the control sequence using multiple-shooting (this can be further improved)
    def initializeUpdater(self, proximity_epsilon=1e-3):

        self.differentiable()

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        p = []

        # "Lift" initial conditions
        Xk = casadi.SX.sym('X0', self.n_state)
        w += [Xk]
        lbw += np.zeros(self.n_state).tolist()
        ubw += np.zeros(self.n_state).tolist()
        w0 += np.zeros(self.n_state).tolist()

        # formulate the NLP
        for k in range(self.control_horizon):
            # New NLP variable for the control
            Uk = casadi.SX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0.]

            # new NLP variable for the complementarity variable
            Lamk = casadi.SX.sym('lam' + str(k), self.n_lam)
            w += [Lamk]
            lbw += self.n_lam * [0.]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0.]

            # Add complementarity equation
            g += [self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset]
            lbg += self.n_lam * [0.]
            ubg += self.n_lam * [inf]

            g += [casadi.dot(self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset, Lamk)]
            lbg += [0.]
            ubg += [0.]

            # Integrate till the end of the interval
            Xnext = self.A @ Xk + self.B @ Uk + self.C @ Lamk

            # compute the current cost
            Uk_ref = SX.sym('Uk_ref' + str(k), self.n_control)
            p += [Uk_ref]

            Ck = dot(Xk, self.Q @ Xk) + dot(Uk, self.R @ Uk) + dot(Uk - Uk_ref, Uk - Uk_ref) / proximity_epsilon
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.SX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0.]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0.]
            ubg += self.n_state * [0.]

        # Add the final cost
        J = J + dot(Xk, self.QN @ Xk)

        # Create an NLP solver and solve
        p += [self.lcs_theta]
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g), 'p': vertcat(*p)}
        self.oc_solver = casadi.nlpsol('solver', 'ipopt', prob, opts)

        self.lbw = DM(lbw)
        self.ubw = DM(ubw)
        self.lbg = DM(lbg)
        self.ubg = DM(ubg)
        self.w0 = DM(w0)

        # this is  warming start for acceleration of ipopt solver
        self.warm_start = []

    def EvaluateMS(self, lcs_learner, init_state_batch, control_traj_batch):

        if not hasattr(self, 'solver'):
            self.initializeUpdater()

        lcs_theta = lcs_learner.computeLCSMats()

        # compute the state batch_trajectory
        pred_state_traj_batch, pred_lam_traj_batch = lcs_learner.sim_dyn(init_state_batch, control_traj_batch)
        # # debug for plotting
        # plt.plot(true_state_traj_batch[0])
        # plt.plot(pred_state_traj_batch[0])
        # plt.show()

        # ===============================================
        # do the update of the control sequence
        batch_size = len(control_traj_batch)
        updated_control_traj_batch = []
        updated_pred_state_traj_batch = []
        updated_pred_lam_traj_batch = []
        # this is for warm start for the next iteration
        w_opt_batch = []
        for i in range(batch_size):
            u_traj = control_traj_batch[i]
            init_state = init_state_batch[i]

            # step upt the oc solver
            lbw = self.lbw
            ubw = self.ubw
            lbw[0:self.n_state] = DM(init_state)
            ubw[0:self.n_state] = DM(init_state)
            oc_para = vertcat(u_traj.flatten(), lcs_theta)
            # warm start
            if not self.warm_start:
                init_w = self.w0
                init_w[0:self.n_state] = DM(init_state)
            else:
                init_w = self.warm_start[i]
                init_w[0:self.n_state] = DM(init_state)

            # Solve the NLP
            sol = self.oc_solver(x0=init_w, lbx=lbw, ubx=ubw, lbg=self.lbg, ubg=self.ubg, p=oc_para)
            w_opt = sol['x']
            w_opt_batch += [w_opt]

            # extract the optimal control and state
            sol_traj = w_opt[0:self.control_horizon * (self.n_state + self.n_control + self.n_lam)].reshape(
                (self.n_state + self.n_control + self.n_lam, -1))
            x_traj = casadi.horzcat(sol_traj[0:self.n_state, :],
                                    w_opt[self.control_horizon * (self.n_state + self.n_control + self.n_lam):]).T
            u_traj = sol_traj[self.n_state:self.n_state + self.n_control, :].T
            lam_traj = sol_traj[self.n_state + self.n_control:, :].T

            updated_control_traj_batch += [u_traj.full()]
            updated_pred_state_traj_batch += [x_traj.full()]
            updated_pred_lam_traj_batch += [lam_traj.full()]

        self.warm_start = w_opt_batch

        return updated_control_traj_batch


# compute the gradient of the control input using the recovery matrix (Jin. et al. IJRR)


# evaluation object to evaluate the learned lcs model using a control cost function
# random_initial condition
class LCS_evaluation2:
    def __init__(self, lcs_learner):
        self.name = 'lcs evaluation'

        # define the system variables
        self.n_state = lcs_learner.n_state
        self.n_control = lcs_learner.n_control
        self.n_lam = lcs_learner.n_lam

        # define the system variables
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)
        self.lam = SX.sym('lam', self.n_lam)

    def setCostFunction(self, Q, R, QN, control_horizon):
        self.Q = DM(Q)
        self.R = DM(R)
        self.QN = DM(QN)
        self.control_horizon = control_horizon

        # define the control cost function
        self.path_cost = dot(self.x, self.Q @ self.x) + dot(self.u, self.R @ self.u)
        self.final_cost = dot(self.x, self.QN @ self.x)

        self.path_cost_fn = Function('path_cost_fn', [self.x, self.u], [self.path_cost])
        self.final_cost_fn = Function('final_cost_fn', [self.x], [self.final_cost])

    def computeCost(self, control_traj_batch, state_traj_batch):
        cost_batch = []
        batch_size = len(control_traj_batch)
        for i in range(batch_size):
            u_traj = control_traj_batch[i]
            x_traj = state_traj_batch[i]

            cost = 0.0
            for t in range(self.control_horizon):
                curr_x = x_traj[t]
                curr_u = u_traj[t]
                cost += self.path_cost_fn(curr_x, curr_u)

            cost += self.final_cost_fn(x_traj[-1])

            cost_batch += [cost]

        return cost_batch

    def differentiable(self):
        self.A = SX.sym('A', self.n_state, self.n_state)
        self.B = SX.sym('B', self.n_state, self.n_control)
        self.C = SX.sym('C', self.n_state, self.n_lam)

        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.F = SX.sym('F', self.n_lam, self.n_lam)
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.lcs_theta = vertcat(vec(self.A), vec(self.B), vec(self.C),
                                 vec(self.D), vec(self.E), vec(self.F),
                                 vec(self.lcp_offset))

        # define the dynamics
        self.f = self.A @ self.x + self.B @ self.u + self.C @ self.lam

        # define the gradient of lam with respect to lcp_theta
        self.dist = self.D @ self.x + self.E @ self.u + self.F @ self.lam + self.lcp_offset
        g = diag(self.lam) @ self.dist
        dg_dlam = jacobian(g, self.lam)
        dg_dx = jacobian(g, self.x)
        dg_du = jacobian(g, self.u)
        dlam_dx = -inv(dg_dlam) @ dg_dx
        dlam_du = -inv(dg_dlam) @ dg_du

        # differentiate
        df_dx = jacobian(self.f, self.x) + jacobian(self.f, self.lam) @ dlam_dx
        df_du = jacobian(self.f, self.u) + jacobian(self.f, self.lam) @ dlam_du

        self.dfdx_fn = Function('dfdx_fn', [self.x, self.u, self.lam, self.lcs_theta], [df_dx])
        self.dfdu_fn = Function('dfdx_fn', [self.x, self.u, self.lam, self.lcs_theta], [df_du])

        # compute the gradient of the cost function
        self.dcdx = jacobian(self.path_cost, self.x).T
        self.dcdu = jacobian(self.path_cost, self.u).T
        self.dhdx = jacobian(self.final_cost, self.x).T

        # establish the functions for the above gradient
        self.dcdx_fn = Function('dcdx_fn', [self.x, self.u], [self.dcdx])
        self.dcdu_fn = Function('dcdu_fn', [self.x, self.u], [self.dcdu])
        self.dhdx_fn = Function('dhdx_fn', [self.x], [self.dhdx])

    # Then I think about to update the control sequence using multiple-shooting (this can be further improved)
    def initializeUpdater(self, proximity_epsilon=1e-2):

        self.differentiable()

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        p = []

        # "Lift" initial conditions
        Xk = casadi.SX.sym('X0', self.n_state)
        w += [Xk]
        lbw += np.zeros(self.n_state).tolist()
        ubw += np.zeros(self.n_state).tolist()
        w0 += np.zeros(self.n_state).tolist()

        # formulate the NLP
        for k in range(self.control_horizon):
            # New NLP variable for the control
            Uk = casadi.SX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0.]

            # new NLP variable for the complementarity variable
            Lamk = casadi.SX.sym('lam' + str(k), self.n_lam)
            w += [Lamk]
            lbw += self.n_lam * [0.]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0.]

            # Add complementarity equation
            g += [self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset]
            lbg += self.n_lam * [0.]
            ubg += self.n_lam * [inf]

            g += [casadi.dot(self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset, Lamk)]
            lbg += [0.]
            ubg += [0.]

            # Integrate till the end of the interval
            Xnext = self.A @ Xk + self.B @ Uk + self.C @ Lamk

            # compute the current cost
            Uk_ref = SX.sym('Uk_ref' + str(k), self.n_control)
            p += [Uk_ref]

            Ck = dot(Xk, self.Q @ Xk) + dot(Uk, self.R @ Uk) + dot(Uk - Uk_ref, Uk - Uk_ref) / proximity_epsilon
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.SX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0.]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0.]
            ubg += self.n_state * [0.]

        # Add the final cost
        J = J + dot(Xk, self.QN @ Xk)

        # Create an NLP solver and solve
        p += [self.lcs_theta]
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g), 'p': vertcat(*p)}
        self.oc_solver = casadi.nlpsol('solver', 'ipopt', prob, opts)

        self.lbw = DM(lbw)
        self.ubw = DM(ubw)
        self.lbg = DM(lbg)
        self.ubg = DM(ubg)
        self.w0 = DM(w0)

        self.prev_init_state_batch = []

    def EvaluateMS(self, lcs_learner, init_state_batch, control_traj_batch):

        if not hasattr(self, 'oc_solver'):
            self.initializeUpdater()

        lcs_theta = lcs_learner.computeLCSMats()

        # compute the state batch_trajectory
        pred_state_traj_batch, pred_lam_traj_batch = lcs_learner.sim_dyn(init_state_batch, control_traj_batch)
        # # debug for plotting
        # plt.plot(true_state_traj_batch[0])
        # plt.plot(pred_state_traj_batch[0])
        # plt.show()

        # ===============================================
        # do the update of the control sequence
        batch_size = len(control_traj_batch)
        updated_control_traj_batch = []
        updated_pred_state_traj_batch = []
        updated_pred_lam_traj_batch = []
        # this is for warm start for the next iteration
        curr_init_state_batch = []
        for i in range(batch_size):

            init_state = init_state_batch[i]

            # step upt the oc solver
            lbw = self.lbw
            ubw = self.ubw
            lbw[0:self.n_state] = DM(init_state)
            ubw[0:self.n_state] = DM(init_state)
            init_w = self.w0
            init_w[0:self.n_state] = DM(init_state)

            # ========================= no action
            # u_traj = control_traj_batch[i]
            # oc_para = vertcat(u_traj.flatten(), lcs_theta)

            # ========================= do the correspondence
            # if len(self.prev_init_state_batch) == 0:
            #     u_traj = control_traj_batch[i]
            #     oc_para = vertcat(u_traj.flatten(), lcs_theta)
            # else:
            #     # check the current initial is close to which previous initial condition
            #     close_index = find_closest(self.prev_init_state_batch, init_state)
            #     u_traj = control_traj_batch[close_index]
            #     oc_para = vertcat(u_traj.flatten(), lcs_theta)
            # curr_init_state_batch += [init_state]

            # ========================= do the meaning
            sum_u_traj = 0
            for j in range(batch_size):
                sum_u_traj += control_traj_batch[i]
            mean_u_traj = sum_u_traj / batch_size
            oc_para = vertcat(mean_u_traj.flatten(), lcs_theta)

            # Solve the NLP
            sol = self.oc_solver(x0=init_w, lbx=lbw, ubx=ubw, lbg=self.lbg, ubg=self.ubg, p=oc_para)
            w_opt = sol['x']

            # extract the optimal control and state
            sol_traj = w_opt[0:self.control_horizon * (self.n_state + self.n_control + self.n_lam)].reshape(
                (self.n_state + self.n_control + self.n_lam, -1))
            x_traj = casadi.horzcat(sol_traj[0:self.n_state, :],
                                    w_opt[self.control_horizon * (self.n_state + self.n_control + self.n_lam):]).T
            u_traj = sol_traj[self.n_state:self.n_state + self.n_control, :].T
            lam_traj = sol_traj[self.n_state + self.n_control:, :].T

            updated_control_traj_batch += [u_traj.full()]
            updated_pred_state_traj_batch += [x_traj.full()]
            updated_pred_lam_traj_batch += [lam_traj.full()]

        self.prev_init_state_batch = curr_init_state_batch

        return updated_control_traj_batch


# compute the gradient of the control input using the recovery matrix (Jin. et al. IJRR)


# evaluation object to evaluate the learned lcs model using a control cost function
# random_initial condition
class MPC_Controller:

    def __init__(self, lcs_learner):
        self.name = 'lcs optimal control'

        # define the system variables
        self.n_state = lcs_learner.n_state
        self.n_control = lcs_learner.n_control
        self.n_lam = lcs_learner.n_lam

        # define the system variables
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)
        self.lam = SX.sym('lam', self.n_lam)

        # define the system matrices
        self.A = SX.sym('A', self.n_state, self.n_state)
        self.B = SX.sym('B', self.n_state, self.n_control)
        self.C = SX.sym('C', self.n_state, self.n_lam)

        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.F = SX.sym('F', self.n_lam, self.n_lam)
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.lcs_theta = vertcat(vec(self.A), vec(self.B), vec(self.C),
                                 vec(self.D), vec(self.E), vec(self.F),
                                 vec(self.lcp_offset))

        # define the dynamics
        self.f = self.A @ self.x + self.B @ self.u + self.C @ self.lam

    def set_cost_function(self, Q, R, QN):
        self.Q = DM(Q)
        self.R = DM(R)
        self.QN = DM(QN)

        # define the control cost function
        self.path_cost = dot(self.x, self.Q @ self.x) + dot(self.u, self.R @ self.u)
        self.final_cost = dot(self.x, self.QN @ self.x)
        self.path_cost_fn = Function('path_cost_fn', [self.x, self.u], [self.path_cost])
        self.final_cost_fn = Function('final_cost_fn', [self.x], [self.final_cost])

    def compute_cost(self, control_traj_batch, state_traj_batch):
        cost_batch = []
        batch_size = len(control_traj_batch)
        for i in range(batch_size):
            u_traj = control_traj_batch[i]
            x_traj = state_traj_batch[i]

            cost = 0.0
            control_horizon = u_traj.shape[0]
            for t in range(control_horizon):
                curr_x = x_traj[t]
                curr_u = u_traj[t]
                cost += self.path_cost_fn(curr_x, curr_u)

            cost += self.final_cost_fn(x_traj[-1])

            cost_batch += [cost]

        return cost_batch

    def initialize_mpc(self, mpc_horizon):

        self.mpc_horizon = mpc_horizon

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = casadi.SX.sym('X0', self.n_state)
        w += [Xk]
        lbw += np.zeros(self.n_state).tolist()
        ubw += np.zeros(self.n_state).tolist()
        w0 += np.zeros(self.n_state).tolist()

        # formulate the NLP
        for k in range(self.mpc_horizon):
            # New NLP variable for the control
            Uk = casadi.SX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0.]

            # new NLP variable for the complementarity variable
            Lamk = casadi.SX.sym('lam' + str(k), self.n_lam)
            w += [Lamk]
            lbw += self.n_lam * [0.]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0.]

            # Add complementarity equation
            g += [self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset]
            lbg += self.n_lam * [0.]
            ubg += self.n_lam * [inf]

            g += [casadi.dot(self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset, Lamk)]
            lbg += [0.]
            ubg += [0.]

            # Integrate till the end of the interval
            Xnext = self.A @ Xk + self.B @ Uk + self.C @ Lamk

            Ck = dot(Xk, self.Q @ Xk) + dot(Uk, self.R @ Uk)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.SX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0.]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0.]
            ubg += self.n_state * [0.]

        # Add the final cost
        J = J + dot(Xk, self.QN @ Xk)

        # Create an NLP solver and solve
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g), 'p': self.lcs_theta}
        self.oc_solver = casadi.nlpsol('solver', 'ipopt', prob, opts)

        self.lbw = DM(lbw)
        self.ubw = DM(ubw)
        self.lbg = DM(lbg)
        self.ubg = DM(ubg)
        self.w0 = DM(w0)

    def oc(self, lcs_theta, init_state):
        # set the optimal control bounds
        lbw = self.lbw
        ubw = self.ubw
        init_w = self.w0
        lbw[0:self.n_state] = DM(init_state)
        ubw[0:self.n_state] = DM(init_state)
        init_w[0:self.n_state] = DM(init_state)

        # set the optimal control parameter
        oc_parameters = DM(lcs_theta)
        sol = self.oc_solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=oc_parameters)
        w_opt = sol['x']

        # extract the optimal control and state
        sol_traj = w_opt[0:self.mpc_horizon * (self.n_state + self.n_control + self.n_lam)].reshape(
            (self.n_state + self.n_control + self.n_lam, -1))
        x_traj = casadi.horzcat(sol_traj[0:self.n_state, :],
                                w_opt[self.mpc_horizon * (self.n_state + self.n_control + self.n_lam):]).T.full()
        u_traj = sol_traj[self.n_state:self.n_state + self.n_control, :].T.full()
        lam_traj = sol_traj[self.n_state + self.n_control:, :].T.full()

        sol = {'state_traj_opt': x_traj,
               'control_traj_opt': u_traj,
               'lam_traj_opt': lam_traj
               }

        return sol

    def mpc_step(self, lcs_learner, curr_state, lcs_theta=None):

        # take out the current lcs system parameter
        if lcs_theta is None:
            lcs_theta = lcs_learner.computeLCSMats()

        # set the optimal control bounds
        lbw = self.lbw
        ubw = self.ubw
        init_w = self.w0
        lbw[0:self.n_state] = DM(curr_state)
        ubw[0:self.n_state] = DM(curr_state)
        init_w[0:self.n_state] = DM(curr_state)

        # set the optimal control parameter
        oc_parameters = DM(lcs_theta)
        sol = self.oc_solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=oc_parameters)
        w_opt = sol['x']
        self.w0 = w_opt

        curr_control = w_opt[self.n_state:self.n_state + self.n_control].full().flatten()

        return curr_control


# compute the gradient of the control input using the recovery matrix (Jin. et al. IJRR)


# evaluation object to evaluate the learned lcs model using a control cost function
# random_initial condition
class MPC_Controller_KL:

    def __init__(self, lcs_learner):
        self.name = 'lcs optimal control'

        # define the system variables
        self.n_state = lcs_learner.n_state
        self.n_control = lcs_learner.n_control
        self.n_lam = lcs_learner.n_lam

        # define the system variables
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)
        self.lam = SX.sym('lam', self.n_lam)

        # define the system matrices
        self.A = SX.sym('A', self.n_state, self.n_state)
        self.B = SX.sym('B', self.n_state, self.n_control)
        self.C = SX.sym('C', self.n_state, self.n_lam)

        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.F = SX.sym('F', self.n_lam, self.n_lam)
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.lcs_theta = vertcat(vec(self.A), vec(self.B), vec(self.C),
                                 vec(self.D), vec(self.E), vec(self.F),
                                 vec(self.lcp_offset))

        # define the dynamics
        self.f = self.A @ self.x + self.B @ self.u + self.C @ self.lam

    def set_cost_function(self, Q, R, QN):
        self.Q = DM(Q)
        self.R = DM(R)
        self.QN = DM(QN)

        # define the control cost function
        self.path_cost = dot(self.x, self.Q @ self.x) + dot(self.u, self.R @ self.u)
        self.final_cost = dot(self.x, self.QN @ self.x)
        self.path_cost_fn = Function('path_cost_fn', [self.x, self.u], [self.path_cost])
        self.final_cost_fn = Function('final_cost_fn', [self.x], [self.final_cost])

    def compute_cost(self, control_traj_batch, state_traj_batch):
        cost_batch = []
        batch_size = len(control_traj_batch)
        for i in range(batch_size):
            u_traj = control_traj_batch[i]
            x_traj = state_traj_batch[i]

            cost = 0.0
            control_horizon = u_traj.shape[0]
            for t in range(control_horizon):
                curr_x = x_traj[t]
                curr_u = u_traj[t]
                cost += self.path_cost_fn(curr_x, curr_u)

            cost += self.final_cost_fn(x_traj[-1])

            cost_batch += [cost]

        return cost_batch

    def initialize_mpc_KL(self, mpc_horizon):

        self.mpc_horizon = mpc_horizon

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        U_ref = SX.sym('U_ref', self.n_control)
        X_ref = SX.sym('X_ref', self.n_state)
        kl_epsilon = SX.sym('kl_epsilon')
        p = [kl_epsilon, X_ref, U_ref]

        # "Lift" initial conditions
        Xk = casadi.SX.sym('X0', self.n_state)
        w += [Xk]
        lbw += np.zeros(self.n_state).tolist()
        ubw += np.zeros(self.n_state).tolist()
        w0 += np.zeros(self.n_state).tolist()

        # formulate the NLP
        for k in range(self.mpc_horizon):
            # New NLP variable for the control
            Uk = casadi.SX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0.]

            # new NLP variable for the complementarity variable
            Lamk = casadi.SX.sym('lam' + str(k), self.n_lam)
            w += [Lamk]
            lbw += self.n_lam * [0.]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0.]

            # Add complementarity equation
            g += [self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset]
            lbg += self.n_lam * [0.]
            ubg += self.n_lam * [inf]

            g += [casadi.dot(self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset, Lamk)]
            lbg += [0.]
            ubg += [0.]

            # Integrate till the end of the interval
            Xnext = self.A @ Xk + self.B @ Uk + self.C @ Lamk

            # compute the cost function
            Kl_dist = dot(Xk - X_ref, Xk - X_ref) + dot(Uk - U_ref, Uk - U_ref)
            Ck = dot(Xk, self.Q @ Xk) + dot(Uk, self.R @ Uk) + kl_epsilon * Kl_dist
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.SX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0.]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0.]
            ubg += self.n_state * [0.]

        # Add the final cost
        J = J + dot(Xk, self.QN @ Xk)

        # Create an NLP solver and solve
        p += [self.lcs_theta]
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g), 'p': vertcat(*p)}
        self.oc_solver = casadi.nlpsol('solver', 'ipopt', prob, opts)

        self.lbw = DM(lbw)
        self.ubw = DM(ubw)
        self.lbg = DM(lbg)
        self.ubg = DM(ubg)
        self.w0 = DM(w0)

    def mpc_step_kl(self, lcs_theta, curr_state, kl_epsilon, x_ref, u_ref):

        # set the optimal control bounds
        lbw = self.lbw
        ubw = self.ubw
        init_w = self.w0
        lbw[0:self.n_state] = DM(curr_state)
        ubw[0:self.n_state] = DM(curr_state)
        init_w[0:self.n_state] = DM(curr_state)

        # set the optimal control parameter
        oc_parameters = vertcat(kl_epsilon,  x_ref, u_ref, DM(lcs_theta))

        # print(oc_parameters.shape)
        # print(kl_epsilon)
        # print(x_ref.shape)
        # print(u_ref.shape)
        # print(lcs_theta.shape)


        sol = self.oc_solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=oc_parameters)
        w_opt = sol['x']
        self.w0 = w_opt

        curr_control = w_opt[self.n_state:self.n_state + self.n_control].full().flatten()

        return curr_control


class MPC_Controller_KL2:

    def __init__(self, lcs_learner):
        self.name = 'lcs optimal control'

        # define the system variables
        self.n_state = lcs_learner.n_state
        self.n_control = lcs_learner.n_control
        self.n_lam = lcs_learner.n_lam

        # define the system variables
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)
        self.lam = SX.sym('lam', self.n_lam)

        # define the system matrices
        self.A = SX.sym('A', self.n_state, self.n_state)
        self.B = SX.sym('B', self.n_state, self.n_control)
        self.C = SX.sym('C', self.n_state, self.n_lam)

        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.F = SX.sym('F', self.n_lam, self.n_lam)
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.lcs_theta = vertcat(vec(self.A), vec(self.B), vec(self.C),
                                 vec(self.D), vec(self.E), vec(self.F),
                                 vec(self.lcp_offset))

        # define the dynamics
        self.f = self.A @ self.x + self.B @ self.u + self.C @ self.lam

    def set_cost_function(self, Q, R, QN):
        self.Q = DM(Q)
        self.R = DM(R)
        self.QN = DM(QN)

        # define the control cost function
        self.path_cost = dot(self.x, self.Q @ self.x) + dot(self.u, self.R @ self.u)
        self.final_cost = dot(self.x, self.QN @ self.x)
        self.path_cost_fn = Function('path_cost_fn', [self.x, self.u], [self.path_cost])
        self.final_cost_fn = Function('final_cost_fn', [self.x], [self.final_cost])

    def compute_cost(self, control_traj_batch, state_traj_batch):
        cost_batch = []
        batch_size = len(control_traj_batch)
        for i in range(batch_size):
            u_traj = control_traj_batch[i]
            x_traj = state_traj_batch[i]

            cost = 0.0
            control_horizon = u_traj.shape[0]
            for t in range(control_horizon):
                curr_x = x_traj[t]
                curr_u = u_traj[t]
                cost += self.path_cost_fn(curr_x, curr_u)

            cost += self.final_cost_fn(x_traj[-1])

            cost_batch += [cost]

        return cost_batch

    def initialize_mpc_KL(self, mpc_horizon):

        self.mpc_horizon = mpc_horizon

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        U_ref = SX.sym('U_ref', self.n_control)
        kl_epsilon = SX.sym('kl_epsilon')
        p = [kl_epsilon, U_ref]

        # "Lift" initial conditions
        Xk = casadi.SX.sym('X0', self.n_state)
        w += [Xk]
        lbw += np.zeros(self.n_state).tolist()
        ubw += np.zeros(self.n_state).tolist()
        w0 += np.zeros(self.n_state).tolist()

        # formulate the NLP
        for k in range(self.mpc_horizon):
            # New NLP variable for the control
            Uk = casadi.SX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0.]

            # new NLP variable for the complementarity variable
            Lamk = casadi.SX.sym('lam' + str(k), self.n_lam)
            w += [Lamk]
            lbw += self.n_lam * [0.]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0.]

            # Add complementarity equation
            g += [self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset]
            lbg += self.n_lam * [0.]
            ubg += self.n_lam * [inf]

            g += [casadi.dot(self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset, Lamk)]
            lbg += [0.]
            ubg += [0.]

            # Integrate till the end of the interval
            Xnext = self.A @ Xk + self.B @ Uk + self.C @ Lamk

            # compute the cost function
            Kl_dist = dot(Uk - U_ref, Uk - U_ref)
            Ck = dot(Xk, self.Q @ Xk) + dot(Uk, self.R @ Uk) + kl_epsilon * Kl_dist
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.SX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0.]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0.]
            ubg += self.n_state * [0.]

        # Add the final cost
        J = J + dot(Xk, self.QN @ Xk)

        # Create an NLP solver and solve
        p += [self.lcs_theta]
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g), 'p': vertcat(*p)}
        self.oc_solver = casadi.nlpsol('solver', 'ipopt', prob, opts)

        self.lbw = DM(lbw)
        self.ubw = DM(ubw)
        self.lbg = DM(lbg)
        self.ubg = DM(ubg)
        self.w0 = DM(w0)

    def mpc_step_kl(self, lcs_theta, curr_state, kl_epsilon,  u_ref):

        # set the optimal control bounds
        lbw = self.lbw
        ubw = self.ubw
        init_w = self.w0
        lbw[0:self.n_state] = DM(curr_state)
        ubw[0:self.n_state] = DM(curr_state)
        init_w[0:self.n_state] = DM(curr_state)

        # set the optimal control parameter
        oc_parameters = vertcat(kl_epsilon,  u_ref, DM(lcs_theta))

        # print(oc_parameters.shape)
        # print(kl_epsilon)
        # print(x_ref.shape)
        # print(u_ref.shape)
        # print(lcs_theta.shape)


        sol = self.oc_solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=oc_parameters)
        w_opt = sol['x']
        self.w0 = w_opt

        curr_control = w_opt[self.n_state:self.n_state + self.n_control].full().flatten()

        return curr_control
