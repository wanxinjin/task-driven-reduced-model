import numpy as np
from casadi import *


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
        dyn_loss + lcp_loss / epsilon
        loss = dot(self.dyn[2:4] - self.x_next[2:4], self.dyn[2:4] - self.x_next[2:4]) + lcp_loss / epsilon
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

    def compute_lambda(self, x_batch, u_batch, x_next_batch, theta_val):

        # prepare the data
        batch_size = x_batch.shape[0]
        data_batch = np.hstack((x_batch, u_batch, x_next_batch))
        theta_val_batch = np.tile(theta_val, (batch_size, 1))
        data_theta_batch = np.hstack((data_batch, theta_val_batch))

        # compute the lam_phi solution
        sol_batch = self.inner_QPSolver(lbx=0.0, p=data_theta_batch.T)
        loss_opt_batch = sol_batch['f'].full().flatten()
        lam_phi_opt_batch = sol_batch['x'].full().T

        return lam_phi_opt_batch, loss_opt_batch

    def gradient_step(self, x_batch, u_batch, x_next_batch, theta_val, lam_phi_opt_batch, second_order=False):
        batch_size = x_batch.shape[0]
        data_batch = np.hstack((x_batch, u_batch, x_next_batch))
        theta_val_batch = np.tile(theta_val, (batch_size, 1))

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

    def dyn_prediction(self, x_batch, u_batch, theta_val):
        self.differetiable()

        batch_size = x_batch.shape[0]
        theta_val_batch = np.tile(theta_val, (batch_size, 1))
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


# class for using a lcs to do mpc control
class LCS_MPC:
    def __init__(self, A, B, C, D, E, F, lcp_offset):
        self.A = DM(A)
        self.B = DM(B)
        self.C = DM(C)
        self.D = DM(D)
        self.E = DM(E)
        self.F = DM(F)
        self.lcp_offset = DM(lcp_offset)

        self.n_state = self.A.shape[0]
        self.n_control = self.B.shape[1]
        self.n_lam = self.C.shape[1]

        # define the system variable
        x = casadi.MX.sym('x', self.n_state)
        u = casadi.MX.sym('u', self.n_control)
        xu_pair = vertcat(x, u)
        lam = casadi.MX.sym('lam', self.n_lam)

        # dynamics
        dyn = self.A @ x + self.B @ u + self.C @ lam
        self.dyn_fn = Function('dyn_fn', [xu_pair, lam], [dyn])

        # loss function
        lcp_loss = dot(self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset, lam)

        # constraints
        dis_cstr = self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset
        lam_cstr = lam
        total_cstr = vertcat(dis_cstr, lam_cstr)
        self.dis_cstr_fn = Function('dis_cstr_fn', [lam, xu_pair], [dis_cstr])

        # establish the qp solver to solve for LCP
        quadprog = {'x': lam, 'f': lcp_loss, 'g': total_cstr, 'p': xu_pair}
        opts = {'printLevel': 'none', }
        self.lcpSolver = qpsol('S', 'qpoases', quadprog, opts)

    def forward(self, x_t, u_t):
        xu_pair = vertcat(DM(x_t), DM(u_t))
        sol = self.lcpSolver(p=xu_pair, lbg=0.)
        lam_t = sol['x'].full().flatten()
        x_next = self.dyn_fn(xu_pair, lam_t).full().flatten()
        return x_next, lam_t

    def oc_setup(self, mpc_horizon):

        self.mpc_horizon = mpc_horizon

        # set the cost function parameters
        Q = MX.sym('Q', self.n_state, self.n_state)
        R = MX.sym('R', self.n_control, self.n_control)
        QN = MX.sym('QN', self.n_state, self.n_state)

        # define the parameters
        oc_parameters = vertcat(vec(Q), vec(R), vec(QN))

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
        Xk = casadi.MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += np.zeros(self.n_state).tolist()
        ubw += np.zeros(self.n_state).tolist()
        w0 += np.zeros(self.n_state).tolist()

        # formulate the NLP
        for k in range(self.mpc_horizon):
            # New NLP variable for the control
            Uk = casadi.MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0.]

            # new NLP variable for the complementarity variable
            Lamk = casadi.MX.sym('lam' + str(k), self.n_lam)
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
            Ck = dot(Xk, Q @ Xk) + dot(Uk, R @ Uk)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0.]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0.]
            ubg += self.n_state * [0.]

        # Add the final cost
        J = J + dot(Xk, QN @ Xk)

        # Create an NLP solver and solve
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g), 'p': oc_parameters}
        self.solver = casadi.nlpsol('solver', 'ipopt', prob, opts)
        self.lbw = DM(lbw)
        self.ubw = DM(ubw)
        self.lbg = DM(lbg)
        self.ubg = DM(ubg)
        self.w0 = DM(w0)

    def mpc(self, init_state, mat_Q, mat_R, mat_QN, init_guess=None):

        if init_guess is not None:
            self.w0 = init_guess['w_opt']

        # construct the parameter vector
        oc_parameters = vertcat(vec(mat_Q), vec(mat_R), vec(mat_QN))

        self.lbw[0:self.n_state] = DM(init_state)
        self.ubw[0:self.n_state] = DM(init_state)
        self.w0[0:self.n_state] = DM(init_state)

        # Solve the NLP
        sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=oc_parameters)
        w_opt = sol['x']
        g = sol['g']

        # extract the optimal control and state
        sol_traj = w_opt[0:self.mpc_horizon * (self.n_state + self.n_control + self.n_lam)].reshape(
            (self.n_state + self.n_control + self.n_lam, -1))
        x_traj = casadi.horzcat(sol_traj[0:self.n_state, :],
                                w_opt[self.mpc_horizon * (self.n_state + self.n_control + self.n_lam):]).T
        u_traj = sol_traj[self.n_state:self.n_state + self.n_control, :].T
        lam_traj = sol_traj[self.n_state + self.n_control:, :].T

        opt_sol = {'state_traj_opt': x_traj.full(),
                   'control_traj_opt': u_traj.full(),
                   'lam_traj_opt': lam_traj.full(),
                   'w_opt': w_opt,
                   }

        return opt_sol


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

        self.lcp_theta = vertcat(vec(self.D), vec(self.E), vec(self.G), vec(self.H), vec(self.lcp_offset))
        self.n_lcp_theta = self.lcp_theta.numel()
        self.dyn_theta = vertcat(vec(self.A), vec(self.B), vec(self.C))
        self.n_dyn_theta = self.dyn_theta.numel()

        self.F = stiffness * np.eye(self.n_lam) + self.G@self.G.T+ self.H - self.H.T
        self.F_fn = Function('F_fn', [self.lcp_theta], [self.F])
        self.D_fn = Function('D_fn', [self.lcp_theta], [self.D])
        self.E_fn = Function('E_fn', [self.lcp_theta], [self.E])
        self.G_fn = Function('G_fn', [self.lcp_theta], [self.G])
        self.H_fn = Function('H_fn', [self.lcp_theta], [self.H])
        self.A_fn = Function('A_fn', [self.dyn_theta], [self.A])
        self.B_fn = Function('B_fn', [self.dyn_theta], [self.B])
        self.C_fn = Function('C_fn', [self.dyn_theta], [self.C])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.lcp_theta], [self.lcp_offset])

    def differetiable(self):

        # lcp loss
        self.dist = self.D @ self.x + self.E @ self.u + self.F @ self.lam + self.lcp_offset
        lcp_data_theta = vertcat(self.x, self.u, self.lcp_theta)
        self.lcp_loss = dot(self.dist, self.lam)
        quadprog = {'x': self.lam, 'f': self.lcp_loss, 'g': self.dist, 'p': lcp_data_theta}
        opts = {'printLevel': 'none', }
        self.lcp_Solver = qpsol('lcp_Solver', 'qpoases', quadprog, opts)

        # define the dynamics loss
        self.x_next = SX.sym('x_next', self.n_state)
        data = vertcat(self.x, self.u, self.x_next)
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

    def compute_lambda(self, x_batch, u_batch, lcp_theta):
        self.differetiable()

        # prepare the data
        batch_size = x_batch.shape[0]
        lcp_data_batch = np.hstack((x_batch, u_batch))
        lcp_theta_batch = np.tile(lcp_theta, (batch_size, 1))
        lcp_data_theta_batch = np.hstack((lcp_data_batch, lcp_theta_batch))

        # compute the lam solution
        sol_batch = self.lcp_Solver(lbx=0.0, lbg=0.0, p=lcp_data_theta_batch.T)
        lcp_loss_opt_batch = sol_batch['f'].full().flatten()
        lam_opt_batch = sol_batch['x'].full().T

        return lam_opt_batch, lcp_loss_opt_batch

    def dyn_regression(self, x_batch, u_batch, lam_opt_batch, x_next_batch):
        # prepare the data
        batch_size = x_batch.shape[0]

        I = np.eye(self.n_state)
        kron_x = np.kron(x_batch, I)
        kron_u = np.kron(u_batch, I)
        kron_lam_opt = np.kron(lam_opt_batch, I)
        kron_x_next = x_next_batch.flatten()

        mat_A = np.hstack((kron_x, kron_u, kron_lam_opt))
        vec_b = kron_x_next

        # do the regression for dyn_theta
        dyn_theta_opt = inv(mat_A.T @ mat_A) @ (mat_A.T @ vec_b)
        dyn_loss_opt = dot(mat_A @ dyn_theta_opt - vec_b, mat_A @ dyn_theta_opt - vec_b) / batch_size

        return dyn_theta_opt.full().flatten(), dyn_loss_opt

    def gradient_step(self, x_batch, u_batch, x_next_batch, lam_opt_batch, dyn_theta_opt, lcp_theta):
        batch_size = x_batch.shape[0]
        data_batch = np.hstack((x_batch, u_batch, x_next_batch))
        dyn_theta_opt_batch = np.tile(dyn_theta_opt, (batch_size, 1))
        lcp_theta_batch = np.tile(lcp_theta, (batch_size, 1))

        # compute the gradient value
        dlcp = self.dloss_dlcp_fn(data_batch.T, lam_opt_batch.T, dyn_theta_opt_batch.T, lcp_theta_batch.T)
        dlcp_mean = dlcp.full().mean(axis=1)

        return dlcp_mean.flatten()

    def dyn_prediction(self, x_batch, u_batch, dyn_theta_opt, lcp_theta):
        self.differetiable()

        # prepare the data
        batch_size = x_batch.shape[0]
        lcp_data_batch = np.hstack((x_batch, u_batch))
        lcp_theta_batch = np.tile(lcp_theta, (batch_size, 1))
        lcp_data_theta_batch = np.hstack((lcp_data_batch, lcp_theta_batch))

        # compute the lam solution
        sol_batch = self.lcp_Solver(lbx=0.0, lbg=0.0, p=lcp_data_theta_batch.T)
        lam_opt_batch = sol_batch['x'].full().T

        # compute the next state batch
        dyn_theta_op_batch = np.tile(dyn_theta_opt, (batch_size, 1))
        x_next_batch = self.dyn_fn(x_batch.T, u_batch.T, lam_opt_batch.T, dyn_theta_op_batch.T).full().T

        return x_next_batch, lam_opt_batch

    # do statistics for the modes


def statiModes(lam_batch, tol=1e-5):
    # dimension of the lambda
    n_lam = lam_batch.shape[1]
    # total number of modes
    total_n_mode = float(2 ** n_lam)

    # do the statistics for the modes
    lam_batch_mode = np.where(lam_batch < tol, 0, 1)
    unique_mode_list, mode_count_list = np.unique(lam_batch_mode, axis=0, return_counts=True)
    mode_frequency_list = mode_count_list / lam_batch.shape[0]

    return unique_mode_list, mode_frequency_list


# do the plot of differnet mode
def plotModes(lam_batch, tol=1e-5):
    # do the statistics for the modes
    lam_batch_mode = np.where(lam_batch < tol, 0, 1)
    unique_mode_list, mode_indices = np.unique(lam_batch_mode, axis=0, return_inverse=True)

    return unique_mode_list, mode_indices
