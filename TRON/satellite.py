import numpy as np
from TRON.utils import regularize

SX_DIM = 6 # 12 if nonlinear
SU_DIM = 3


class SatelliteEnv:
    def __init__(self, args):

        self.m_ego = args['m_ego']
        self.m_target = args['m_target']

        self.A_ego = args['A_ego']
        self.A_target = args['A_target']

        self.Cd_ego = args['Cd_ego']
        self.Cd_target = args['Cd_target']

        self.mu = args['mu']        
        self.a = args['a']
        self.omega = args['omega']
        self.orbit_radius = args['orbit_radius']

        self.Q = args['Q']
        self.Qf = args['Qf']
        self.R = args['R']
        self.x_start = args['x_start']        
        self.alpha = args['alpha']
        self.dt = args['dt']
        self.horizon_length = args['horizon_length']
        self.cost_function = args['cost_function']

        # AULA
        self.theta = args['theta']
        self.eta = args['eta']

        # ADMM
        self.y = args['y']
        self.lam = args['lam']
        self.rho = args['rho']

    def f_nonlinear(self, x, u):
        '''
        Continuous time dynamics
        '''
        x_dot = np.zeros(SX_DIM)
        x_dot[:6] = x[6:]
        r_ego = x[3:6] - x[0:3]
        rd_ego = x[9:12] - x[6:9]
        r_target = x[3:6]
        rd_target = x[9:12]

        y_cw = r_target / np.linalg.norm(r_target)
        z_cw = np.cross(y_cw, rd_target)
        z_cw = z_cw / np.linalg.norm(z_cw)
        x_cw = np.cross(y_cw, z_cw)
        u_world = np.vstack([x_cw, y_cw, z_cw]).dot(u)
        F_ego = u_world + self.gravitation_force(r_ego, self.m_ego)
        F_ego += self.drag_force(r_ego, rd_ego, self.A_ego, self.Cd_ego)
        F_target = self.gravitation_force(r_target, self.m_target)
        F_target += self.drag_force(r_target, rd_target, self.A_target, self.Cd_target)
        x_dot[9:12] = F_target / self.m_target
        x_dot[6:9] = x_dot[9:12] - F_ego / self.m_ego

        return x_dot

    def gravitation_force(self, r, mass):
        mu = self.mu
        rmag = np.linalg.norm(r)
        F = -mu * mass / rmag**3 * r
        return F

    def drag_force(self, r, v, A, Cd):
        alt = np.linalg.norm(r) - self.a
        density = self.atmospheric_density(alt)
        v_rel = v + np.cross(self.omega, r)
        F = -0.5 * Cd * A * density * np.linalg.norm(v_rel) * v_rel
        return F

    def atmospheric_density(self, alt):
        density = 9.201e-5 * np.exp(-5.301e-5 * alt)
        return density

    def f_linear(self, x, u):
        x_dot = np.zeros(SX_DIM)
        n_ = np.sqrt(self.mu / self.orbit_radius**3)
        x_dot[0:3] = x[3:6]
        x_dot[3] = 2 * n_ * x[4] + u[0] / self.m_ego
        x_dot[4] = -2 * n_ * x[3] + 3 * n_**2 * x[1] + u[1] / self.m_ego
        x_dot[5] = -n_**2 * x[2] + u[2] / self.m_ego

        return x_dot

    def g(self, x, u):
        '''
        Discrete time dynamics
        '''
        k1 = self.f_linear(x, u)
        k2 = self.f_linear(x + 0.5 * self.dt * k1, u)
        k3 = self.f_linear(x + 0.5 * self.dt * k2, u)
        k4 = self.f_linear(x + self.dt * k3, u)

        return x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def cost(self, x, u, t, sparse=True):
        cost = 0.0
        cost += 0.5 * x.T.dot(self.Q.dot(x))
        cost += 0.5 * u.T.dot(self.R.dot(u))
        if sparse:            
            cost += self.sparse_control_cost(u, t)
        return cost

    def sparse_control_cost(self, u, t):
        sparse_cost = 0.0
        if self.cost_function == 'l1':            
            sparse_cost += self.alpha * np.linalg.norm(u, ord=1)
        elif self.cost_function == 'admm':
            sparse_cost += self.alpha * np.linalg.norm(self.y[t], ord=1) + self.lam[t].dot(u - self.y[t]) + (self.rho / 2.0) * (u - self.y[t]).dot(u - self.y[t])
        elif self.cost_function == 'aula':
            for d in range(SU_DIM):
                sparse_cost += self.get_aula_sparse_cost(u[d], self.theta[t, d])
        else:
            raise NotImplementedError()
        return sparse_cost

    def get_aula_sparse_cost(self, u, theta):
        if u >= 0:
            return self.eta * np.log(theta[0] + theta[1] * np.exp((-2 * self.alpha * u) / self.eta)) + self.alpha * u
        else:
            return self.eta * np.log(theta[1] + theta[0] * np.exp((2 * self.alpha * u) / self.eta)) - self.alpha * u

    def final_cost(self, x):
        cost = 0.5 * x.T.dot(self.Qf.dot(x))
        return cost

    def quadratize_cost(self, x, u, t, it):
        Qt = self.Q
        qt = np.zeros(SX_DIM)

        qsp, Qsp = self.quadratize_sparse_control_cost(u, t)
        Qt = regularize(Qt)
        Rt = Qsp + self.R
        rt = qsp
        Pt = np.zeros((SU_DIM, SX_DIM))

        return Pt, qt, Qt, rt, Rt

    def quadratize_final_cost(self, x):
        Qf = self.Qf
        qf = np.zeros(SX_DIM)

        return qf, Qf

    def quadratize_sparse_control_cost(self, u, t):
        Qsp = np.zeros((SU_DIM, SU_DIM))
        qsp = np.zeros(SU_DIM)
        if self.cost_function == 'l1':            
            qsp[0] = self.alpha if u[0] >= 0 else -self.alpha
            qsp[1] = self.alpha if u[1] >= 0 else -self.alpha
            qsp[2] = self.alpha if u[2] >= 0 else -self.alpha
        elif self.cost_function == 'admm':
            qsp = self.lam[t] + self.rho * (u - self.y[t])
            Qsp = np.diag(np.ones(SU_DIM) * self.rho)
        elif self.cost_function == 'aula':
            for d in range(SU_DIM):
                qsp[d], Qsp[d, d] = self.get_aula_quadratized_sparse_cost(u[d], self.theta[t, d])
        else:
            raise NotImplementedError()

        return qsp, Qsp

    def get_aula_quadratized_sparse_cost(self, u, theta):
        if u >= 0:
            expterm = np.exp((-2 * self.alpha * u) / self.eta)
            qsp = self.alpha - ((2 * self.alpha * theta[1] * expterm) / (theta[0] + theta[1] * expterm))
            Qsp = (4 * self.alpha**2 * theta[0] * theta[1] * expterm) / (self.eta * (theta[1] * expterm + theta[0])**2)
            return qsp, Qsp
        else:
            expterm = np.exp((2 * self.alpha * u) / self.eta)
            qsp = ((2 * self.alpha * theta[0] * expterm) / (theta[1] + theta[0] * expterm)) - self.alpha
            Qsp = (4 * self.alpha**2 * theta[0] * theta[1] * expterm) / (self.eta * (theta[0] * expterm + theta[1])**2)
            return qsp, Qsp

    def get_cost(self, xs, us):
        return self.get_state_action_cost(xs, us) + self.get_sparse_control_cost(us)

    def get_state_action_cost(self, xs, us):
        horizon_length = len(xs) - 1
        cost = 0.0
        for i in range(horizon_length):
            cost += self.cost(xs[i], us[i], i, sparse=False)
        cost += self.final_cost(xs[-1])
        return cost

    def get_sparse_control_cost(self, us):
        cost = 0.0
        for u in us[:-1]:
            cost += self.alpha * np.linalg.norm(u, ord=1)

        return cost

    def rollout(self, l, L, verbose=True):
        xs = []
        us = []
        x = self.x_start
        xs.append(x)
        for t in range(self.horizon_length):
            if verbose:
                print(bcolors.OKGREEN+str(t)+' : '+str(x)+bcolors.ENDC)
            u = L[t].dot(x) + l[t]
            x = self.g(x, u)

            xs.append(x)
            us.append(u)

        if verbose:
            print(bcolors.OKGREEN+str(t)+' : '+str(x)+bcolors.ENDC)

        return xs, us

    def update_y(self, y):
        self.y = y
        return

    def update_lam(self, lam):
        self.lam = lam
        return

    def update_rho(self, rho):
        self.rho = rho
        return

    def update_theta(self, theta):
        self.theta = theta
        return

    def update_eta(self, eta):
        self.eta = eta
        return

    def get_all_sparse_control_costs(self, us):
        costs = np.zeros((self.horizon_length, SU_DIM, 2))
        t = 0
        for u in us[:-1]:
            for d in range(SU_DIM):
                costs[t, d, 0] = u[d]
                costs[t, d, 1] = -u[d]
            t += 1

        return costs
