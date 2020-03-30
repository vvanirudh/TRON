import numpy as np
from TRON.utils import regularize, cpMatrix, rotFromErr, skewSymmetric, ThirdorderDerivative, errFromRot, bcolors

NX_DIM = 6
NU_DIM = 3
NDIM = 3
NB_DIM = 27
NZ_DIM = 3

OVERFLOW_CAP = 1e5


class Obstacle:
    def __init__(self, pos, radius, dim, robot_radius):
        self.pos = pos
        self.radius = radius
        self.dim = dim
        self.robot_radius = robot_radius
        return

    def dist(self, x):
        return np.linalg.norm(x[:NDIM] - self.pos) - self.robot_radius - self.radius


class NeedleEnv:
    def __init__(self, args):
        self.obstacles = []
        self.obstacle_factor = args['obstacle_factor']
        self.scale_factor = args['scale_factor']
        self.robot_radius = args['robot_radius']
        self.rot_cost = args['rot_cost']
        self.Q = args['Q']
        self.R = args['R']
        self.x_goal = args['x_goal']
        self.x_start = args['x_start']
        self.u_nominal = args['u_nominal']
        self.bottom_left = args['bottom_left']
        self.top_right = args['top_right']
        self.dt = args['dt']
        self.horizon_length = args['horizon_length']
        self.cost_function = args['cost_function']

        self.sparse_cost_coeff = args['sparse_cost_coeff']
        self.kmax = args['kmax']

        self.theta = args['theta']
        self.eta = args['eta']

        self.y = args['y']
        self.lam = args['lam']
        self.rho = args['rho']

        self.d = 0.0009765625 / 100.0  # For finite difference

    def add_obstacle(self, pos, radius, dim=3):
        self.obstacles.append(Obstacle(pos, radius, dim, self.robot_radius))
        return

    def exp_obstacle_cost(self, dist):
        if dist >= 0:
            return self.obstacle_factor * np.exp(-self.scale_factor * dist)
        else:
            return min(self.obstacle_factor * np.exp(-self.scale_factor * dist), OVERFLOW_CAP)

    def obstacle_cost(self, x):
        cost = 0.0
        ind = 0
        for obs in self.obstacles:
            d = x[:NDIM] - obs.pos
            d[obs.dim] = 0  # TODO: What is this doing?
            distr = np.linalg.norm(d)
            dist = distr - self.robot_radius - obs.radius
            cost += self.exp_obstacle_cost(dist)
            ind += 1

        for i in range(NDIM):
            dist = x[i] - self.bottom_left[i] - self.robot_radius
            cost += self.exp_obstacle_cost(dist)
            ind += 1

        for i in range(NDIM):
            dist = self.top_right[i] - x[i] - self.robot_radius
            cost += self.exp_obstacle_cost(dist)
            ind += 1

        return cost

    def quadratize_obstacle_cost(self, x, q, Q):
        QObs = np.zeros((NDIM, NDIM))
        qObs = np.zeros(NDIM)

        for obs in self.obstacles:
            d = x[:NDIM] - obs.pos
            d[obs.dim] = 0
            distr = np.linalg.norm(d)
            dist = distr - self.robot_radius - obs.radius
            d /= distr

            n = np.zeros(3)
            n[obs.dim] = 1
            d_ortho = np.dot(skewSymmetric(n), d)

            a0 = self.exp_obstacle_cost(dist)
            a1 = -self.scale_factor * a0
            a2 = -self.scale_factor * a1

            b2 = a1 / distr
            QObs += a2 * (np.outer(d, d)) + b2 * np.outer(d_ortho, d_ortho)
            qObs += a1*d

        for i in range(NDIM):
            dist = (x[i] - self.bottom_left[i]) - self.robot_radius
            d = np.zeros(NDIM)
            d[i] = 1.0

            a0 = self.exp_obstacle_cost(dist)
            a1 = -self.scale_factor*a0
            a2 = -self.scale_factor*a1

            QObs += a2*(np.outer(d, d))
            qObs += a1*d

        for i in range(NDIM):
            dist = (self.top_right[i] - x[i]) - self.robot_radius
            d = np.zeros(NDIM)
            d[i] = -1.0

            a0 = self.exp_obstacle_cost(dist)
            a1 = -self.scale_factor*a0
            a2 = -self.scale_factor*a1

            QObs += a2*(np.outer(d, d))
            qObs += a1*d

        QObs = regularize(QObs)
        Q[:NDIM, :NDIM] = QObs + Q[:NDIM, :NDIM]
        q[:NDIM] = qObs - QObs.dot(x[:NDIM]) + q[:NDIM]
        return q, Q

    def cost(self, x, u, t, obstacle=True, sparse=True):
        cost = 0.0
        if t == 0:
            cost += 0.5 * \
                (x - self.x_start).T.dot(self.Q.dot(x - self.x_start))
        else:
            if obstacle:
                cost += self.obstacle_cost(x)
        cost += 0.5 * \
            (u - self.u_nominal).T.dot(self.R.dot(u - self.u_nominal))
        if sparse:
            cost += self.sparse_control_cost(u, t)
        return cost

    def sparse_control_cost(self, u, t):
        # TODO: Check this. There is a bug in the aula part since cost sometimes goes negative
        # The cost function is \lambda * \|w\|_1
        sparse_cost = 0.0
        if self.cost_function == 'l1':
            sparse_cost += self.sparse_cost_coeff * np.abs(u[1])
            # sparse_cost += self.sparse_cost_coeff * np.abs(self.kmax - u[2])
        elif self.cost_function == 'aula':
            if u[1] >= 0:
                sparse_cost += self.eta * \
                    np.log(self.theta[t, 0] + self.theta[t, 1]
                           * np.exp((-2 * self.sparse_cost_coeff * u[1]) / self.eta)) + self.sparse_cost_coeff * u[1]
            else:
                sparse_cost += self.eta * \
                    np.log(self.theta[t, 1] + self.theta[t, 0]
                           * np.exp((2 * self.sparse_cost_coeff * u[1]) / self.eta)) - self.sparse_cost_coeff * u[1]
        elif self.cost_function == 'admm':
            sparse_cost += self.sparse_cost_coeff * np.abs(self.y[t]) + self.lam[t] * (u[1] - self.y[t]) + (self.rho / 2.0) * (u[1] - self.y[t])**2
        else:
            raise NotImplementedError()
        return sparse_cost

    def quadratize_sparse_control_cost(self, u, t):
        Qsp = np.zeros((NU_DIM, NU_DIM))
        qsp = np.zeros(NU_DIM)

        if self.cost_function == 'l1':

            if u[1] >= 0:
                qsp[1] = self.sparse_cost_coeff
            else:
                qsp[1] = -self.sparse_cost_coeff

            # if u[2] < self.kmax:
            #     qsp[2] = -self.sparse_cost_coeff
            # else:
            #     qsp[2] = self.sparse_cost_coeff

        elif self.cost_function == 'aula':
            if u[1] >= 0:
                expterm = np.exp(
                    (-2 * self.sparse_cost_coeff * u[1]) / self.eta)
                qsp[1] = self.sparse_cost_coeff - ((2 * self.sparse_cost_coeff * self.theta[t, 1] * expterm) / (
                    self.theta[t, 0] + self.theta[t, 1] * expterm))

                Qsp[1, 1] = (4 * self.sparse_cost_coeff**2 * self.theta[t, 0] * self.theta[t, 1]
                             * expterm) / (self.eta * (self.theta[t, 1] * expterm + self.theta[t, 0])**2)

            else:
                expterm = np.exp(
                    (2 * self.sparse_cost_coeff * u[1]) / self.eta)
                qsp[1] = ((2 * self.sparse_cost_coeff * self.theta[t, 0]
                           * expterm) / (self.theta[t, 1] + self.theta[t, 0] * expterm)) - self.sparse_cost_coeff
                Qsp[1, 1] = (4 * self.sparse_cost_coeff**2 * self.theta[t, 0] * self.theta[t, 1]
                             * expterm) / (self.eta * (self.theta[t, 0] * expterm + self.theta[t, 1])**2)

        elif self.cost_function == 'admm':
            qsp[1] = self.lam[t] + self.rho * (u[1] - self.y[t])
            Qsp[1, 1] = self.rho

        else:
            raise NotImplementedError

        return qsp, Qsp

    def final_cost(self, x):
        cost = 0.5 * \
            (x - self.x_goal).T.dot(self.Q.dot(x - self.x_goal))
        return cost

    def quadratize_cost(self, x, u, t, it):
        if t == 0:
            Qt = self.Q
            qt = -self.Q.dot(self.x_start)
        else:
            Qt = np.zeros((NX_DIM, NX_DIM))
            qt = np.zeros(NX_DIM)

            if it < 1:
                Qt[3:, 3:] = self.rot_cost * np.eye(3)

            qt, Qt = self.quadratize_obstacle_cost(x, qt, Qt)
        qsp, Qsp = self.quadratize_sparse_control_cost(u, t)

        Qt = regularize(Qt)
        Rt = self.R + Qsp
        rt = qsp - self.R.dot(self.u_nominal)
        Pt = np.zeros((NU_DIM, NX_DIM))

        # qscalart = self.cost(x, u, t) + 0.5 * x.T.dot(Qt.dot(x)) + 0.5 * \
        #     u.T.dot(Rt.dot(u)) - x.T.dot(qt + Qt.dot(x)) - \
        #     u.T.dot(rt + Rt.dot(u))

        return Pt, qt, Qt, rt, Rt

    def quadratize_final_cost(self, x):
        Q = self.Q
        q = -self.Q.dot(self.x_goal)

        # qscalar_final = self.final_cost(
        #     x) + 0.5 * x.T.dot(Q.dot(x)) - x.T.dot(q + Q.dot(x))

        return q, Q

    def se_dynamics(self, x, u):
        '''
        u denotes the control input
        u is a 3D vector where the elements are [v, w, k] where 
          v is the linear forward speed of the needle
          w is the angular speed with which the needle is rotated at the base
          k is the desired curvature of the needle (given w, the spin intervals decide the curvature obtained)

        x denotes the state vector
        x is a 6D-vector where the first 3 elements define the rigid body position p
        and the last 3 elements define the rigid body orientation
        '''
        U = np.zeros((4, 4))
        v = np.zeros(3)
        w = np.zeros(3)

        w[0] = u[0] * u[2]
        w[2] = u[1]

        v[2] = u[0]

        U[0:3, 0:3] = cpMatrix(w)
        U[0:3, 3] = v

        S = np.zeros((4, 4))
        S[0:3, 0:3] = rotFromErr(x[3:])
        S[0:3, 3] = x[0:3]
        S[3, 3] = 1.0

        k1 = np.dot(S, U)
        k2 = np.dot(S + 0.5 * self.dt * k1, U)
        k3 = np.dot(S + 0.5 * self.dt * k2, U)
        k4 = np.dot(S + self.dt * k3, U)

        SN = S + 1.0/6.0 * self.dt * (k1 + 2*k2 + 2*k3 + k4)

        xnext = np.zeros(6)
        xnext[0:3] = SN[0:3, 3]
        xnext[3:] = errFromRot(SN[0:3, 0:3])

        M = np.sqrt(u.T.dot(u)) * np.eye(6) * 0.01

        return xnext, M

    def g(self, x, u):
        xnext, _ = self.se_dynamics(x, u)
        return xnext

    def inverse_dynamics(self, xnext, u):
        U = np.zeros((4, 4))
        v = np.zeros(3)
        w = np.zeros(3)

        w[0] = u[0] * u[2]
        w[2] = u[1]

        v[2] = u[0]

        U[0:3, 0:3] = cpMatrix(w)
        U[0:3, 3] = v

        Sn = np.zeros((4, 4))
        Sn[0:3, 0:3] = rotFromErr(xnext[3:])
        Sn[0:, 3] = xnext[0:3]
        Sn[3, 3] = 1.0

        k1 = np.dot(Sn, U)
        k2 = np.dot(Sn - 0.5 * self.dt * k1, U)
        k3 = np.dot(Sn - 0.5 * self.dt * k2, U)
        k4 = np.dot(Sn - self.dt * k3, U)

        S = Sn - 1.0/6.0 * self.dt * (k1 + 2*k2 + 2*k3 + k4)

        x = np.zeros(6)
        x[0:3] = S[0:3, 3]
        x[3:] = errFromRot(S[0:3, 0:3])

        return x

    def linearize_discrete_dynamics(self, xstar, ustar):
        At = np.zeros((NX_DIM, NX_DIM))
        Bt = np.zeros((NX_DIM, NU_DIM))
        ct = np.zeros(NX_DIM)
        F = np.zeros((NX_DIM, NX_DIM, NX_DIM))
        G = np.zeros((NX_DIM, NX_DIM, NU_DIM))
        e = np.zeros((NX_DIM, NX_DIM))

        orig_xn, orig_M = self.se_dynamics(xstar, ustar)

        for c in range(0, NX_DIM):
            augrr = xstar.copy()
            augr = xstar.copy()
            augl = xstar.copy()
            augll = xstar.copy()

            augrr[c] += 2 * self.d
            augr[c] += self.d
            augl[c] -= self.d
            augll[c] -= 2 * self.d

            augrrxn, augrrM = self.se_dynamics(augrr, ustar)
            augrxn, augrM = self.se_dynamics(augr, ustar)
            auglxn, auglM = self.se_dynamics(augl, ustar)
            augllxn, augllM = self.se_dynamics(augll, ustar)

            tmpd = ThirdorderDerivative(augll[c], augl[c], augr[c], augrr[c],
                                        augllxn, auglxn, augrxn, augrrxn)

            At[0:NX_DIM, c] = tmpd.copy()
            # FIX: Removed the next loop since its redundant
            # for i in range(NX_DIM):
            #     F[i, 0:NX_DIM, c] = np.zeros(NX_DIM)

        for c in range(0, NU_DIM):
            augru, augrru, auglu, augllu = ustar.copy(
            ), ustar.copy(), ustar.copy(), ustar.copy()
            augru[c] += self.d
            auglu[c] -= self.d
            augrru[c] += 2 * self.d
            augllu[c] -= 2 * self.d

            augrrxn, augrrM = self.se_dynamics(xstar, augrru)
            augrxn, augrM = self.se_dynamics(xstar, augru)
            auglxn, auglM = self.se_dynamics(xstar, auglu)
            augllxn, augllM = self.se_dynamics(xstar, augllu)

            tmpd = ThirdorderDerivative(augllu[c], auglu[c], augru[c], augrru[c],
                                        augllxn, auglxn, augrxn, augrrxn)

            Bt[0:NX_DIM, c] = tmpd.copy()
            for i in range(0, NX_DIM):
                G[i, 0:NX_DIM, c] = (augrM[0:NX_DIM, i] -
                                     auglM[0:NX_DIM, i]) / (2.0 * self.d)

        ct = orig_xn - np.dot(At, xstar) - np.dot(Bt, ustar)

        for i in range(NX_DIM):
            e[i] = orig_M[0:NX_DIM, i] - \
                np.dot(F[i], xstar) - np.dot(G[i], ustar)

        return At, Bt, ct, F, G, e

    def linearize_discrete_inverse_dynamics(self, xnstar, ustar):
        Abart, Bbart, cbart = np.zeros((NX_DIM, NX_DIM)), np.zeros(
            (NX_DIM, NU_DIM)), np.zeros(NX_DIM)

        orig_x = np.zeros(NX_DIM)
        orig_x = self.inverse_dynamics(xnstar, ustar)

        for c in range(NX_DIM):
            augrxn, augrrxn, auglxn, augllxn = xnstar.copy(
            ), xnstar.copy(), xnstar.copy(), xnstar.copy()
            augrrxn[c] += 2 * self.d
            augrxn[c] += self.d
            auglxn[c] -= self.d
            augllxn[c] -= 2 * self.d

            grrbar = self.inverse_dynamics(augrrxn, ustar)
            grbar = self.inverse_dynamics(augrxn, ustar)
            glbar = self.inverse_dynamics(auglxn, ustar)
            gllbar = self.inverse_dynamics(augllxn, ustar)

            tmpd = ThirdorderDerivative(augllxn[c], auglxn[c], augrxn[c], augrrxn[c],
                                        gllbar, glbar, grbar, grrbar)

            Abart[0:NX_DIM, c] = tmpd.copy()

        for c in range(NU_DIM):
            augru, augrru, auglu, augllu = ustar.copy(
            ), ustar.copy(), ustar.copy(), ustar.copy()
            augrru[c] += 2 * self.d
            augru[c] += self.d
            auglu[c] -= self.d
            augllu[c] -= 2 * self.d

            grrbar = self.inverse_dynamics(xnstar, augrru)
            grbar = self.inverse_dynamics(xnstar, augru)
            glbar = self.inverse_dynamics(xnstar, auglu)
            gllbar = self.inverse_dynamics(xnstar, augllu)

            tmpd = ThirdorderDerivative(augllu[c], auglu[c], augru[c], augrru[c],
                                        gllbar, glbar, grbar, grrbar)

            Bbart[0:NX_DIM, c] = tmpd.copy()

        cbart = orig_x - np.dot(Abart, xnstar) - np.dot(Bbart, ustar)

        return Abart, Bbart, cbart

    def update_theta(self, theta):
        '''
        Function that updates the lagrange multipliers
        '''
        self.theta = theta.copy()
        return

    def update_eta(self, eta):
        '''
        Function that updates the KL penalty coefficient
        '''
        self.eta = eta
        return

    def update_y(self, y):
        '''
        Function that updates the dummy admm variables
        '''
        self.y = y
        return

    def update_lam(self, lam):
        '''
        Function that updates the admm lagrange multipliers
        '''
        self.lam = lam
        return

    def update_rho(self, rho):
        '''
        Function that updates the admm penalty term
        '''
        self.rho = rho
        return

    def get_all_sparse_control_costs(self, us):
        costs = np.zeros((self.horizon_length, 2))
        t = 0
        for u in us[:-1]:
            costs[t, 0] = u[1]
            costs[t, 1] = -u[1]
            t += 1

        return costs

    # def get_cost(self, plan):
    #     xs = [p[0] for p in plan]
    #     us = [p[1] for p in plan]
    #     return self.get_state_action_cost(xs, us) + self.get_sparse_control_cost(us)

    def get_cost(self, xs, us):
        return self.get_state_action_cost(xs, us) + self.get_sparse_control_cost(us)

    def get_state_action_cost(self, xs, us):
        horizon_length = len(xs) - 1
        cost = 0.0
        for i in range(horizon_length):
            cost += self.cost(xs[i], us[i], i, obstacle=True, sparse=False)
        cost += self.final_cost(xs[-1])
        return cost

    def get_sparse_control_cost(self, us):
        cost = 0.0
        for u in us[:-1]:
            cost += self.sparse_cost_coeff * np.abs(u[1])

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
