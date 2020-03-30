import numpy as np
from TRON.utils import regularize, bcolors

X_DIM = 3
U_DIM = 2
DIM = 2

OVERFLOW_CAP = 1e5


class Obstacle:
    def __init__(self, pos, radius, dim, robot_radius):
        self.pos = pos
        self.radius = radius
        self.dim = dim
        self.robot_radius = robot_radius
        return

    def dist(self, x):
        return np.linalg.norm(x[:DIM] - self.pos) - self.robot_radius - self.radius


class DiffDriveEnv:
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

        self.theta = args['theta']
        self.eta = args['eta']
        return

    # def add_obstacle(self, obs):
    #     self.obstacles.append(obs)
    #     return

    def add_obstacle(self, pos, radius, dim=2):
        self.obstacles.append(Obstacle(pos, radius, dim, self.robot_radius))
        return

    def get_aula_obstacle_cost(self, theta, dist):
        if dist >= 0:
            return self.eta * np.log(theta[0] + theta[1] * np.exp((-self.obstacle_factor * self.scale_factor * dist) / self.eta))
        else:
            return self.eta * np.log(theta[1] + theta[0] * np.exp((self.obstacle_factor * self.scale_factor * dist) / self.eta)) - ((self.obstacle_factor * self.scale_factor * dist) / self.eta)

    def compute_dist_to_obstacle(self, x, obs):
        d = x[:DIM] - obs.pos
        distr = np.linalg.norm(d)
        dist = distr - self.robot_radius - obs.radius

        return dist

    def compute_dist_to_boundary(self, x, dim, bottom=False):
        if bottom:            
            dist = x[dim] - self.bottom_left[dim] - self.robot_radius
        else:
            dist = self.top_right[dim] - self.robot_radius - x[dim]

        return dist

    def obstacle_cost(self, x, t):
        '''
        Obstacle cost
        '''
        cost = 0.0
        ind = 0
        # Obstacles
        for obs in self.obstacles:
            dist = self.compute_dist_to_obstacle(x, obs)
            if self.cost_function == 'exp':
                cost += self.obstacle_factor * \
                    np.exp(-self.scale_factor * dist)
            elif self.cost_function == 'hinge':
                cost += self.obstacle_factor * \
                    max(0, -self.scale_factor * dist)
            elif self.cost_function == 'aula':
                # cost += self.obstacle_factor * (self.theta[t, ind, 1] * -self.scale_factor * dist)
                cost += self.get_aula_obstacle_cost(self.theta[t, ind], dist)
            else:
                raise NotImplementedError
            ind += 1

        # Bottom and left boundaries
        for i in range(DIM):
            dist = self.compute_dist_to_boundary(x, i, bottom=True)
            if self.cost_function == 'exp':
                cost += self.obstacle_factor * \
                    np.exp(-self.scale_factor * dist)
            elif self.cost_function == 'hinge':
                cost += self.obstacle_factor * \
                    max(0, -self.scale_factor * dist)
            elif self.cost_function == 'aula':
                # cost += self.obstacle_factor * (self.theta[t, ind, 1] * -self.scale_factor * dist)
                cost += self.get_aula_obstacle_cost(self.theta[t, ind], dist)
            else:
                raise NotImplementedError
            ind += 1

        # Top and right boundaries
        for i in range(DIM):
            dist = self.compute_dist_to_boundary(x, i, bottom=False)
            if self.cost_function == 'exp':
                cost += self.obstacle_factor * \
                    np.exp(-self.scale_factor * dist)
            elif self.cost_function == 'hinge':
                cost += self.obstacle_factor * \
                    max(0, -self.scale_factor * dist)
            elif self.cost_function == 'aula':
                # cost += self.obstacle_factor * (self.theta[t, ind, 1] * -self.scale_factor * dist)
                cost += self.get_aula_obstacle_cost(self.theta[t, ind], dist)
            else:
                raise NotImplementedError
            ind += 1

        return cost

    def quadratize_obstacle_cost(self, x, q, Q):
        '''
        Quadratize obstacle cost around x
        '''
        QObs = np.zeros((DIM, DIM))
        qObs = np.zeros(DIM)

        # Obstacles
        for obs in self.obstacles:
            d = x[:DIM] - obs.pos
            distr = np.linalg.norm(d)
            d /= distr
            dist = distr - obs.radius - self.robot_radius

            d_ortho = np.array([d[1], -d[0]])
            a0 = self.obstacle_factor * min(np.exp(-self.scale_factor * dist), OVERFLOW_CAP)
            a1 = -self.scale_factor * a0
            a2 = -self.scale_factor * a1

            b2 = a1 / distr
            QObs += a2*(np.outer(d, d)) + b2*(np.outer(d_ortho, d_ortho))
            qObs += a1*d

        # Bottom and left boundaries
        for i in range(DIM):
            dist = (x[i] - self.bottom_left[i]) - self.robot_radius
            d = np.zeros(DIM)
            d[i] = 1.0

            a0 = self.obstacle_factor * min(np.exp(-self.scale_factor*dist), OVERFLOW_CAP)
            a1 = -self.scale_factor*a0
            a2 = -self.scale_factor*a1

            QObs += a2*(np.outer(d, d))
            qObs += a1*d

        # Right and top boundaries
        for i in range(DIM):
            dist = (self.top_right[i] - x[i]) - self.robot_radius
            d = np.zeros(DIM)
            d[i] = -1.0

            a0 = self.obstacle_factor * min(np.exp(-self.scale_factor*dist), OVERFLOW_CAP)
            a1 = -self.scale_factor*a0
            a2 = -self.scale_factor*a1

            QObs += a2*(np.outer(d, d))
            qObs += a1*d

        QObs = regularize(QObs)
        Q[:DIM, :DIM] = QObs + Q[:DIM, :DIM]
        q[:DIM] = qObs - QObs.dot(x[:DIM]) + q[:DIM]
        return q, Q

    def quadratize_obstacle_cost_hinge(self, x, q, Q):
        QObs = np.zeros((DIM, DIM))
        qObs = np.zeros(DIM)

        # Obstacles
        for obs in self.obstacles:
            d = x[:DIM] - obs.pos
            distr = np.linalg.norm(d)
            dist = distr - obs.radius - self.robot_radius

            if dist > 0:
                # Hinge cost is 0, so nothing to do here
                pass
            else:
                # Hinge cost is self.obstacle_factor * -self.scale_factor * dist
                a0 = -self.obstacle_factor * self.scale_factor
                a1 = a0 / (distr**2)
                a2 = 1.0 / distr

                qObs += a0 * (d / distr)
                QObs += a1 * (distr * np.eye(DIM) - a2*np.outer(d, d))

        # Bottom and left boundaries
        for i in range(DIM):
            dist = (x[i] - self.bottom_left[i]) - self.robot_radius
            d = np.zeros(DIM)
            d[i] = 1.0

            if dist > 0:
                # Hinge cost is 0, so nothing to do here
                pass
            else:
                # Hinge cost is self.obstacle_factor * -self.scale_factor * dist
                a0 = -self.obstacle_factor * self.scale_factor

                qObs += a0 * d
                # Hessian is zero

        # Top and right boundaries
        for i in range(DIM):
            dist = (self.top_right[i] - x[i]) - self.robot_radius
            d = np.zeros(DIM)
            d[i] = -1.0

            if dist > 0:
                # Hinge cost is 0, so nothing to do here
                pass
            else:
                # Hinge cost is self.obstacle_factor * -self.scale_factor * dist
                a0 = -self.obstacle_factor * self.scale_factor

                qObs += a0 * d
                # Hessian is zero

        QObs = regularize(QObs)
        Q[:DIM, :DIM] = QObs + Q[:DIM, :DIM]
        q[:DIM] = qObs - QObs.dot(x[:DIM]) + q[:DIM]
        return q, Q

    def quadratize_obstacle_cost_aula(self, x, q, Q, t):
        QObs = np.zeros((DIM, DIM))
        qObs = np.zeros(DIM)
        scaleobstacle = self.obstacle_factor * self.scale_factor

        ind = 0
        # Obstacles
        for obs in self.obstacles:
            d = x[:DIM] - obs.pos
            distr = np.linalg.norm(d)
            dist = distr - obs.radius - self.robot_radius

            #a0 = -self.obstacle_factor * self.scale_factor * self.theta[t, ind, 1]
            #a1 = a0 / (distr**2)
            #a2 = 1.0 / distr
            thetaprod = self.theta[t, ind, 0] * self.theta[t, ind, 1]
            a0 = (scaleobstacle * dist) / self.eta
            a1 = (scaleobstacle * d) / distr
            expa0 = min(np.exp(a0), OVERFLOW_CAP)
            denom = self.theta[t, ind, 1] + self.theta[t, ind, 0] * expa0
            a2 = 1.0 / denom**2
            a3 = -self.theta[t, ind, 1] * scaleobstacle * \
                (distr * np.eye(DIM) - np.outer(d, d)/distr) * (1.0 / distr**2)
            a4 = ((-thetaprod * scaleobstacle**2) /
                  (self.eta * distr**2)) * expa0 * np.outer(d, d)

            qObs += -self.theta[t, ind, 1] * a1 / denom
            QObs += a2 * (denom * a3 - a4)

            ind += 1

        # Bottom and left boundaries
        for i in range(DIM):
            dist = (x[i] - self.bottom_left[i]) - self.robot_radius
            d = np.zeros(DIM)
            d[i] = 1.0

            # a0 = -self.obstacle_factor * self.scale_factor * self.theta[t, ind, 1]
            thetaprod = self.theta[t, ind, 0] * self.theta[t, ind, 1]
            a0 = scaleobstacle * dist / self.eta
            expa0 = min(np.exp(a0), OVERFLOW_CAP)
            denom = self.theta[t, ind, 1] + self.theta[t, ind, 0] * expa0
            a2 = (thetaprod * scaleobstacle**2) / (self.eta * denom**2)

            qObs += (-self.theta[t, ind, 1] * scaleobstacle / denom) * d
            QObs[i, i] += a2 * expa0

            ind += 1

        # Top and right boundaries
        for i in range(DIM):
            dist = (self.top_right[i] - x[i]) - self.robot_radius
            d = np.zeros(DIM)
            d[i] = -1.0

            thetaprod = self.theta[t, ind, 0] * self.theta[t, ind, 1]
            a0 = scaleobstacle * dist / self.eta
            expa0 = min(np.exp(a0), OVERFLOW_CAP)
            denom = self.theta[t, ind, 1] + self.theta[t, ind, 0] * expa0
            a2 = (thetaprod * scaleobstacle**2) / (self.eta * denom**2)

            qObs += (-self.theta[t, ind, 1] * scaleobstacle / denom) * d
            QObs -= a2 * expa0

            ind += 1

        QObs = regularize(QObs)
        Q[:DIM, :DIM] = QObs + Q[:DIM, :DIM]
        q[:DIM] = qObs - QObs.dot(x[:DIM]) + q[:DIM]
        return q, Q

    def f(self, x, u):
        '''
        Continuous time dynamics
        '''
        x_dot = np.zeros(X_DIM)
        x_dot[0] = 0.5 * (u[0] + u[1])*np.cos(x[2])
        x_dot[1] = 0.5 * (u[0] + u[1])*np.sin(x[2])
        x_dot[2] = (u[1] - u[0])/2.58

        return x_dot

    def g(self, x, u):
        '''
        Discrete time dynamics
        '''
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5*self.dt*k1, u)
        k3 = self.f(x + 0.5*self.dt*k2, u)
        k4 = self.f(x + self.dt*k3, u)

        return x + (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def cost(self, x, u, t, obstacle=True):
        '''
        Cost function
        '''
        cost = 0.0
        if t == 0:
            cost += ((x - self.x_start).T.dot(self.Q.dot(x - self.x_start)))
        cost += ((u - self.u_nominal).T.dot(self.R.dot(u - self.u_nominal)))
        if obstacle:
            cost += self.obstacle_cost(x, t)
        return cost

    def final_cost(self, x):
        '''
        Final cost function
        '''
        cost = ((x - self.x_goal).T.dot(self.Q.dot(x - self.x_goal)))
        return cost

    def quadratize_cost(self, x, u, t, it):
        '''
        Quadratizes the cost around (x, u)
        '''
        Q = np.zeros((X_DIM, X_DIM))
        q = np.zeros(X_DIM)
        if t == 0:
            Q = self.Q
            q = -self.Q.dot(self.x_start)
        else:
            if it < 2:
                Q[2, 2] = self.rot_cost
                q[2] = -self.rot_cost * (np.pi/2)

        R = self.R
        r = -(self.R.dot(self.u_nominal))
        P = np.zeros((U_DIM, X_DIM))

        if self.cost_function == 'exp':
            q, Q = self.quadratize_obstacle_cost(x, q, Q)
        elif self.cost_function == 'hinge':
            q, Q = self.quadratize_obstacle_cost_hinge(x, q, Q)
        elif self.cost_function == 'aula':
            q, Q = self.quadratize_obstacle_cost_aula(x, q, Q, t)
        else:
            raise NotImplementedError

        return P, q, Q, r, R

    def quadratize_final_cost(self, x):
        '''
        Quadratizes the final cost around x
        '''
        Q = self.Q
        q = -(self.Q.dot(self.x_goal))

        return q, Q

    def in_collision(self, x):
        '''
        Checks if given configuration is in collision with any obstacle
        '''
        pos = x[:DIM]
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs.pos) - \
                obs.radius - self.robot_radius
            if dist < 0:
                return True

        return False

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

    def update_obstacle_factor(self, obstacle_factor):
        self.obstacle_factor = obstacle_factor
        return

    def rollout(self, l, L, verbose=True):
        xs = []
        us = []
        x = self.x_start
        xs.append(x)
        for t in range(self.horizon_length):
            if verbose:
                print(bcolors.OKGREEN+str(t) + ' : '+str(x)+bcolors.ENDC)
            u = L[t].dot(x) + l[t]
            x = self.g(x, u)

            if self.in_collision(x) and verbose:
                print(bcolors.FAIL+'In collision at time step '+str(t)+bcolors.ENDC)
            xs.append(x)
            us.append(u)
        if verbose:
            print(bcolors.OKGREEN+str(t) + ' : '+str(x)+bcolors.ENDC)

        return xs, us

    def get_all_obstacle_costs(self, xs):
        costs = np.zeros((self.horizon_length, len(self.obstacles) + 2*DIM, 2))
        t = 0
        # Removing the last one since its the final state (and doesnt have obstacle cost)
        for x in xs[:-1]:
            ind = 0
            for obs in self.obstacles:
                d = x[:DIM] - obs.pos
                distr = np.linalg.norm(d)
                dist = distr - self.robot_radius - obs.radius
                costs[t, ind, 1] = -self.obstacle_factor * \
                    self.scale_factor * dist
                ind += 1
            for i in range(DIM):
                dist = x[i] - self.bottom_left[i] - self.robot_radius
                costs[t, ind, 1] = -self.obstacle_factor * \
                    self.scale_factor * dist
                ind += 1
            for i in range(DIM):
                dist = self.top_right[i] - x[i] - self.robot_radius
                costs[t, ind, 1] = -self.obstacle_factor * \
                    self.scale_factor * dist
                ind += 1
            t += 1

        return costs

    def get_hinge_obstacle_cost(self, xs):
        cost = 0.0
        for x in xs[:-1]:
            for obs in self.obstacles:
                d = x[:DIM] - obs.pos
                distr = np.linalg.norm(d)
                dist = distr - self.robot_radius - obs.radius
                cost += max(0, -self.scale_factor * dist)
            for i in range(DIM):
                dist = x[i] - self.bottom_left[i] - self.robot_radius
                cost += max(0, -self.scale_factor * dist)
            for i in range(DIM):
                dist = self.top_right[i] - x[i] - self.robot_radius
                cost += max(0, -self.scale_factor * dist)

        return self.obstacle_factor * cost

    def get_state_action_cost(self, xs, us):
        cost = 0.0
        ind = 0
        for x in xs[:-1]:
            cost += self.cost(x, us[ind], ind, obstacle=False)
            ind += 1
        cost += self.final_cost(xs[-1])
        return cost

    def get_cost(self, xs, us):
        return self.get_state_action_cost(xs, us) + self.get_hinge_obstacle_cost(xs)
