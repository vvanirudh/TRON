from TRON.needle import NeedleEnv, NX_DIM, NU_DIM
from TRON.ilqr import ilqr
import time
from TRON.utils import bcolors
import numpy as np
from TRON.argsparser import argsparser, get_specific_args
from TRON.plot_utils import plot_cost, plot_needle
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})


def soft_threshold(s, tau):
    y = np.zeros_like(s)
    y[s > tau] = s[s > tau] - tau
    y[s < -tau] = s[s < -tau] + tau

    return y


def main_needle():
    args = argsparser()
    args['needle'] = True
    args = get_specific_args(args)
    # num_obstacles = 1
    # AULA Lagrange multipliers
    theta = np.ones((args['horizon_length'], 2))
    theta[:, 0] = 0.55
    theta[:, 1] = 0.45
    args['theta'] = theta.copy()

    # ADMM Dummy variables
    horizon_length = args['horizon_length']
    y = np.zeros(horizon_length)
    args['y'] = y.copy()
    # ADMM Lagrange multipliers
    lam = np.zeros(horizon_length)
    args['lam'] = lam.copy()
    # ADMM Penalty parameter
    rho = args['rho']
    
    env = NeedleEnv(args)

    # env.add_obstacle(np.array([-1, 1, 0]), 0.4, 2)
    # env.add_obstacle(np.array([1, 1, 0]), 0.4, 2)
    # env.add_obstacle(np.array([1, -1, 0]), 0.4, 2)
    # env.add_obstacle(np.array([-1, -1, 0]), 0.4, 2)

    # env.add_obstacle(np.array([-1, 0, 0]), 0.4, 1)
    # env.add_obstacle(np.array([1, 0, 1]), 0.4, 1)
    # env.add_obstacle(np.array([1, 0, -1]), 0.4, 1)

    # env.add_obstacle(np.array([0, 0, -0.5]), 0.5, 1)
    # env.add_obstacle(np.array([0, 0, 0.5]), 0.5, 0)

    # env.add_obstacle(np.array([0, 1, 1]), 0.4, 0)
    # env.add_obstacle(np.array([0, 1, -1]), 0.4, 0)
    # env.add_obstacle(np.array([0, -1, -1]), 0.4, 0)


    ################## AULA ####################
    
    # Initialize linear policy
    horizon_length = args['horizon_length']
    L = [np.zeros((NU_DIM, NX_DIM)) for _ in range(horizon_length)]
    t_u_nominal = args['u_nominal'].copy()
    t_u_nominal[2] = 0.0
    l = [t_u_nominal.copy() / 2.0 for _ in range(horizon_length)]
    total_costs = []
    aula_times = []

    # run ilqr with augmentation
    # aula cost functions
    env.cost_function = 'aula'
    iteration = 0
    outer_iteration = 0
    aula_initial_time = time.time()
    while iteration < args['n_iters']:
        # if outer_iteration % 5 == 0 and outer_iteration != 0:
        #     args['eta'] = 0.9 * args['eta']
        #     env.update_eta(args['eta'])
        #     print(bcolors.WARNING+'Updated eta to ' +
        #           str(args['eta'])+bcolors.ENDC)

        start = time.time()
        x_start = args['x_start']
        u_nominal = args['u_nominal']
        it, l, L, totalcost, times = ilqr(env, horizon_length, x_start, u_nominal, env.g,
                                   env.quadratize_final_cost, env.final_cost, env.quadratize_cost, env.cost, l, L, verbose=True, maxIter=args['n_iters'] - iteration, terminate=True)
        iteration += it + 1
        end = time.time()

        print(bcolors.OKGREEN+'ITERATION ' + str(iteration) +
              ' augmented ilqr. true cost : '+str(totalcost[-1])+' time : '+str(end - start)+bcolors.ENDC)

        # Update multipliers
        xs, us = env.rollout(l, L, verbose=False)
        # us = [p[1] for p in plan]
        costs = env.get_all_sparse_control_costs(us)
        max_costs = np.max(costs, axis=1).reshape(horizon_length, 1)

        theta_new = theta * \
            np.exp((1.0 / args['eta']) * (costs - max_costs))
        sum_theta = np.sum(theta_new, axis=1).reshape(horizon_length, 1)
        theta = np.divide(theta_new, sum_theta)

        env.update_theta(theta)
        args['theta'] = theta.copy()
        total_costs += totalcost
        aula_times += [x - aula_initial_time for x in times]

        outer_iteration += 1

    xs, us = env.rollout(l, L, verbose=False)


    ################# iLQR ######################
        
    # Reinitialize the linear policy
    L = [np.zeros((NU_DIM, NX_DIM)) for _ in range(horizon_length)]
    t_u_nominal = args['u_nominal'].copy()
    t_u_nominal[2] = 0.0
    l = [t_u_nominal.copy() / 2.0 for _ in range(horizon_length)]

    # run ilqr without augmentation
    # non-aula cost functions
    env.cost_function = 'l1'
    iteration = 0
    start = time.time()
    x_start = args['x_start']
    u_nominal = args['u_nominal']
    it, l, L, ilqr_total_costs, ilqr_times = ilqr(env, horizon_length, x_start, u_nominal, env.g,
                                      env.quadratize_final_cost, env.final_cost, env.quadratize_cost, env.cost, l, L, verbose=True, maxIter=args['n_iters'], terminate=False)
    end = time.time()    
    print(bcolors.OKGREEN+'ITERATION ' + str(iteration) +
          ' ilqr. true cost : '+str(ilqr_total_costs[-1])+' time : '+str(end - start)+bcolors.ENDC)
    ilqr_xs, ilqr_us = env.rollout(l, L, verbose=False)
    iteration += it
    ilqr_times = [x - start for x in ilqr_times]

    #################### ADMM #######################

    # Reinitialize the linear policy
    L = [np.zeros((NU_DIM, NX_DIM)) for _ in range(horizon_length)]
    t_u_nominal = args['u_nominal'].copy()
    t_u_nominal[2] = 0.0
    l = [t_u_nominal.copy() / 2.0 for _ in range(horizon_length)]
    admm_total_costs = []
    admm_times = []

    # run admm
    env.cost_function = 'admm'
    iteration = 0
    outer_iteration = 0
    admm_initial_time = time.time()
    while iteration < args['n_iters']:
        start = time.time()
        x_start = args['x_start']
        u_nominal = args['u_nominal']
        it, l, L, totalcost, times = ilqr(env, horizon_length, x_start, u_nominal, env.g,
                                          env.quadratize_final_cost, env.final_cost,
                                          env.quadratize_cost, env.cost, l, L, verbose=True,
                                          maxIter=args['n_iters'] - iteration, terminate=True)
        iteration += it + 1
        end = time.time()
        print(bcolors.OKGREEN+'ITERATION ' + str(iteration) +
              ' admm ilqr. true cost : '+str(totalcost[-1])+' time : '+str(end - start)+bcolors.ENDC)

        # Update dummy variables
        admm_xs, admm_us = env.rollout(l, L, verbose=False)
        us_1 = np.array([u[1] for u in admm_us])
        y = soft_threshold(us_1 + lam/rho, args['sparse_cost_coeff'] / rho)
        env.update_y(y)
        args['y'] = y.copy()

        # Update admm lagrange multipliers
        lam = lam + rho * (us_1 - y)
        env.update_lam(lam)
        args['lam'] = lam.copy()

        admm_total_costs += totalcost
        admm_times += [x - admm_initial_time for x in times]

        outer_iteration += 1

    admm_xs, admm_us = env.rollout(l, L, verbose=False)

    #################### PLOTTING ####################

    plt.gcf().set_size_inches([11.16, 7.26])
    plot_needle(args, us, admm_us, subplot=True)
    plot_cost(total_costs, aula_times, ilqr_total_costs, ilqr_times, admm_total_costs, admm_times, subplot=True, log=False)
    filename = 'plot/'+'needle'+'.png'
    plt.savefig(filename, format='png')
    filename = 'plot/'+'needle'+'.pdf'
    plt.savefig(filename, format='pdf')
    plt.show()


if __name__ == '__main__':
    main_needle()
