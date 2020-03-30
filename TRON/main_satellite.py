from TRON.satellite import SatelliteEnv, SX_DIM, SU_DIM
from TRON.ilqr import ilqr
import time
from TRON.utils import bcolors
import numpy as np
from TRON.argsparser import argsparser, get_specific_args
from TRON.plot_utils import plot_cost, plot_satellite
from TRON.main_needle import soft_threshold
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


def main_satellite():
    args = argsparser()
    args['satellite'] = True
    args = get_specific_args(args)
    horizon_length = args['horizon_length']

    # AULA lagrange multipliers
    theta = np.ones((args['horizon_length'], SU_DIM, 2)) * 0.5
    args['theta'] = theta.copy()

    # ADMM dummy variables
    horizon_length = args['horizon_length']
    y = np.zeros((horizon_length, SU_DIM)) * 0.05
    args['y'] = y.copy()
    # ADMM Lagrange multipliers
    lam = np.zeros((horizon_length, SU_DIM))
    args['lam'] = lam.copy()
    rho = args['rho']

    env = SatelliteEnv(args)

    ##################### AULA ########################

    L = [np.zeros((SU_DIM, SX_DIM)) for _ in range(horizon_length)]
    l = [np.ones(SU_DIM) for _ in range(horizon_length)]
    aula_total_costs = []
    aula_times = []

    env.cost_function = 'aula'
    iteration = 0
    outer_iteration = 0
    aula_initial_time = time.time()
    while iteration < args['n_iters']:
        start = time.time()
        x_start = args['x_start']
        u_nominal = np.zeros(SU_DIM)
        it, l, L, totalcost, times = ilqr(env, args['horizon_length'], x_start, u_nominal,
                                          env.g, env.quadratize_final_cost, env.final_cost,
                                          env.quadratize_cost, env.cost, l, L, verbose=True,
                                          maxIter=args['n_iters'] - iteration, terminate=True)
        iteration += it + 1
        end = time.time()
        print(bcolors.OKGREEN+'ITERATION ' + str(iteration) +
              ' aula ilqr. true cost : '+str(totalcost[-1])+' time : '+str(end - start)+bcolors.ENDC)

        # Update multipliers
        xs, us = env.rollout(l, L, verbose=False)
        costs = env.get_all_sparse_control_costs(us)
        max_costs = np.max(costs, axis=2).reshape(horizon_length, SU_DIM, 1)

        theta_new = theta * np.exp((1.0 / args['eta']) * (costs - max_costs))
        sum_theta = np.sum(theta_new, axis=2).reshape(horizon_length, SU_DIM, 1)
        theta = np.divide(theta_new, sum_theta)

        env.update_theta(theta)
        args['theta'] = theta.copy()
        aula_total_costs += totalcost
        aula_times += [x - aula_initial_time for x in times]

        outer_iteration += 1

    xs, us = env.rollout(l, L, verbose=False)
    
    ##################### iLQR ########################

    L = [np.zeros((SU_DIM, SX_DIM)) for _ in range(horizon_length)]
    l = [np.ones(SU_DIM) for _ in range(horizon_length)]

    env.cost_function = 'l1'
    iteration = 0
    start = time.time()
    x_start = args['x_start']
    u_nominal = np.zeros(SU_DIM)
    it, l, L, ilqr_total_costs, ilqr_times = ilqr(env, args['horizon_length'], x_start, u_nominal,
                                                   env.g, env.quadratize_final_cost, env.final_cost,
                                                   env.quadratize_cost, env.cost, l, L, verbose=True,
                                                   maxIter=args['n_iters'], terminate=False)
    end = time.time()
    print(bcolors.OKGREEN+'ITERATION '+str(it+1)+' ilqr. true cost : '+str(ilqr_total_costs[-1])+' time : '+str(end - start)+bcolors.ENDC)

    ilqr_xs, ilqr_us = env.rollout(l, L, verbose=False)
    ilqr_times = [x - start for x in ilqr_times]


    ##################### ADMM #########################
    
    L = [np.zeros((SU_DIM, SX_DIM)) for _ in range(horizon_length)]
    l = [np.ones(SU_DIM) for _ in range(horizon_length)]
    admm_total_costs = []
    admm_times = []

    env.cost_function = 'admm'
    iteration = 0
    outer_iteration = 0
    admm_initial_time = time.time()
    while iteration < args['n_iters']:
        start = time.time()
        x_start = args['x_start']
        u_nominal = np.zeros(SU_DIM)
        it, l, L, totalcost, times = ilqr(env, args['horizon_length'], x_start, u_nominal,
                                          env.g, env.quadratize_final_cost, env.final_cost,
                                          env.quadratize_cost, env.cost, l, L, verbose=True,
                                          maxIter=2, terminate=True)
        iteration += it + 1
        end = time.time()
        print(bcolors.OKGREEN+'ITERATION ' + str(iteration) +
              ' admm ilqr. true cost : '+str(totalcost[-1])+' time : '+str(end - start)+bcolors.ENDC)

        # Update dummy variables
        admm_xs, admm_us = env.rollout(l, L, verbose=False)
        admm_us = np.array(admm_us)
        y = soft_threshold(admm_us + lam / rho, args['alpha'] / rho)
        env.update_y(y)
        args['y'] = y.copy()

        # Update admm lagrange multipliers
        lam = lam + rho * (admm_us - y)
        env.update_lam(lam)
        args['lam'] = lam.copy()

        if outer_iteration == 0:
            scale_lam = args['alpha'] / np.array([np.max(lam[:, i]) for i in range(SU_DIM)])
            lam = lam * scale_lam
            env.update_lam(lam)
            args['lam'] = lam.copy()

        admm_total_costs += totalcost
        admm_times += [x - admm_initial_time for x in times]

        outer_iteration += 1

    admm_xs, admm_us = env.rollout(l, L, verbose=False)

    ####################### PLOTTING ####################
    plt.gcf().set_size_inches([11.16, 7.26])
    plot_satellite(args, us, None, admm_us, subplot=True)
    plot_cost(aula_total_costs, aula_times, ilqr_total_costs, ilqr_times, admm_total_costs, admm_times, subplot=True, log=True, ylim=[1, 10**8])
    filename = 'plot/'+'satellite'+'.png'
    plt.savefig(filename, format='png')
    filename = 'plot/'+'satellite'+'.pdf'
    plt.savefig(filename, format='pdf')
    plt.show()


if __name__ == '__main__':
    main_satellite()
