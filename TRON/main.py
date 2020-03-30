from TRON.diffdrive import DiffDriveEnv, X_DIM, U_DIM, DIM
from TRON.ilqr import ilqr
import time
from TRON.utils import bcolors
import numpy as np
from TRON.plot_utils import plot, plot_cost
from TRON.argsparser import argsparser, get_specific_args
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})


def main():
    args = argsparser()
    args['diffdrive'] = True
    args = get_specific_args(args)
    num_obstacles = 11
    # theta = np.ones((args['horizon_length'], num_obstacles + 2*DIM, 2)) * 0.5
    theta = np.ones((args['horizon_length'], num_obstacles + 2*DIM, 2))
    theta[:, :, 0] = 0.9
    theta[:, :, 1] = 0.1
    args['theta'] = theta.copy()

    env = DiffDriveEnv(args)

    env.add_obstacle(np.array([0, -13.5]), 2.0)
    env.add_obstacle(np.array([10, -5]), 2.0)
    env.add_obstacle(np.array([-9.5, -5]), 2.0)
    env.add_obstacle(np.array([-2, 3]), 2.0)
    env.add_obstacle(np.array([8, 7]), 2.0)
    env.add_obstacle(np.array([11, 20]), 2.0)
    env.add_obstacle(np.array([-12, 8]), 2.0)
    env.add_obstacle(np.array([-11, 21]), 2.0)
    env.add_obstacle(np.array([-1, 16]), 2.0)
    env.add_obstacle(np.array([-11, -19]), 2.0)
    env.add_obstacle(np.array([10 + np.sqrt(2), -15 - np.sqrt(2)]), 2.0)


    ############### AULA ######################
    
    # Initialize the linear policy
    horizon_length = args['horizon_length']
    l = [np.zeros(U_DIM) for _ in range(horizon_length)]
    L = [np.zeros((U_DIM, X_DIM)) for _ in range(horizon_length)]
    total_costs = []
    aula_times = []

    # run ilqr with augmentation
    # aula cost functions
    # else:
    env.cost_function = 'aula'
    iteration = 0
    outer_iteration = 0
    aula_initial_time = time.time()
    # for iteration in range(args['n_iters']):
    while iteration < args['n_iters']:
        # if outer_iteration % 5 == 0 and outer_iteration != 0:
        #     args['eta'] = 0.9 * args['eta']
        #     # args['eta'] = 0.5 * args['eta']
        #     env.update_eta(args['eta'])
        start = time.time()
        it, l, L, total_cost, end_times = ilqr(env, env.horizon_length, env.x_start, env.u_nominal, env.g,
                                               env.quadratize_final_cost, env.final_cost, env.quadratize_cost, env.cost, l, L, verbose=True, maxIter=None, terminate=True)
        iteration += it + 1
        end = time.time()

        print(bcolors.OKGREEN+'ITERATION ' + str(iteration) +
              ' ilqr. Num iter : '+str(it)+' true cost: '+str(total_cost[-1])+' time : '+str(end - start)+bcolors.ENDC)
        # Update multipliers
        xs, us = env.rollout(l, L, verbose=False)
        costs = env.get_all_obstacle_costs(xs)
        max_costs = np.max(costs, axis=2).reshape(
            args['horizon_length'], num_obstacles + 2*DIM, 1)

        theta_new = theta * np.exp((1.0/args['eta']) * (costs - max_costs))
        sum_theta = np.sum(theta_new, axis=2).reshape(
            args['horizon_length'], num_obstacles + 2*DIM, 1)
        theta = np.divide(theta_new, sum_theta)

        env.update_theta(theta)
        args['theta'] = theta.copy()
        total_costs += total_cost
        aula_times += [x - aula_initial_time for x in end_times]
        if args['plot_debug']:
            plot(args, env, xs)
            plt.show()

        outer_iteration += 1

    xs, us = env.rollout(l, L, verbose=True)


    ############### iLQR ######################

    # Reinitialize the linear policy
    l = [np.zeros(U_DIM) for _ in range(horizon_length)]
    L = [np.zeros((U_DIM, X_DIM)) for _ in range(horizon_length)]

    # run ilqr without augmentation
    # Non-aula cost functions
    # if args['cost_function'] != 'aula':
    env.cost_function = 'hinge'
    start = time.time()
    it, l, L, ilqr_total_costs, ilqr_times = ilqr(env, env.horizon_length, env.x_start, env.u_nominal, env.g, env.quadratize_final_cost,
                                                  env.final_cost, env.quadratize_cost, env.cost, l, L, verbose=True, maxIter=args['n_iters'], terminate=False)
    end = time.time()
    ilqr_times = [x - start for x in ilqr_times]
    print(bcolors.OKGREEN+'ilqr Num iter : '+str(it) + ' true cost: '+str(ilqr_total_costs[-1]) +
          ' time : '+str(end - start)+bcolors.ENDC)
    ilqr_xs, ilqr_us = env.rollout(l, L, verbose=False)
    
    ############## PLOTTING #################

    plt.gcf().set_size_inches([11.16, 7.26])
    plot(args, env, xs, ilqr_xs, subplot=True)
    if args['cost_function'] == 'aula':
        plot_cost(total_costs, aula_times, ilqr_total_costs, ilqr_times, subplot=True)
    # plt.show()
    filename = 'plot/'+args['cost_function']+'_' + \
        str(args['n_iters'])+'_'+str(num_obstacles)+'.png'
    plt.savefig(filename, format='png')
    filename = 'plot/'+args['cost_function']+'_' + \
        str(args['n_iters'])+'_'+str(num_obstacles)+'.pdf'
    plt.savefig(filename, format='pdf')


if __name__ == '__main__':
    main()
