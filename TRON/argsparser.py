import argparse
import numpy as np
from TRON.diffdrive import X_DIM, U_DIM
from TRON.needle import NX_DIM, NU_DIM
from TRON.satellite import SX_DIM, SU_DIM


def argsparser():
    parser = argparse.ArgumentParser()

    env_args = parser.add_argument_group('Environment arguments')
    env_args.add_argument('--diffdrive', action='store_true')
    env_args.add_argument('--needle', action='store_true')

    cost_args = parser.add_argument_group('Cost function arguments')
    cost_args.add_argument('--cost_function', type=str,
                           default='aula', help="Should be one of {exp, hinge, aula, l1}")
    cost_args.add_argument('--sparse_cost_coeff', type=float, default=10.0,
                           help='Scale of the sparse control cost')

    run_args = parser.add_argument_group('Run arguments')
    run_args.add_argument('--n_iters', type=int, default=200,
                          help='Number of iLQR iterations')

    aula_args = parser.add_argument_group('AuLa arguments')
    # aula_args.add_argument('--theta_lr', type=float, default=0.05, help='Learning rate for lagrange multipliers')
    aula_args.add_argument('--eta', type=float, default=10.0,
                           help='Exponentiated gradient descent inverse learning rate')

    debug_args = parser.add_argument_group('debug arguments')
    debug_args.add_argument('--plot_debug', action='store_true')

    exp_args = parser.add_argument_group('experiment arguments')
    exp_args.add_argument('--exp', action='store_true')

    args = vars(parser.parse_args())
    return args


def get_specific_args(args):
    if args['diffdrive']:
        args['horizon_length'] = 150
        args['dt'] = 1.0/6.0
        args['rot_cost'] = 0.4
        args['obstacle_factor'] = 10.0
        args['scale_factor'] = 1.0
        args['robot_radius'] = 3.35/2.0

        args['Q'] = 30 * np.eye(X_DIM)
        args['R'] = 0.6 * np.eye(U_DIM)

        args['x_goal'] = np.array([0, 25, np.pi])
        args['x_start'] = np.array([0, -25, np.pi])

        args['u_nominal'] = np.array([2.5, 2.5])

        args['bottom_left'] = np.array([-20, -30])
        args['top_right'] = np.array([20, 30])

        if args['cost_function'] in ['l1']:
            raise Exception(
                'Cost function should be one of {exp, hinge, aula}')

        if args['exp']:
            # Reproducing paper experiments
            args['eta'] = 1.0
            args['n_iters'] = 200

    elif args['needle']:

        args['horizon_length'] = 100
        args['Q'] = 100 * np.eye(NX_DIM)
        R = np.eye(NU_DIM)
        R[0, 0] = 1.5
        R[1, 1] = 0.5
        # R[1, 1] = 1.0  # FIX: Changed this from the previous line
        R[2, 2] = 1.0
        args['R'] = R

        args['kmax'] = 2.0
        # u is a 3D vector where the elements are [v, w, k]
        args['u_nominal'] = np.zeros(NU_DIM)
        args['u_nominal'][:3] = np.array([0.9, 0.0, 0.5*args['kmax']])

        args['dt'] = 0.1
        args['rot_cost'] = 1.0
        args['obstacle_factor'] = 1.0
        args['scale_factor'] = 1.0
        args['robot_radius'] = 0.0

        args['x_start'] = np.zeros(NX_DIM)
        args['x_goal'] = np.zeros(NX_DIM)
        args['x_start'][:3] = np.array([0.0, 0.0, -2.8])
        args['x_goal'][:3] = np.array([-1.5, -1.5, 2.8])

        args['bottom_left'] = np.array([-3.0, -3.0, -3.0])
        args['top_right'] = np.array([3.0, 3.0, 3.0])

        if args['cost_function'] in ['exp', 'hinge']:
            raise Exception('Cost function should be one of {l1, aula}')

        if args['exp']:
            # Reproducing paper experiments
            args['sparse_cost_coeff'] = 10.0
            args['n_iters'] = 100
            args['eta'] = 0.4
            args['rho'] = 100.0  # 100

    elif args['satellite']:
        args['mu'] = 3.99e14
        args['a'] = 6731e3
        args['alt'] = 5e5
        args['orbit_radius'] = args['a'] + args['alt']
        args['horizon_length'] = 100
        args['tf'] = 6000
        args['dt'] = args['tf'] / (args['horizon_length'] - 1)
        args['r0'] = np.array([args['orbit_radius'], 0., 0.])
        args['v0'] = np.array([0., np.sqrt(args['mu'] / np.linalg.norm(args['r0'])), 0.])
        # args['x_start'] = np.concatenate([[0., -100., 0.], args['r0'], np.zeros(3), args['v0']])
        # args['x_start'] = np.zeros(SX_DIM)
        args['x_start'] = np.array([-0.9784410296423234, -0.20647857489084415, -0.004444054850698648, -0.960228995455973, 0.2781013279142361, -0.02489834729357783])
        # args['x_start'] = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        # args['Qf'] = 1000. * np.diag(np.array([1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.]))
        args['Qf'] = 100.0 * np.eye(SX_DIM)

        args['m_ego'] = 4.6
        args['m_target'] = args['m_ego']

        args['u_min'] = -5e-3
        args['u_max'] = 5e-3
        args['Q'] = 0.0 * np.eye(SX_DIM)
        args['rho'] = 0.1
        args['R'] = 1 * np.eye(SU_DIM)

        args['omega'] = np.array([0., 0., 2 * np.pi / ((23 * 60 + 56) * 60 + 4)])
        args['A_ego'] = 0.01
        args['A_target'] = 0.01
        args['Cd_ego'] = 1.0
        args['Cd_target'] = 1.0

        args['alpha'] = 1.0
        args['rho'] = 0.1
        args['stopping_criterion'] = 0.0045
        args['orbit_radius'] = 7.231e6

        if args['exp']:
            # Reproducing paper experiments
            args['alpha'] = 100.0
            args['rho'] = 1.0
            args['n_iters'] = 300
            args['eta'] = 0.1
        
    return args

