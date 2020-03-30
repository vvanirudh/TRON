import torch
import numpy as np
from torch import autograd
import argparse
import matplotlib.pyplot as plt
import matplotlib

from TRON.newton import newton_step
from TRON.utils import bcolors

torch.set_default_tensor_type('torch.DoubleTensor')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})

N = 1000
DIM = 2
HALF_DIM = int(DIM/2)

np.random.seed(0)
X = np.random.randn(N, DIM)
X[:, :HALF_DIM] = X[:, :HALF_DIM] * 10
X[:, HALF_DIM:] = X[:, HALF_DIM:] * 0.1
X = torch.from_numpy(X)

y = np.random.randn(N)
y = torch.from_numpy(y)

lam = 0.05


def f(w):
    return torch.mean((torch.mv(X, w) - y)**2)


def g_1(w, i):
    return -lam*w[i]


def g_2(w, i):
    return lam*w[i]


def obj_implicit(w, theta, eta):
    # Do the log sum exp trick here!
    obj = f(w)
    for i in range(DIM):
        a_i = torch.max(g_1(w, i), g_2(w, i))
        g_i = torch.exp((1./eta) * torch.stack((g_1(w, i)-a_i, g_2(w, i)-a_i)))
        obj += eta * torch.log(torch.dot(theta[i, :], g_i)) + a_i
    # a = torch.max(g_1(w), g_2(w))

    # g = torch.exp((1./eta) * torch.stack((g_1(w)-a, g_2(w)-a)))
    # return f(w)+eta*torch.log(torch.dot(theta, g)) + a
    return obj


def obj_explicit(w, theta):
    obj = f(w)
    for i in range(DIM):
        g_i = torch.stack((g_1(w, i), g_2(w, i)))
        obj += torch.dot(theta[i, :], g_i)
    # g = torch.stack((g_1(w), g_2(w)))
    # return f(w) + torch.dot(theta, g)
    return obj


def calc_obj(w):
    # objval = f(w) + torch.max(g_1(w), g_2(w))
    objval = f(w)
    for i in range(DIM):
        objval += torch.max(g_1(w, i), g_2(w, i))
    return objval


def calc_grad(output, var):
    grad, = autograd.grad(output, var)
    return grad[0]


def solve_implicit_game(args, theta_0, w_0):
    n_iters, eta = args.n_iters, args.eta
    theta, w = theta_0.clone(), w_0.clone()

    w_vals, theta_vals, obj_vals = [], [], []
    eta_vals = []
    objval = calc_obj(w)

    w_vals.append(w.numpy())
    theta_vals.append(theta.numpy())
    obj_vals.append(objval.item())
    eta_vals.append(eta)

    print(bcolors.OKBLUE+'Initial Objective Value is ' +
          str(objval.item())+bcolors.ENDC)

    for it in range(n_iters):
        print("theta: {}".format(theta))
        eta = 0.9 * eta
        # Convert w to variable
        w_var = autograd.Variable(w, requires_grad=True)
        theta_var = autograd.Variable(theta, requires_grad=True)

        # 1. First update w
        # calculate the objective
        obj = obj_implicit(w_var, theta_var, eta)
        if np.isnan(obj.item()):
            print(bcolors.FAIL+'Obj became NAN'+bcolors.ENDC)
            import ipdb
            ipdb.set_trace()
        # calculate grad
        # grad = calc_grad(obj, w_var)
        # Apply newton step
        w_var_new = newton_step(
            args, obj, w_var, lambda x: obj_implicit(x, theta_var, eta))
        w_new = w_var_new.data

        # 2. Next update theta
        theta_new = theta.clone()
        for i in range(DIM):
            a = torch.max(g_1(w_new, i), g_2(w_new, i))
            theta_new[i, :] = theta[i, :] * \
                torch.exp(
                    (1./eta) * torch.stack((g_1(w_new, i)-a, g_2(w_new, i)-a)))
            theta_new[i, :] = theta_new[i, :] / torch.sum(theta_new[i, :])
            if np.isnan(theta_new[i, 0].item()):
                print(bcolors.FAIL+'Theta became NAN'+bcolors.ENDC)
                import ipdb
                ipdb.set_trace()

        # Calculate objective value
        objval = calc_obj(w_new)
        w_vals.append(w_new.numpy())
        theta_vals.append(theta_new.numpy())
        obj_vals.append(objval.item())
        eta_vals.append(eta)

        print(bcolors.OKBLUE+'Objective value at iteration ' +
              str(it)+' is '+str(objval.item())+bcolors.ENDC)

        # Update estimates
        w = w_new.clone()
        theta = theta_new.clone()

    print(bcolors.OKGREEN+'Final objective value is ' +
          str(objval.item())+bcolors.ENDC)
    print(bcolors.OKGREEN+'Final theta value is ' +
          str(theta.numpy())+bcolors.ENDC)
    # print(bcolors.OKGREEN+'For final w, g_1 is greater than g_2? : ' +
    #       str((g_1(w) > g_2(w)).item())+bcolors.ENDC)

    return w_vals, theta_vals, obj_vals, eta_vals


def solve_explicit_game(args, theta_0, w_0):
    n_iters, eta = args.n_iters, args.eta
    theta, w = theta_0.clone(), w_0.clone()

    w_vals, theta_vals, obj_vals = [], [], []
    eta_vals = []
    objval = calc_obj(w)

    # w_vals.append(w.numpy())
    # theta_vals.append(theta.numpy())
    # obj_vals.append(objval.item())
    # eta_vals.append(eta)

    print(bcolors.OKBLUE+'Initial Objective Value is ' +
          str(objval.item())+bcolors.ENDC)

    for it in range(n_iters):
        print("theta: {}".format(theta))
        eta = args.eta  # * np.sqrt(it+1)

        w_var = autograd.Variable(w, requires_grad=True)
        theta_var = autograd.Variable(theta, requires_grad=True)

        # first update w:
        obj = obj_explicit(w_var, theta_var)
        w_var_new = newton_step(
            args, obj, w_var, lambda x: obj_explicit(x, theta_var))
        w_new = w_var_new.data

        # 2. update theta
        theta_new = theta.clone()
        for i in range(DIM):
            a = torch.max(g_1(w_new, i), g_2(w_new, i))
            theta_new[i, :] = theta[i, :] * \
                torch.exp(
                    (1./eta) * torch.stack((g_1(w_new, i)-a, g_2(w_new, i)-a)))
            theta_new[i, :] = theta_new[i, :] / torch.sum(theta_new[i, :])
            if np.isnan(theta_new[i, 0].item()) or np.isnan(theta_new[i, 1].item()):
                print(bcolors.FAIL+'Theta became NAN'+bcolors.ENDC)
                import ipdb
                ipdb.set_trace()

        # Calculate objective value
        objval = calc_obj(w_new)
        w_vals.append(w_new.numpy())
        theta_vals.append(theta_new.numpy())
        obj_vals.append(objval.item())
        eta_vals.append(eta)

        print(bcolors.OKBLUE+'Objective value at iteration ' +
              str(it)+' is '+str(objval.item())+bcolors.ENDC)

        # Update estimates
        w = w_new.clone()
        theta = theta_new.clone()

    print(bcolors.OKGREEN+'Final objective value is ' +
          str(objval.item())+bcolors.ENDC)
    print(bcolors.OKGREEN+'Final theta value is ' +
          str(theta.numpy())+bcolors.ENDC)
    print(bcolors.OKGREEN+'For final w, g_1 is greater than g_2? : ' +
          str((g_1(w) > g_2(w)).item())+bcolors.ENDC)

    return w_vals, theta_vals, obj_vals, eta_vals


def solve_gd(args, w_0):
    n_iters = args.n_iters
    lr = args.lr
    w = w_0.clone()

    w_vals, obj_vals = [], []
    lr_vals = []
    objval = calc_obj(w)

    w_vals.append(w.numpy())
    obj_vals.append(objval.item())
    lr_vals.append(lr)

    print(bcolors.OKBLUE+'Initial Objective Value is ' +
          str(objval.item())+bcolors.ENDC)

    for it in range(n_iters):
        lr = args.lr / np.sqrt(it + 1)
        # Convert w to variable
        w_var = autograd.Variable(w, requires_grad=True)
        # Compute gradient w.r.t w
        obj = calc_obj(w_var)

        grad = calc_grad(obj, w_var)

        # Update w
        w_new = w - lr * grad.data

        # Calculate objective value
        objval = calc_obj(w_new)
        w_vals.append(w_new.numpy())
        obj_vals.append(objval.item())
        lr_vals.append(lr)

        print(bcolors.OKBLUE+'Objective value at iteration ' +
              str(it)+' is '+str(objval.item())+bcolors.ENDC)

        # Update estimates
        w = w_new

    return w_vals, obj_vals, lr_vals


def solve_newton(args, w_0):
    n_iters = args.n_iters
    w = w_0.clone()

    w_vals, obj_vals = [], []
    objval = calc_obj(w)

    w_vals.append(w.numpy())
    obj_vals.append(objval.item())

    print(bcolors.OKBLUE+'Initial Objective Value is ' +
          str(objval.item())+bcolors.ENDC)

    for it in range(n_iters):
        # Convert w to variable
        w_var = autograd.Variable(w, requires_grad=True)
        # Compute gradient w.r.t w
        obj = calc_obj(w_var)

        w_var_new = newton_step(
            args, obj, w_var, calc_obj)

        # Update w
        w_new = w_var_new.data

        # Calculate objective value
        objval = calc_obj(w_new)
        w_vals.append(w_new.numpy())
        obj_vals.append(objval.item())

        print(bcolors.OKBLUE+'Objective value at iteration ' +
              str(it)+' is '+str(objval.item())+bcolors.ENDC)

        # Update estimates
        w = w_new

    return w_vals, obj_vals


def main():
    parser = argparse.ArgumentParser()

    run_args = parser.add_argument_group('Run arguments')
    run_args.add_argument('--n_iters', type=int,
                          default=100, help='Number of iterations')

    obj_args = parser.add_argument_group('Objective arguments')
    obj_args.add_argument('--eta', type=float, default=1,
                          help='Coefficient on the KL penalty for theta objective')

    newton_args = parser.add_argument_group('Newton descent arguments')
    newton_args.add_argument('--alpha', type=float, default=0.2,
                             help='Alpha parameter in newton backtracking line search')
    newton_args.add_argument('--beta', type=float, default=0.5,
                             help='Beta parameter in newton backtracking line search')
    newton_args.add_argument('--cgsteps', type=int, default=10,
                             help='Number of conjugate gradient iterations to calculate newton decrement')

    gd_args = parser.add_argument_group('gradient descent arguments')
    gd_args.add_argument('--lr', type=float, default=5e-4,
                         help='Learning rate of gradient descent')

    exp_args = parser.add_argument_group('Experiment arguments')
    exp_args.add_argument('--implicit', action='store_true',
                          help='Implicit game. all enabled by default')
    exp_args.add_argument('--explicit', action='store_true',
                          help='Explicit game. all enabled by default')
    exp_args.add_argument('--gd', action='store_true',
                          help='Gradient Descent. all enabled by default')
    exp_args.add_argument('--newton', action='store_true',
                          help='newton method. all enabled by default')

    args = parser.parse_args()

    if not (args.implicit or args.explicit or args.gd or args.newton):
        args.implicit = True
        args.explicit = False
        args.gd = True
        args.newton = True

    w_0 = torch.ones(DIM) * 0.01
    theta_0 = torch.ones(DIM, 2) * 0.5
    # theta_0[0] = 0.0001
    # theta_0[1] = 0.9999

    if args.implicit:
        print
        print(bcolors.HEADER+'IMPLICIT VERSION'+bcolors.ENDC)
        implicit_w_vals, implicit_theta_vals, implicit_obj_vals, implicit_eta_vals = solve_implicit_game(
            args, theta_0, w_0)

    if args.explicit:
        print
        print(bcolors.HEADER+'EXPLICIT VERSION'+bcolors.ENDC)
        explicit_w_vals, explicit_theta_vals, explicit_obj_vals, explicit_eta_vals = solve_explicit_game(
            args, theta_0, w_0)

    if args.gd:
        print
        print(bcolors.HEADER+'GRADIENT DESCENT'+bcolors.ENDC)
        gd_w_vals, gd_obj_vals, gd_lr_vals = solve_gd(args, w_0)

    if args.newton:
        print
        print(bcolors.HEADER+'NEWTON METHOD'+bcolors.ENDC)
        newton_w_vals, newton_obj_vals = solve_newton(args, w_0)

    plt.style.use('seaborn-whitegrid')
    # plt.style.use('dark_background')
    plt.gcf().set_size_inches([11.16, 7.26])

    xaxis = np.arange(args.n_iters+1)

    plt.subplot(1, 2, 1)
    if args.implicit:
        plt.plot(xaxis, implicit_obj_vals, color='blue',
                 linestyle='-', label='\\textsc{Tron}', linewidth=3)
    if args.explicit:
        plt.plot(xaxis, explicit_obj_vals, color='red',
                 linestyle='--', label='Explicit Newton Game')
    if args.gd:
        plt.plot(xaxis, gd_obj_vals, color='green',
                 linestyle='-', label='Subgradient Method', linewidth=3)
    if args.newton:
        plt.plot(xaxis, newton_obj_vals, color='red',
                 linestyle='-', label='Newton Method', linewidth=3)

    plt.xlabel('Iterations')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.title('Plot of objective vs iterations')

    plt.subplot(1, 2, 2)
    if args.implicit:
        implicit_theta_vals = np.array(implicit_theta_vals)
        plt.plot(xaxis, implicit_theta_vals[:, 0, 0],
                 color='steelblue', linestyle='-', label=r'$\theta_0$', linewidth=3)
        plt.plot(xaxis, implicit_theta_vals[:, 0, 1],
                 color='steelblue', linestyle='-.', label=r'$\bar{\theta}_0$', linewidth=3)
        plt.plot(xaxis, implicit_theta_vals[:, 1, 0],
                 color='peru', linestyle='-', label=r'$\theta_1$', linewidth=3)
        plt.plot(xaxis, implicit_theta_vals[:, 1, 1],
                 color='peru', linestyle='-.', label=r'$\bar{\theta}_1$', linewidth=3)
    if args.explicit:
        explicit_theta_vals = np.array(explicit_theta_vals)
        plt.plot(xaxis, explicit_theta_vals[:, 0],
                 color='red', linestyle='-', label='Explicit theta0')
        plt.plot(xaxis, explicit_theta_vals[:, 1],
                 color='red', linestyle=':', label='Explicit theta1')

    plt.xlabel('Iterations')
    plt.ylabel('Theta $\\theta$')
    plt.legend()
    plt.title('Plot of $\\theta$ vs iterations')

    # plt.subplot(3, 1, 3)
    # if args.implicit:
    #     implicit_eta_vals = np.array(implicit_eta_vals)
    #     plt.plot(xaxis, implicit_eta_vals, color='green',
    #              linestyle='-', label='Implicit eta')
    # if args.explicit:
    #     explicit_eta_vals = np.array(explicit_eta_vals)
    #     plt.plot(xaxis, explicit_eta_vals, color='red',
    #              linestyle='-', label='Explicit eta')

    # plt.xlabel('Iterations')
    # plt.ylabel('eta value')
    # plt.yscale('log')
    # plt.legend()

    plt.gcf().set_size_inches([11.16, 8.26])

    plot_name = 'plot/plot_'
    if args.implicit:
        plot_name += 'implicit_'+str(args.eta)+'_'
    if args.explicit:
        plot_name += 'explicit_'+str(args.eta)+'_'
    if args.gd:
        plot_name += 'gd_'+str(args.lr)+'_'
    plot_name += str(args.n_iters)+'_'+str(DIM)
    plot_name += '.png'
    plt.savefig(plot_name, format='png')
    plot_name = 'plot/lasso.pdf'
    plt.savefig(plot_name, format='pdf')

    print(implicit_w_vals[-1])

    plt.show()


if __name__ == '__main__':
    main()
