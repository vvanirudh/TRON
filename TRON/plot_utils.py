import matplotlib.pyplot as plt
import numpy as np


def plot_env(env):
    '''
    Helper function to plot the environment
    '''
    # Get bounds of the environment
    xaxis = np.arange(env.bottom_left[0], env.top_right[0], 0.1)
    yaxis = np.arange(env.bottom_left[1], env.top_right[1], 0.1)

    # Set plt limits
    plt.xlim([xaxis[0], xaxis[-1]])
    plt.ylim([yaxis[0], yaxis[-1]])

    # Plot obstacles
    for obs in env.obstacles:
        plot_obstacle(obs)

    # Plot start and goal pose
    #plot_vector(env.x_start, env.robot_radius)
    #plot_vector(env.x_goal, env.robot_radius)
    plot_marker(env.x_start, 'black')
    plot_marker(env.x_goal, 'green')


def plot_obstacle(obs):
    '''
    Plot an obstacle as a disk
    '''
    pos = obs.pos
    radius = obs.radius

    circle = plt.Circle(pos, radius, color='black')

    ax = plt.gca()
    ax.add_artist(circle)


def plot_vector(pose, length):
    '''
    Plot the robot pose
    '''
    x_1, y_1 = pose[0], pose[1]
    theta = pose[2]

    x_2 = x_1 + length*np.cos(theta)
    y_2 = y_1 + length*np.sin(theta)

    ax = plt.gca()
    ax.quiver(x_1, y_1, x_2, y_2, angles='xy', scale_units='xy', scale=1)
    plt.draw()


def plot_marker(pt, color):
    '''
    Plots a marker
    '''
    plt.plot(pt[0], pt[1], marker='D', color=color)


def plot_path(path, robot_radius, color):
    '''
    Plot the robot path (2D)
    '''
    path_np = np.array(path)
    path_np = path_np[:, :2]  # Get only x, y

    ax = plt.gca()
    T = len(path) * 2.0

    for t in range(len(path)):
        circle = plt.Circle(path_np[t, :], robot_radius, color=color, alpha=(t+1)/T)
        ax.add_artist(circle)


def plot(args, env, xs, ilqr_xs, subplot=False):
    if subplot:
        plt.subplot(1, 2, 1)
    plot_env(env)
    plot_path(xs, args['robot_radius'], color='b')
    plot_path(ilqr_xs, args['robot_radius'], color='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of robot path')


def plot_cost(costs, times, ilqr_costs=None, ilqr_times=None, admm_costs=None, admm_times=None, subplot=False, log=True, ylim=None):
    if subplot:
        plt.subplot(1, 2, 2)
    plt.plot(times, costs, 'b-', label='\\textsc{Tron}', linewidth=3)
    if ilqr_costs:        
        plt.plot(ilqr_times, ilqr_costs, 'r-', label='\\textsc{Ilqr}', linewidth=3)
    if admm_costs:
        plt.plot(admm_times, admm_costs, 'g-', label='\\textsc{Admm}', linewidth=3)
    plt.legend()
    plt.xlabel('Time (in seconds)')
    plt.ylabel('Cost of trajectory')
    if log:
        plt.yscale('log')
    if ylim:
        plt.ylim(ylim)
    plt.title('Plot of trajectory cost vs time')

def plot_needle(args, us, admm_us, subplot=False):
    if subplot:
        plt.subplot(1, 2, 1)

    plt.plot(range(len(us)), [u[1] for u in us], 'b-', label='\\textsc{Tron} $\omega$', linewidth=3)
    # plt.plot(range(len(ilqr_us)), [u[1] for u in ilqr_us], 'r-')
    plt.plot(range(len(admm_us)), [u[1] for u in admm_us], 'g-', label='\\textsc{Admm} $\omega$', linewidth=3)

    # plt.ylim([-5, 5])

    plt.xlabel('Time step along trajectory')
    plt.ylabel('Angular speed of needle (rad/s)')

    plt.title('Angular speed vs time step')

def plot_satellite(args, us, ilqr_us=None, admm_us=None, subplot=False):
    if subplot:
        plt.subplot(1, 2, 1)

    # plt.plot(range(len(us)), [u[0] for u in us], 'b-', label='AULA $u_1$')
    # plt.plot(range(len(us)), [u[1] for u in us], 'b--', label='AULA $u_2$')
    # plt.plot(range(len(us)), [u[2] for u in us], 'b:', label='AULA $u_3$')
    plt.plot(range(len(us)), [np.linalg.norm(u, ord=1) for u in us], 'b-', label='\\textsc{Tron}', linewidth=3)

    if ilqr_us:
        # plt.plot(range(len(us)), [u[0] for u in ilqr_us], 'r-', label='ILQR $u_1$')
        # plt.plot(range(len(us)), [u[1] for u in ilqr_us], 'r--', label='ILQR $u_2$')
        # plt.plot(range(len(us)), [u[2] for u in ilqr_us], 'r:', label='ILQR $u_3$')

        plt.plot(range(len(us)), [np.linalg.norm(u, ord=1) for u in ilqr_us], 'r-', label='\\textsc{Ilqr}', linewidth=3)

    if admm_us:        
        # plt.plot(range(len(us)), [u[0] for u in admm_us], 'g-', label='ADMM $u_1$')
        # plt.plot(range(len(us)), [u[1] for u in admm_us], 'g--', label='ADMM $u_2$')
        # plt.plot(range(len(us)), [u[2] for u in admm_us], 'g:', label='ADMM $u_3$')

        plt.plot(range(len(us)), [np.linalg.norm(u, ord=1) for u in admm_us], 'g-', label='\\textsc{Admm}', linewidth=3)

    plt.xlabel('Time step along trajectory')
    plt.ylabel('L1-norm of control input $\|u\|_1$')
    plt.legend()
    # plt.ylim([-0.01, 0.01])

    plt.title('L1-norm of control input vs time step')
