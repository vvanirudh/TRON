import numpy as np
from TRON.utils import bcolors
import time

###################### DIFFDRIVE ##################################

DEFAULTSTEPSIZE = 0.0009765625


def finite_diff_jacobian_x(x, u, f, step=DEFAULTSTEPSIZE):
    xDim = x.shape[0]

    A = np.zeros((xDim, xDim))
    ar = np.copy(x)
    al = np.copy(x)

    for i in range(xDim):
        ar[i] += step
        al[i] -= step
        A[:, i] = (f(ar, u) - f(al, u)) / (2*step)
        ar[i] = al[i] = x[i]

    return A


def finite_diff_jacobian_u(x, u, f, step=DEFAULTSTEPSIZE):
    xDim = x.shape[0]
    uDim = u.shape[0]

    B = np.zeros((xDim, uDim))
    br = np.copy(u)
    bl = np.copy(u)

    for i in range(uDim):
        br[i] += step
        bl[i] -= step
        B[:, i] = (f(x, br) - f(x, bl)) / (2*step)
        br[i] = bl[i] = u[i]

    return B


def ilqr(env, horizon_length, init_x, u_nominal, g, quadratize_final_cost, final_cost, quadratize_cost, cost, l, L, verbose, maxIter, terminate):
    xDim = init_x.shape[0]
    uDim = u_nominal.shape[0]

    costs = []
    times = []

    if maxIter is None:
        maxIter = 1000

    if L is None:
        L = [np.zeros((uDim, xDim)) for _ in range(horizon_length)]
    if l is None:
        l = [np.zeros(uDim) for _ in range(horizon_length)]

    # Initialize xhat, uhat with current control
    xHat, uHat = env.rollout(l, L, verbose=False)

    xHatNew = [np.zeros(xDim) for _ in range(horizon_length+1)]
    uHatNew = [np.zeros(uDim) for _ in range(horizon_length)]

    oldCost = np.inf

    for it in range(maxIter):
        alpha = 1.0
        sub_iter = 0
        while True:
            newCost = 0.0
            xHatNew[0] = np.copy(init_x)

            for t in range(horizon_length):
                uHatNew[t] = (1.0 - alpha)*uHat[t] + \
                    L[t].dot(xHatNew[t] - (1.0 - alpha)*xHat[t]) + alpha*l[t]
                xHatNew[t+1] = g(xHatNew[t], uHatNew[t])
                newCost += cost(xHatNew[t], uHatNew[t], t)

            newCost += final_cost(xHatNew[horizon_length])
            rel_progress = np.abs((oldCost - newCost) / newCost)
            if (newCost < oldCost) or (rel_progress < 1e-4):
                break

            alpha *= 0.5
            sub_iter += 1

        xHat = [x for x in xHatNew]
        uHat = [u for u in uHatNew]

        costs.append(env.get_cost(xHat, uHat))
        times.append(time.time())

        if verbose:
            print(bcolors.OKBLUE+"Iter : "+str(it) + " alpha : "+str(alpha) +
                  " rel. progress : "+str(rel_progress)+" cost : "+str(newCost)+bcolors.ENDC)

        if terminate and (rel_progress < 1e-4) and (maxIter > 5):
            return (it, l, L, costs, times)

        oldCost = newCost

        s, S = quadratize_final_cost(xHat[horizon_length])

        for t in range(horizon_length-1, -1, -1):
            A = finite_diff_jacobian_x(xHat[t], uHat[t], g)
            B = finite_diff_jacobian_u(xHat[t], uHat[t], g)

            c = xHat[t+1] - A.dot(xHat[t]) - B.dot(uHat[t])

            P, q, Q, r, R = quadratize_cost(xHat[t], uHat[t], t, it)

            C = B.T.dot(S.dot(A)) + P
            D = A.T.dot(S.dot(A)) + Q
            E = B.T.dot(S.dot(B)) + R
            d = A.T.dot(s + S.dot(c)) + q
            e = B.T.dot(s + S.dot(c)) + r

            L[t] = -1*np.linalg.lstsq(E, C, rcond=None)[0]
            l[t] = -1*np.linalg.lstsq(E, e, rcond=None)[0]

            S = D + C.T.dot(L[t])
            s = d + C.T.dot(l[t])

    return (it, l, L, costs, times)


#################################### NEEDLE ######################################


def compute_nominal_plan_from_policy(L, l, se_dynamics_fn, x_start):
    udim, xdim = L[0].shape
    horizon_length = len(L)
    g = np.zeros(xdim)
    M = np.zeros((xdim, xdim))

    plan = [[np.zeros(xdim), np.zeros(udim)] for _ in range(horizon_length+1)]
    plan[0][0] = x_start

    for i in range(horizon_length):
        plan[i][1] = np.dot(L[i], plan[i][0]) + l[i]
        g, M = se_dynamics_fn(plan[i][0], plan[i][1])
        plan[i+1][0] = g.copy()
        if np.any(np.isnan(plan[i][1])):
            import ipdb
            ipdb.set_trace()

    plan[horizon_length][1] = np.zeros(udim)

    return plan


def expected_cost(plan, L, linearize_discrete_dynamics_fn, quadratize_cost_fn, quadratize_final_cost_fn):
    udim, xdim = L[0].shape
    horizon_length = len(L)

    S = np.zeros((xdim, xdim))
    svec = np.zeros(xdim)
    sscale = 0.0

    qvecT, QT, qT = quadratize_final_cost_fn(plan[horizon_length][0])
    S = QT.copy()
    svec = qvecT.copy()
    sscale = qT

    for t in range(horizon_length-1, -1, -1):
        At, Bt, avect, Ft, Gt, evect = linearize_discrete_dynamics_fn(
            plan[t][0], plan[t][1])
        if np.any(np.isnan(plan[t][0])) or np.any(np.isnan(plan[t][1])):
            import ipdb
            ipdb.set_trace()
        Pt, qvect, Qt, rvect, Rt, qscalet = quadratize_cost_fn(
            plan[t][0], plan[t][1], t, 20)

        Ct = Qt + np.dot(At.T, np.dot(S, At))
        Dt = Rt + np.dot(Bt.T, np.dot(S, Bt))
        Et = Pt + np.dot(Bt.T, np.dot(S, At))
        cvect = qvect + np.dot(At.T, np.dot(S, avect)) + np.dot(At.T, svec)
        dvect = rvect + np.dot(Bt.T, np.dot(S, avect)) + np.dot(Bt.T, svec)
        escalet = qscalet + sscale + 0.5 * \
            np.dot(avect.T, np.dot(S, avect)) + \
            np.dot(avect.T, svec)

        for i in range(xdim):
            Ct += np.dot(Ft[i].T, np.dot(S, Ft[i]))
            Dt += np.dot(Gt[i].T, np.dot(S, Gt[i]))
            Et += np.dot(Gt[i].T, np.dot(S, Ft[i]))
            cvect += np.dot(Ft[i].T, np.dot(S, evect[i]))
            dvect += np.dot(Gt[i].T, np.dot(S, evect[i]))
            escalet += 0.5 * np.dot(evect[i].T, np.dot(S, evect[i]))

        kvect = np.dot(L[t], -plan[t][0]) + plan[t][1]
        S = Ct + np.dot(L[t].T, np.dot(Dt, L[t])) + \
            np.dot(L[t].T, Et) + np.dot(L[t].T, Et).T
        svec = np.dot(L[t].T, np.dot(Dt, kvect)) + \
            np.dot(Et.T, kvect) + cvect + np.dot(L[t].T, dvect)
        sscale = 0.5 * np.dot(kvect.T, np.dot(Dt, kvect)) + \
            np.dot(kvect.T, dvect) + escalet

    return 0.5 * np.dot(plan[0][0].T, np.dot(S, plan[0][0])) + np.dot(svec.T, plan[0][0]) + sscale


def b_value_iteration(plan, quadratize_final_cost_fn, quadratize_cost_fn, linearize_discrete_dynamics_fn):
    '''
    plan should be a list of lists
    the outer list of length horizon_length+1
    the inner list of length 2 containing x, u
    x should be of dim xdim
    u should be of dim udim
    '''
    xstart, ustart = plan[0][0], plan[0][1]
    udim, xdim = ustart.shape[0], xstart.shape[0]
    horizon_length = len(plan) - 1

    L = [np.zeros((udim, xdim)) for _ in range(horizon_length)]
    l = [np.zeros(udim) for _ in range(horizon_length)]

    S = np.zeros((xdim, xdim))
    svec = np.zeros(xdim)
    sscale = 0.0

    qvecT, QT, qT = quadratize_final_cost_fn(plan[horizon_length][0])

    S = QT.copy()
    svec = qvecT.copy()
    sscale = qT

    for t in range(horizon_length-1, -1, -1):
        At, Bt, avect, Ft, Gt, evect = linearize_discrete_dynamics_fn(
            plan[t][0], plan[t][1])
        Pt, qvect, Qt, rvect, Rt, qscalet = quadratize_cost_fn(
            plan[t][0], plan[t][1], t, 20)

        Ct = Qt + np.dot(At.T, np.dot(S, At))
        Dt = Rt + np.dot(Bt.T, np.dot(S, Bt))
        Et = Pt + np.dot(Bt.T, np.dot(S, At))
        cvect = qvect + np.dot(At.T, np.dot(S, avect)) + np.dot(At.T, svec)
        dvect = rvect + np.dot(Bt.T, np.dot(S, avect)) + np.dot(Bt.T, svec)
        escalet = qscalet + sscale + 0.5 * \
            np.dot(avect.T, np.dot(S, avect)) + \
            np.dot(avect.T, svec)

        for i in range(xdim):
            Ct += np.dot(Ft[i].T, np.dot(S, Ft[i]))
            Dt += np.dot(Gt[i].T, np.dot(S, Gt[i]))
            Et += np.dot(Gt[i].T, np.dot(S, Ft[i]))
            cvect += np.dot(Ft[i].T, np.dot(S, evect[i]))
            dvect += np.dot(Gt[i].T, np.dot(S, evect[i]))
            escalet += 0.5 * np.dot(evect[i].T, np.dot(S, evect[i]))

        L[t] = -1 * np.linalg.lstsq(Dt, Et, rcond=None)[0]
        l[t] = -1 * np.linalg.lstsq(Dt, dvect, rcond=None)[0]

        S = Ct + np.dot(Et.T, L[t])
        svec = cvect + np.dot(Et.T, l[t])
        sscale = escalet + 0.5 * np.dot(dvect.T, l[t])

    return L, l


def adjust_path(oldplan, L, l, se_dynamics_fn, x_start, epsilon):
    udim, xdim = L[0].shape
    horizon_length = len(L)
    plan = [[np.zeros(xdim), np.zeros(udim)] for _ in range(horizon_length+1)]
    plan[0][0] = x_start

    for i in range(horizon_length):
        u = (1 - epsilon) * oldplan[i][1] + L[i].dot(plan[i]
                                                     [0] - (1 - epsilon) * oldplan[i][0]) + l[i] * epsilon
        plan[i][1] = u

        xn, _ = se_dynamics_fn(plan[i][0], plan[i][1])
        plan[i+1][0] = xn.copy()

    return plan


def ilqr_needle(env, horizon_length, init_x, u_nominal, se_dynamics, quadratize_final_cost, quadratize_cost, cost, linearize_discrete_dynamics, l, L, verbose, maxIter, terminate):
    xdim = init_x.shape[0]
    udim = u_nominal.shape[0]

    plan = [[np.zeros(xdim), np.zeros(udim)] for _ in range(horizon_length+1)]
    tmpplan = [[np.zeros(xdim), np.zeros(udim)]
               for _ in range(horizon_length+1)]

    costs = []

    if L is None:
        L = [np.zeros((udim, xdim)) for _ in range(horizon_length)]
    if l is None:
        u_nominal[2] = 0.0
        l = [u_nominal.copy() / 2.0 for _ in range(horizon_length)]

    plan = compute_nominal_plan_from_policy(L, l, se_dynamics, init_x)

    costnew = 0.0
    costold = expected_cost(plan, L, linearize_discrete_dynamics,
                            quadratize_cost, quadratize_final_cost)

    if verbose:
        print('Initial cost', costold)

    it = 0
    while True:
        it += 1
        L, l = b_value_iteration(
            plan, quadratize_final_cost, quadratize_cost, linearize_discrete_dynamics)

        tmpplan = compute_nominal_plan_from_policy(L, l, se_dynamics, init_x)
        costnew = expected_cost(
            tmpplan, L, linearize_discrete_dynamics, quadratize_cost, quadratize_final_cost)

        alpha = 1.0
        while costnew > costold:
            alpha *= 0.5
            tmpplan = adjust_path(plan, L, l, se_dynamics, init_x, alpha)
            costnew = expected_cost(
                tmpplan, L, linearize_discrete_dynamics, quadratize_cost, quadratize_final_cost)

        rel_progress = np.abs(costnew - costold) / costold
        if verbose:
            print(bcolors.OKBLUE+"Iter : "+str(it) + " alpha : "+str(alpha) +
                  " rel. progress : "+str(rel_progress)+" cost : "+str(costnew)+bcolors.ENDC)

        costs.append(env.get_cost(tmpplan))

        if (it == maxIter) or (terminate and rel_progress < 1e-4):
            # totalcost = costnew
            # truecost = env.get_cost(tmpplan)
            nominalplan = tmpplan
            break
        else:
            plan = tmpplan
            costold = costnew

    return it, L, l, nominalplan, costs
