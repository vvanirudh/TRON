import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def regularize(Q):
    w, v = np.linalg.eig(Q)

    w[w < 0] = 0

    Q_new = np.real(v.dot(np.diag(w).dot(v.T)))
    return Q_new

def skewSymmetric(x):
    assert x.shape[0] == 3
    result = np.zeros((3, 3))
    result[0, 1] = -x[2]
    result[0, 2] = x[1]
    result[1, 0] = x[2]
    result[1, 2] = -x[0]
    result[2, 0] = -x[1]
    result[2, 1] = x[0]

    return result

def cpMatrix(x):
    assert x.shape[0] == 3
    A = np.zeros((3, 3))

    A[0, 1] = -x[2]
    A[0, 2] = x[1]
    A[1, 0] = x[2]
    A[1, 2] = -x[0]
    A[2, 0] = -x[1]
    A[2, 1] = x[0]

    return A

def rotFromErr(q):
    assert q.shape[0] == 3
    rr = q[0]**2 + q[1]**2 + q[2]**2
    if rr == 0:
        return np.eye(3)
    else:
        r = np.sqrt(rr)
        return cpMatrix(q * (np.sin(r) / r)) + np.eye(3) * np.cos(r) + np.outer(q, q) * ((1 - np.cos(r)) / rr)

def errFromRot(R):
    assert R.shape == (3, 3)
    q = np.zeros(3)
    q[0] = R[2, 1] - R[1, 2]
    q[1] = R[0, 2] - R[2, 0]
    q[2] = R[1, 0] - R[0, 1]

    r = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2)
    t = R[0, 0] + R[1, 1] + R[2, 2] - 1

    if r==0:
        return np.zeros(3)
    else:
        return q * np.arctan2(r, t) / r


def ThirdorderDerivative(x1, x2, x3, x4, ys1, ys2, ys3, ys4):
    dim = ys1.shape[0]
    ds = np.zeros(dim)
    mid = (x2 + x3) / 2.0

    for i in range(0, dim):
        y1 = ys1[i]
        y2 = ys2[i]
        y3 = ys3[i]
        y4 = ys4[i]
        a = y1.copy()
        b1 = (y2 - y1) / (x2 - x1)
        b2 = (y3 - y2) / (x3 - x2)
        b3 = (y4 - y3) / (x4 - x3)
        c1 = (b2 - b1) / (x3 - x1)
        c2 = (b3 - b2) / (x4 - x2)
        d = (c2 - c1) / (x4 - x1)

        ds[i] += b1
        ds[i] += (mid - x1) * c1 + (mid - x2) * c1  # TODO: Is the second term c1?
        ds[i] += (mid - x2) * (mid - x3) * d + (mid - x1) * (mid - x3) * d + (mid - x1) * (mid - x2) * d

    return ds
