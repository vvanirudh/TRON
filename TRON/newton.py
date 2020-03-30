import torch
import numpy as np
from torch import autograd

from TRON.utils import bcolors

def newton_step(args, output, var, func):
    def hessian_vector_product(v):        
        grad, = torch.autograd.grad(output, var, create_graph=True)
        z = grad @ v
        _hvp = autograd.grad(z, var, retain_graph=True)[0].data
        return _hvp
    # Calculate newton decrement
    # v = torch.mm(-torch.inverse(hessian), grad)
    grad, = torch.autograd.grad(output, var, retain_graph=True)
    v = get_newton_decrement(hessian_vector_product, -grad, args.cgsteps)
    # Backtracking line search
    t = backtrack(args, var, func, v, grad)
    # Apply the step
    var_new = var + t*v
    return var_new


def backtrack(args, var, func, newton_decrement, grad):
    alpha = args.alpha
    beta = args.beta

    t = 1
    while func(var + t*newton_decrement) > func(var) + alpha * t * torch.dot(grad, newton_decrement):
        t = beta*t

    print(bcolors.BOLD+'Value of t is '+str(t)+bcolors.ENDC)
    return t


def get_newton_decrement(hvp, b, nsteps, residual_tol=1e-10):
    '''
    Implements the conjugate gradient method
    '''
    x = torch.zeros(b.size())  # Solution has the same size as the gradient
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)   # Gradient norm squared

    for i in range(nsteps):
        #print(bcolors.OKGREEN+'CG Iteration '+str(i)+bcolors.ENDC)
        _hvp = hvp(p)
        alpha = rdotr / torch.dot(p, _hvp)
        x += alpha * p
        r -= alpha * _hvp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break

    return x
