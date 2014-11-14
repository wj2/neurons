
import numpy as np
import scipy.optimize as sop



def hyperbolic_tangent(x, a, b, c, d):
    return a*np.tanh(x*d + c) + b

def sigmoid(x, a, b, c, d):
    """ a, b should typically remain fixed while x varies """
    return (a / (1 + np.exp(-x*d - c))) + b

def fit_function(func, xdat, ydat):
    ps, vars = sop.curve_fit(func, xdat, ydat)
    return lambda x: func(x, *ps), ps
