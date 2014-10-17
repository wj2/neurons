
import numpy as np
import scipy.integrate as sp_integrate
import math

def hyperbolic_tangent(x):
    pass

def sigmoid(x, a, b):
    """ a, b should typically remain fixed while x varies """
    return (a / (1 + math.exp(-x))) + b

class RateNetwork(object):
    """ 
    A RateNetwork is a two layer network of m recurrent neurons and n 
      feedforward neurons
    vs is a vector of the rates of the recurrent neurons, v_1, ..., v_m
    us is a vector of the rates of the feedforward neurons, u_1, ..., u_n
    w_ is an mxn matrix of the synaptic weights w_ij from feedforward neuron 
      u_j to recurrent neuron v_i
    m_ is an mxm matrix of the synaptic weights m_ij from recurrent neuron v_j
      to recurrent neuron v_i
    f_big is a function transforming input firing rates to output firing rates
    
    The core differential equation used here is 
        tau_r*dv/dt = -vs + f_big(w_.us + m_.vs)
      which allows linear or nonlinear transfer functions (f_big) and can also
      be used as a strictly feedforward network (ie, set m_ = 0) or a recurrent
      network with any number of inputs and plastic or non-plastic feedforward
      weights

    This class is meant to be subclassed for specific models implementing 
    synaptic plasticity rules or with other variations.
    """

    def compute_step(init, t, params):
        """ 
        Not implemented here, should be implemented by subclasses.

        Must take in an init vector (vs, w_, m_) as first argument and a time t
        as second argument unless simulate function is also overridden. 
        A dictionary of arbitrary parameters are automatically passed to this 
        function from the call to simulate as third argument.
        """
        raise NotImplementedError('the method compute_step is abstract')
        return

    def _dvdt(self, vs, us, w_, m_):
        return (-vs + self.f_big(np.dot(w_, us) + np.dot(m_, vs))) / self.tau_r

    def _dedt(self, e, vs):
        return (-e + vs) / self.tau_e
    
    def _dsdt(self, s, vs):
        return (-s + vs**2) / self.tau_s

    def simulate(vs_0, w_0, m_0, e_0, s_0, tcourse, inputfunc, **kwargs):
        """
        vs_0 is the initial value of the m-length recurrent rates vector
        w_0 is the initial value of the mxn weight matrix of connections from
          feedforward neurons to recurrent neurons
        m_0 is the initial value of the mxm weight matrix of connections 
          between recurrent neurons
        tcourse is a sequence of time values to calculate network state at
        inputfunc is a function accepting a time value t and producing n-length
          vector us, the feedforward neuron activity at time t
        
        Key word arguments are passed as a dictionary to the integration of the
        differential equation. 
        """
        init = (vs_0, w_0, m_0, e_0, s_0)
        self._inputfunc = inputfunc
        out = sp_integrate.odeint(self.compute_step, init, tcourse, kwargs)
        return out

class RateNetworkCov(RateNetwork):
    
    def __init__(self, tau_r, tau_e):
        self.tau_r = tau_r
        self.tau_e = tau_e
        self.tau_s = self.tau_e
    
    def _delta_weights(self, vs, e):
        return np.outer(vs - e, vs - e)

    def compute_step(self, init, t, params):
        vs, w_, m_, e, s = init
        us = self._inputfunc(t)

        e_p1 = self._dedt(e, vs)
        s_p1 = s
        vs_p1 = self._dvdt(vs, us, w_, m_)
        m_p1 = self._delta_weights(vs, e)
        w_p1 = w_

        return vs_p1, w_p1, m_p1, e_p1, s_p1
        

class RateNetworkBCM(RateNetwork):
    
    def __init__(self, tau_r, tau_e, tau_m, lam):
        self.tau_r = tau_r
        self.tau_e = tau_e 
        self.tau_m = tau_m
        self.lam = lam

    def _dmdt(self, vs, theta):
        return np.multiply(np.outer(vs, vs).T, (vs - theta).T).T / self.tau_m

    def _calc_theta(e, s):
        return e + self.lam*np.sqrt(v - e**2)

    def compute_step(init, t, params):
        vs, w_, m_, e, s = init
        us = self._inputfunc(t)
        theta = self._calc_theta(e, s)
        
        e_p1 = self._dedt(e, vs)
        s_p1 = self._dsdt(s, vs)
        vs_p1 = self._dvdt(vs, us, w_, m_)
        m_p1 = self._dmdt(vs, theta)
        w_p1 = w_

        return vs_p1, w_p1, m_p1, e_p1, s_p1
