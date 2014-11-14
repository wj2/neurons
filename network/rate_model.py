
import numpy as np
import scipy.integrate as sp_integrate
import matplotlib.pyplot as plt
import neurons.solver as ns

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

    def __init__(self, m, n, f_big):
        self.f_big = f_big
        self.m = m
        self.n = n

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

    def _unpackage(self, init):
        """ 
        init is an mx1 column, an mxn feedforward weight matrix, an mxm 
        recurrent weight matrix, an mx1 column e, and an mx1 column  s
        """
        vs = np.array(init[:self.m]).reshape(self.m, 1)
        init = init[self.m:]
        w_ = np.array(init[:self.m*self.n]).reshape(self.m, self.n)
        init = init[self.m*self.n:]
        m_ = np.array(init[:self.m*self.m]).reshape(self.m, self.m)
        init = init[self.m*self.m:]
        e = np.array(init[:self.m]).reshape(self.m, 1)
        init = init[self.m:]
        s = np.array(init[:self.m]).reshape(self.m, 1)
        init = init[self.m:]
        assert len(init) == 0
        return vs, w_, m_, e, s

    def _package(self, vs, w_, m_, e, s):
        return np.concatenate((vs.flatten(), w_.flatten(), m_.flatten(), 
                               e.flatten(), s.flatten()))

    def plot_simulation(self, simout, tcourse, figsize=None, iput=None):
        fig = plt.figure(figsize=figsize)
        if iput is None:
            iput = [self._inputfunc(t) for t in tcourse]
        output = simout[:, :self.m]
        simout = simout[:, self.m:]
        rateplot = fig.add_subplot(3, 1, 1)
        rateplot.set_title('rates')
        rateplot.plot(tcourse, iput[:], 'r', label='feedforward')
        for i in xrange(int(self.m)):
            if i == 0:
                rateplot.plot(tcourse, output[:, i], 'b', label='recurrent')
            else:
                rateplot.plot(tcourse, output[:, i], 'b')
        rateplot.legend()

        ffweights = simout[:, :self.m*self.n]
        simout = simout[:, self.m*self.n:]
        weightplot = fig.add_subplot(3, 1, 2)
        weightplot.set_title('weights')
        for i in xrange(int(self.m*self.n)):
            if i == 0:
                weightplot.plot(tcourse, ffweights[:, i], 'r', 
                                label='feedforward')
            else:
                weightplot.plot(tcourse, ffweights[:, i], 'r')
        rcweights = simout[:, :self.m*self.m]
        simout = simout[:, self.m*self.m:]
        for i in xrange(int(self.m*self.m)):
            if i == 0:
                weightplot.plot(tcourse, rcweights[:, i], 'b', label='recurrent')
            else:
                weightplot.plot(tcourse, rcweights[:, i], 'b')
        weightplot.legend()
            
        threshplot = fig.add_subplot(3, 1, 3)
        threshplot.set_title('thresholds')
        etime = simout[:, :self.m]
        simout = simout[:, self.m:]
        for i in xrange(int(self.m)):
            if i == 0:
                threshplot.plot(tcourse, etime[:,i], 'g', label='avg')
            else:
                threshplot.plot(tcourse, etime[:,i], 'g')
        stime = simout[:, :self.m]
        simout = simout[:, self.m:]
        for i in xrange(int(self.m)):
            if i == 0:
                threshplot.plot(tcourse, stime[:, i], 'm', label='var')
            else:
                threshplot.plot(tcourse, stime[:, i], 'm')
        threshplot.legend()
        assert simout.size == 0
        plt.show()

    def simulate(self, vs_0, w_0, m_0, e_0, s_0, tcourse, inputfunc, **kwargs):
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
        init = self._package(vs_0, w_0, m_0, e_0, s_0)
        self._inputfunc = inputfunc
        self._t = 0
        out = sp_integrate.odeint(self.compute_step, init, tcourse)
        return out

    def simulate_euler(self, vs_0, w_0, m_0, e_0, s_0, tend, tstep, inputfunc):
        """
        vs_0 is the initial value of the m-length recurrent rates vector
        w_0 is the initial value of the mxn weight matrix of connections from
          feedforward neurons to recurrent neurons
        m_0 is the initial value of the mxm weight matrix of connections 
          between recurrent neurons
        tcourse is a sequence of time values to calculate network state at
        inputfunc is a function accepting a time value t and producing n-length
          vector us, the feedforward neuron activity at time t
        """
        init = self._package(vs_0, w_0, m_0, e_0, s_0)
        print init
        self._inputfunc = inputfunc
        self._t = 0
        esolver = ns.EulerSolver(self.odec_compute_step, self._t, init)
        ts, ys = esolver.integrate_over(tend, tstep)
        return np.array(ts), np.array(ys)
        
    def simulate_altode(self, vs_0, w_0, m_0, e_0, s_0, t0, tend, dt, 
                        inputfunc, **kwargs):
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
        init = self._package(vs_0, w_0, m_0, e_0, s_0)
        self._inputfunc = inputfunc
        self._t = 0
        integ = sp_integrate.ode(self.odec_compute_step)
        integ.set_initial_value(init, t0)
        result = []
        while integ.successful() and integ.t < tend:
            integ.integrate(integ.t + dt)
            result.append(integ.y)
        return integ, np.array(result)

class RateNetworkCov(RateNetwork):
    
    def __init__(self, m, n, f_big, tau_r, tau_e, tau_w, wsat, alpha):
        self.tau_r = tau_r
        self.tau_e = tau_e
        self.tau_w = tau_w
        self.weightsat = wsat
        self.tau_s = self.tau_e
        self.alpha = alpha
        super(RateNetworkCov, self).__init__(m, n, f_big)
    
    def _dwdt(self, us, vs, e_u, e_v, currweights):
        delt = np.outer(vs - self.alpha*e_v, us - e_u) / self.tau_w
        delt[currweights > self.weightsat] = 0
        delt[currweights < 0] = 0
        # delt[np.diag_indices_from(delt)] = 0
        return delt

    def odec_compute_step(self, t, init):
        return self.compute_step(init, t)

    def compute_step(self, init, t, params=None):
        vs, w_, m_, e, s = self._unpackage(init)
        us = self._inputfunc(t)
        e_p1 = self._dedt(e, vs)
        s_p1 = np.zeros((self.m, 1))
        vs_p1 = self._dvdt(vs, us, w_, m_)
        m_p1 = self._dwdt(vs, vs, e, e, m_)
        # m_p1 = np.zeros((self.m, self.m)) 
        w_p1 = np.zeros((self.m, self.n))
        outpack = self._package(vs_p1, w_p1, m_p1, e_p1, s_p1)
        assert outpack.size == init.size
        return outpack

class RateNetworkBCM(RateNetwork):
    
    def __init__(self, m, n, f_big, tau_r, tau_e, tau_m, wsat, lam):
        self.tau_r = tau_r
        self.tau_e = tau_e
        self.tau_s = tau_e
        self.tau_m = tau_m
        self.lam = lam
        self.wsat = wsat
        super(RateNetworkBCM, self).__init__(m, n, f_big)

    def _dmdt(self, vs, theta):
        delt = np.multiply(np.outer(vs, vs).T, (vs - theta).T).T / self.tau_m
        delt[delt < 0] = 0
        delt[delt > self.wsat] = self.wsat
        return delt

    def _calc_theta(self, e, s):
        return e + self.lam*np.sqrt(s - e**2)

    def odec_compute_step(self, t, init):
        return self.compute_step(init, t)

    def compute_step(self, init, t, params=None):
        vs, w_, m_, e, s = self._unpackage(init)
        us = self._inputfunc(t)
        theta = self._calc_theta(e, s)
        
        e_p1 = self._dedt(e, vs)
        s_p1 = self._dsdt(s, vs)
        vs_p1 = self._dvdt(vs, us, w_, m_)
        m_p1 = self._dmdt(vs, theta)
        w_p1 = np.zeros((self.m, self.n))
        outpack = self._package(vs_p1, w_p1, m_p1, e_p1, s_p1)
        return outpack
