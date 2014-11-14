
import numpy as np
import scipy.integrate as sp_integrate
import matplotlib.pyplot as plt
import math
import neurons.solver as ns

class AdaptiveEIFNeuron(object):
    
    def __init__(self, tau_w, tau_m, delta_t, v_t, v_spk, v_r, a, b):
        self.tau_w = tau_w
        self.tau_m = tau_m
        self.delta_t = delta_t
        self.v_t = v_t
        self.v_spk = v_spk
        self.v_r = v_r
        self.a = a
        self.b = b

    def _dwdt(self, v, w):
        delt = (-w + self.a*v) / self.tau_w
        return delt

    def _dvdt(self, v, w):
        delt = (-v + self.delta_t*math.exp((v - self.v_t) / self.delta_t)
                - w + self.curr) / self.tau_m
        return delt

    def _compute_step(self, t, init):
        v, w = init
        dv = self._dvdt(v, w)
        dw = self._dwdt(v, w)
        return np.array([dv, dw])

    def simulate_euler(self, v, w, curr, tfinal, tbeg=0, dt=.1):
        self.curr = curr
        init = np.array([v, w])
        integ = ns.EulerSolver(self._compute_step, tbeg, init)
        
        while integ.success and integ.t_curr < tfinal:
            integ.integrate(integ.t_curr + dt)
            if integ.y_curr[0] > self.v_spk:
                integ.y_curr[0] = self.v_r
                integ.y_curr[1] = integ.y_curr[1] + self.b
        return integ, np.array(integ.y_record)

    def simulate(self, v, w, curr, tfinal, tbeg=0, dt=.1):
        self.curr = curr

        init = [v, w]
        odeinteg = sp_integrate.ode(self._compute_step)
        odeinteg.set_integrator('dopri5')
        odeinteg.set_initial_value(init, tbeg)
        result = []
        while odeinteg.successful() and odeinteg.t < tfinal:
            if odeinteg.y[0] >= self.v_spk:
                print 'spike!'
                odeinteg._y[0] = self.v_r
                odeinteg._y[1] = odeinteg._y[1] + self.b
                odeinteg.t = odeinteg.t + dt
            else:
                odeinteg.integrate(odeinteg.t+dt)
                print odeinteg._y
                result.append(odeinteg.y)
        # out = sp_integrate.odeint(self._compute_step, init, tcourse)
        return odeinteg, np.array(result)

def sim_and_plot(tau_m, v_t, delta_t, a, b, v_r, curr, tau_w, plot_v, plot_w):
    v_spk = 50
    aeif = AdaptiveEIFNeuron(tau_w, tau_m, delta_t, v_t, v_spk, v_r, a, b)
    v, w = 0, 0
    integ, res = aeif.simulate_euler(v, w, curr, 1000, dt=.01)
    tcourse = np.arange(0, res.shape[0] / 100., .01)
    print res.shape, tcourse.shape
    plot_v.plot(tcourse, res[:, 0], 'b', label='V')
    plot_v.set_ylabel('mV')
    plot_w.plot(tcourse, res[:, 1], 'r', label='W')
    plot_w.set_ylabel('W')
    plot_v.legend()
    plot_w.legend()

def ps2_plot():
    fig = plt.figure()
    tau_m, v_t, delta_t = 20, 15, 2

    plot_av = fig.add_subplot(5, 2, 1)
    plot_av.set_title('a')
    plot_aw = fig.add_subplot(5, 2, 2)
    a, b, v_r, curr, tau_w = 0, 0, 10, 18, 50
    sim_and_plot(tau_m, v_t, delta_t, a, b, v_r, curr, tau_w, plot_av, plot_aw)

    plot_bv = fig.add_subplot(5, 2, 3)
    plot_bv.set_title('b')
    plot_bw = fig.add_subplot(5, 2, 4)
    a, b, v_r, curr, tau_w = 5, 0, 10, 10, 50
    sim_and_plot(tau_m, v_t, delta_t, a, b, v_r, curr, tau_w, plot_bv, plot_bw)

    plot_cv = fig.add_subplot(5, 2, 5)
    plot_cv.set_title('c')
    plot_cw = fig.add_subplot(5, 2, 6)
    a, b, v_r, curr, tau_w = 5, 0, 10, 40, 50
    sim_and_plot(tau_m, v_t, delta_t, a, b, v_r, curr, tau_w, plot_cv, plot_cw)

    plot_dv = fig.add_subplot(5, 2, 7)
    plot_dv.set_title('d')
    plot_dw = fig.add_subplot(5, 2, 8)
    a, b, v_r, curr, tau_w = 0, 5, 10, 30, 200
    sim_and_plot(tau_m, v_t, delta_t, a, b, v_r, curr, tau_w, plot_dv, plot_dw)

    plot_ev = fig.add_subplot(5, 2, 9)
    plot_ev.set_title('e')
    plot_ew = fig.add_subplot(5, 2, 10)
    a, b, v_r, curr, tau_w = 0, 5, 17.5, 30, 100
    sim_and_plot(tau_m, v_t, delta_t, a, b, v_r, curr, tau_w, plot_ev, plot_ew)
    plot_ew.set_xlabel('ms')
    plot_ev.set_xlabel('ms')
    
    plt.show()
