
import math
import scipy.integrate as sp_integrate
import numpy as np
import matplotlib.pyplot as plt

# reasonable plot: hh.compare(hh_neuron, kk_neuron, (0, 10, 20, 50), 
#                             np.arange(0,100,.1)

# a Neuron has a function simulate that returns a tuple of array-likes in which
# the first array-like is membrane potential (mV) and the remaining columns are
# other parameters of interest

def plot(v, m, n, h):
    fig = plt.figure()
    volt_plot = fig.add_subplot(2, 1, 1)
    volt_plot.plot(v)
    gate_plot = fig.add_subplot(2, 1, 2)
    gate_plot.plot(m, label='m')
    gate_plot.plot(n, label='n')
    gate_plot.plot(h, label='h')
    gate_plot.legend()
    plt.show()

def homework_plot():
    hh_neuron = HodgkinHuxleyNeuron()
    kk_neuron = KrinskyKokozNeuron()
    compare(hh_neuron, kk_neuron, (0, 10, 20, 50), np.arange(0, 100, .1))

def comparison_plot(type_one, type_two, ts, special='mV'):
    """
      each type is a dictionary with the same number of entries with the same
      keys, each key's value is a dictionary with the entry special and any number
      of other entries, each value is an array-like of numbers
      variables, and n different datapoints
      -- between each type, k must be equal; the other two may vary
    """
    currents = sorted(type_one.keys())
    numplots = len(currents)
    comp_fig = plt.figure()
    comp_fig.suptitle('Hodgkin-Huxley (left) and Krinsky-Kokoz (right) neurons'
                      ' with a variety of input currents')
    for i, current in enumerate(currents):
        topleft_plot = comp_fig.add_subplot(numplots*2, 2, i*4+1)
        topleft_plot.plot(ts, type_one[current][special])
        topleft_plot.set_ylabel(special)
        topleft_plot.set_title('I='+str(current))
        plt.setp(topleft_plot.get_xticklabels(), visible=False)

        topright_plot = comp_fig.add_subplot(numplots*2, 2, i*4+2, 
                                             sharey=topleft_plot)
        topright_plot.plot(ts, type_two[current][special])
        plt.setp(topright_plot.get_yticklabels(), visible=False)
        plt.setp(topright_plot.get_xticklabels(), visible=False)

        botleft_plot = comp_fig.add_subplot(numplots*2, 2, i*4+3)
        for key in type_one[current].keys():
            if key != special:
                botleft_plot.plot(ts, type_one[current][key], label=key)
        botleft_plot.legend()

        botright_plot = comp_fig.add_subplot(numplots*2, 2, i*4+4, 
                                             sharey=botleft_plot)
        for key in type_two[current].keys():
            if key != special:
                botright_plot.plot(ts, type_two[current][key], label=key)
        botright_plot.legend()
        plt.setp(botright_plot.get_yticklabels(), visible=False)
        if i == numplots - 1:
            botleft_plot.set_xlabel('ms')
            botright_plot.set_xlabel('ms')
        else:
            plt.setp(botright_plot.get_xticklabels(), visible=False)
            plt.setp(botleft_plot.get_xticklabels(), visible=False)
    plt.show()
        
def compare(neuron1, neuron2, currents, tscale):
    n1 = {}
    n2 = {}
    for current in currents:
        n1[current] = {}
        n2[current] = {}
        out_n1 = neuron1.simulate(0, 0, 0, 0, -current, tscale)
        n1[current]['mV'] = out_n1[0]
        n1[current]['m'] = out_n1[1]
        n1[current]['n'] = out_n1[2]
        n1[current]['h'] = out_n1[3]
        out_n2 = neuron2.simulate(0, 0, -current, tscale)
        n2[current]['mV'] = out_n2[0]
        n2[current]['m'] = out_n2[1]
        n2[current]['n'] = out_n2[2]
        n2[current]['h'] = out_n2[3]
    comparison_plot(n1, n2, tscale)

class HodgkinHuxleyNeuron(object):
    """ 
      simulates a Hodgkin-Huxley neuron with constants and equations directly
      from Hodgkin AL and Huxley AF (1952) A Quantitative Description of 
      membrane current and it's application to conduction and excitation in 
      nerve. J. Physiol. 117
    """

    g_L = .3 # mmho/cm2
    g_K = 36.  # mmho/cm2
    g_Na = 120. # mmho/cm2
    e_L = -10.613 # mV
    e_K = 12. # mV
    e_Na = -115. # mV
    C_m = 1. # microF/cm2

    def _dvdt(self, v_m, m, n, h):
        return (-self.g_K*(n**4)*(v_m - self.e_K) 
                - self.g_Na*(m**3)*h*(v_m - self.e_Na) 
                - self.g_L*(v_m - self.e_L) + self.current) / self.C_m

    
    def _dzdt(self, alpha, beta, z): 
        # stands in for dndt, dmdt, dhdt as they would all have this same form
        return alpha*(1. - z) - beta*z

    def _alpha_n(self, v_m):
        return .01*(v_m + 10.) / (math.exp((v_m + 10.) / 10.) - 1.)

    def _beta_n(self, v_m):
        return .125*math.exp(v_m / 80.)

    def _alpha_m(self, v_m):
        return .1*(v_m + 25.) / (math.exp((v_m + 25.) / 10.) - 1.)

    def _beta_m(self, v_m):
        return 4.*math.exp(v_m / 18.)
        
    def _alpha_h(self, v_m):
        return .07*math.exp(v_m / 20.)

    def _beta_h(self, v_m):
        return 1. / (math.exp((v_m + 30.) / 10.) + 1)

    def _compute_step(self, init, t):
        v_m, m, n, h = init
        v_mp1 = self._dvdt(v_m, m, n, h)
        m_p1 = self._dzdt(self._alpha_m(v_m), self._beta_m(v_m), m)
        n_p1 = self._dzdt(self._alpha_n(v_m), self._beta_n(v_m), n)
        h_p1 = self._dzdt(self._alpha_h(v_m), self._beta_h(v_m), h)
        return v_mp1, m_p1, n_p1, h_p1

    def simulate(self, v_m, m, n, h, current, t_steps):
        self.current = current
        out = sp_integrate.odeint(self._compute_step, (v_m, m, n, h), t_steps)
        return -out[:, 0].T, out[:, 1].T, out[:, 2].T, out[:, 3].T


class KrinskyKokozNeuron(object):

    g_L = .3 # mmho/cm2
    g_K = 36.  # mmho/cm2
    g_Na = 120. # mmho/cm2
    e_L = -10.613 # mV
    e_K = 12. # mV
    e_Na = -115. # mV
    C_m = 1. # microF/cm2

    # reduction from m to m_inf
    # reduction from h to (1 - n)
    def _dvdt(self, v_m, m_inf, n):
        return (-self.g_K*(n**4)*(v_m - self.e_K) 
                - self.g_Na*(m_inf**3)*(1 - n)*(v_m - self.e_Na) 
                - self.g_L*(v_m - self.e_L) + self.current) / self.C_m

    def _m_inf(self, alpha_m, beta_m):
        return alpha_m / (alpha_m + beta_m)

    def _dzdt(self, alpha, beta, z): 
        # stands in for dndt, dmdt, dhdt as they would all have this same form
        return alpha*(1. - z) - beta*z

    def _alpha_n(self, v_m):
        return .01*(v_m + 10.) / (math.exp((v_m + 10.) / 10.) - 1.)

    def _beta_n(self, v_m):
        return .125*math.exp(v_m / 80.)

    def _alpha_m(self, v_m):
        return .1*(v_m + 25.) / (math.exp((v_m + 25.) / 10.) - 1.)

    def _beta_m(self, v_m):
        return 4.*math.exp(v_m / 18.)
        
    def _compute_step(self, init, t):
        v_m, n = init
        m_inf = self._m_inf(self._alpha_m(v_m), self._beta_m(v_m))
        v_mp1 = self._dvdt(v_m, m_inf, n)
        n_p1 = self._dzdt(self._alpha_n(v_m), self._beta_n(v_m), n)
        return v_mp1, n_p1

    def simulate(self, v_m, n, current, t_steps):
        self.current = current
        out = sp_integrate.odeint(self._compute_step, (v_m, n), t_steps)
        # let's recover m and h, to see how they look
        nums = out.shape[0]
        hs = np.ones((nums)) - out[:, 1].T
        ms = np.array([self._m_inf(self._alpha_m(v), self._beta_m(v)) 
                       for v in out[:, 0].T])
        return -out[:, 0].T, ms, out[:, 1].T, hs
    
