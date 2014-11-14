
import scipy.stats as sst
import numpy as np
import warnings

class InputPattern(object):
    """
    InputPattern is an abstract class for use in generating patterns of random
    samples from probability distributions for the stimulations of networks.
    """

    def __init__(self, numtrials, gap, duration):
        self._gap = gap
        self._duration = duration
        self._trial_duration = gap + duration
        self.totaltime = gap + numtrials*(self._trial_duration)

    def _get_baseline(self):
        raise NotImplementedError('InputPattern is an abstract class, '
                                  '_get_baseline must be implemented in its '
                                  'children')
        return
        
    def _get_trialnum(self, trial):
        raise NotImplementedError('InputPattern is an abstract class, '
                                  '_get_trialnum must be implemented in its '
                                  'children')
        return

    def __call__(self, t):
        if t > self.totaltime:
            warnings.warn('asked for a time outside of defined block, giving '
                          'baseline, this is probably a feature of your ode '
                          'solver', RuntimeWarning)
        t_mod = t - self._gap
        if t_mod < 0:
            val = self._get_baseline()
        else: 
            tnum = np.floor(t_mod / self._trial_duration)
            if t_mod - tnum*self._trial_duration - self._duration < 0:
                val = self._get_trialnum(tnum)
            else:
                val = self._get_baseline()
        return val[0]
    
class InputPatternGaussian(InputPattern):

    _func = sst.norm.rvs

    def __init__(self, means, stds, duration, gap, baseline_mean, baseline_std,
                 negs=False):
        self._means = means
        self._stds = stds
        self._baseline_mean = baseline_mean
        self._baseline_std = baseline_std
        self._negs = negs
        super(InputPatternGaussian, self).__init__(len(means), gap, duration)

    def _get_baseline(self):
        out = self._func(self._baseline_mean, self._baseline_std, 1)
        if not self._negs:
            out[out < 0] = 0
        return out

    def _get_trialnum(self, trial):
        assert trial == int(trial)
        trial = int(trial)
        out = self._func(self._means[trial], self._stds[trial], 1)
        if not self._negs:
            out[out < 0] = 0
        return out
        

def pattern_gaussians(means, stds, duration, gap, base_mean, base_std):
    try: 
        stds[0]
    except:
        stds = [stds for m in means]
    assert len(stds) == len(means)
    return InputPatternGaussian(means, stds, duration, gap, base_mean, 
                                base_std)
    
