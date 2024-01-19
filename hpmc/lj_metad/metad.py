import os
import time
import math
import itertools
import numpy as np

import hoomd
import gsd.hoomd

from order_parameters import compute_num_liq as compute_op

class CustomAction(hoomd.custom.Action):
    """This custom Action class extends the Action class' functionality to
       include the simulation object. This is needed in order to access the
       alchemical variables stored and managed through param_array, which is
       owned by the integrator in the current API.
    """
    def __init__(self):
        super().__init__()

    def attach(self, simulation):
        """Overload attach method to include simulation object
        """
        super().attach(simulation)
        self._sim = simulation
        

class WTmetadUpdater(CustomAction):
    """
    Custom alchemical updater for HPMC integrator. This updater performs trial moves on alchemical
    parameters.
    """
    def __init__(self, rng, h0, sigma, T, dT, stride, calc_ebetac=True):
        super().__init__()
        self._counters = [0, 0]
        self._rng = rng
        self.h0 = h0
        self.sigma = sigma
        self.T = T
        self.dT = dT
        self.stride = stride
        self.bias_factor = (T+dT)/T
        
        self.current_op = 0.
        self.op_bias_arr = []
        self.hbias_arr = []
        self.current_vbias = 0.
        self.current_ebetac = 0.
        
        self.calc_ebetac = calc_ebetac
        
    def set_init_snapshot(self, snap):
        self.prev = snap
        
    def compute_vbias(self, op):
        
        op = np.asarray(op)
        dop = op-self.op_bias_arr
        vbias = self.hbias_arr * np.exp(-0.5*dop**2/self.sigma**2)
        vbias_tot = np.sum(vbias)
        
        return vbias_tot

    def compute_ebetac(self):

        if not self.calc_ebetac or len(self.op_bias_arr)==0:
            return 1
        
        gamma = self.bias_factor
        # ds will be cancelled out.
        # set up integration space for betav
        min_op = min(self.op_bias_arr)
        max_op = max(self.op_bias_arr)
        interval_op = (max_op-min_op)/100
        op_space = min_op + np.arange(100) * interval_op
        
        betav = np.array([self.compute_vbias(op_i) for op_i in op_space])
        num = np.sum(np.exp(gamma/(gamma-1) * betav))
        den = np.sum(np.exp(1/(gamma-1) * betav))
        ebetac = num/den

        return ebetac
        
    def act(self, timestep):
        
        # before HPMC trial move
        prev_op = compute_op(self.prev)
        prev_vbias = self.compute_vbias(prev_op)

        # after HPMC trial move
        trial = self._sim.state.get_snapshot()
        trial_op = compute_op(trial)
        trial_vbias = self.compute_vbias(trial_op)

        pbias = np.exp(-(trial_vbias-prev_vbias)) # beta in HPMC default to be 1
        
        if  self._rng.random() < pbias:
            # move was accepted
            self.prev = trial
            self._counters[0] += 1
            self.current_op = trial_op
            self.current_vbias = trial_vbias
            
        else:
            # move was rejected
            self._sim.state.set_snapshot(self.prev)
            self._counters[1] += 1
            self.current_vbias = prev_vbias
        
        self.current_ebetac = self.compute_ebetac()
        if timestep%self.stride==0:
            self.op_bias_arr.append(self.current_op)
            current_hbias = self.h0 * np.exp(-self.current_vbias/self.dT)
            self.hbias_arr.append(current_hbias)
            
            
    @hoomd.logging.log(category='scalar', requires_run=True)
    def op(self):
        return self.current_op
    
    @hoomd.logging.log(category='scalar', requires_run=True)
    def vbias(self):
        return self.current_vbias
    
    @hoomd.logging.log(category='scalar', requires_run=True)
    def ebetac(self):
        return self.current_ebetac
        
    @hoomd.logging.log(category='sequence', requires_run=True)
    def bias_moves(self):
        return tuple(self._counters)
    