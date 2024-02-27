import os
import time
import math
import itertools
import numpy as np

import hoomd
import gsd.hoomd
import coxeter


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


class MetaAlchemUpdater(CustomAction):
    """
    Custom meta-alchemical updater for HPMC integrator. This updater performs trial moves on alchemical
    parameters.
    """
    def __init__(self, stepsize, rng, alpha_init):
        self._stepsize = stepsize
        self._counters = [0, 0]
        self._rng = rng
        self.alpha_current = alpha_init
        
    def set_metad_param(self, rng, h0, sigma, T, dT, stride, calc_ebetac=True):
        
        self._rng_metad = rng
        self.h0 = h0
        self.sigma = sigma
        self.T = T
        self.dT = dT
        self.stride = stride
        self.bias_factor = (T+dT)/T
        
        self.current_op = self.alpha_current
        
        self.op_bias_arr = []
        self.hbias_arr = []
        self.current_vbias = 0.0
        self.current_ebetac = 0.0
        
        self.calc_ebetac = calc_ebetac
        
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
        bin_op = 20
        interval_op = (max_op-min_op)/bin_op
        op_space = min_op + np.arange(bin_op) * interval_op
        
        betav = np.array([self.compute_vbias(op_i) for op_i in op_space])
        num = np.sum(np.exp(gamma/(gamma-1) * betav))
        den = np.sum(np.exp(1/(gamma-1) * betav))
        ebetac = num/den

        return ebetac
    
    def verts_from_alpha(self, alpha):
        
        f = coxeter.families.Family323Plus()
        particle = f.get_shape(alpha, 0.2*alpha+0.8)
        
        return particle.vertices/particle.volume**(1/3)
    
    def act(self, timestep):
        
        # alchemical part
        # old shape
        alpha_prev = self.alpha_current
        shape_prev = self._sim._operations.integrator.shape['A']
        
        # try a move
        step = (2 * self._rng.random() - 1) * self._stepsize
        alpha_trial = alpha_prev + step
        if alpha_trial>3: alpha_trial=3
        elif alpha_trial<1: alpha_trial=1
        
        # trial shape
        verts_trial = self.verts_from_alpha(alpha_trial)
        self._sim._operations.integrator.shape['A'] = {"vertices": verts_trial}
        
        # metadynamics part
        # set old shape truncation as prev_op
        prev_op = alpha_prev
        prev_vbias = self.compute_vbias(prev_op)
        
        # set trial shape truncation as trial_op
        trial_op = alpha_trial
        trial_vbias = self.compute_vbias(trial_op)
        
        pbias = np.exp(-(trial_vbias-prev_vbias)) # beta in HPMC default to be 1
        
        overlaps = self._sim._operations.integrator.overlaps
        if overlaps==0 and self._rng_metad.random() < pbias:
            # move was accepted
            self._counters[0] += 1
            self.alpha_current = alpha_trial
            self.current_op = trial_op
            self.current_vbias = trial_vbias

        else:
            # move was rejected
            self._counters[1] += 1
            self._sim._operations.integrator.shape['A'] = shape_prev
            self.current_vbias = prev_vbias
            
        self.current_ebetac = self.compute_ebetac()
        if timestep%self.stride==0:
            self.op_bias_arr.append(self.current_op)
            current_hbias = self.h0 * np.exp(-self.current_vbias/self.dT)
            self.hbias_arr.append(current_hbias)
            
    @hoomd.logging.log(category='scalar', requires_run=True)
    def alpha(self):
        return self.alpha_current
    
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
    def alchem_moves(self):
        return tuple(self._counters)
    