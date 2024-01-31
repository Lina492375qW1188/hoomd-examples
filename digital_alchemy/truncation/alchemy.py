import numpy as np

import hoomd
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
        
class AlchemUpdater(CustomAction):
    """
    Custom alchemical updater for HPMC integrator. This updater performs trial moves on alchemical
    parameters.
    """
    def __init__(self, stepsize, rng, alpha_init):
        self._stepsize = stepsize
        self._counters = [0, 0]
        self._rng = rng
        self.alpha_current = alpha_init
    
    def verts_from_alpha(self, alpha):
        
        platonic = coxeter.families.TruncatedTetrahedronFamily()
        particle = platonic.get_shape(truncation=alpha)
        
        return particle.vertices/particle.volume**(1/3)
    
    def act(self, timestep):
        
        # old shape
        alpha_prev = self.alpha_current
        verts_prev = self._sim._operations.integrator.shape['A']
        
        # try a move
        step = (2 * self._rng.random() - 1) * self._stepsize
        alpha_trial = alpha_prev + step
        if alpha_trial>1: alpha_trial=1
        elif alpha_trial<0: alpha_trial=0
        
        # trial shape
        verts_trial = self.verts_from_alpha(alpha_trial)
        self._sim._operations.integrator.shape['A'] = {"vertices": verts_trial}
        
        overlaps = self._sim._operations.integrator.overlaps
        if  overlaps==0:
            # move was accepted
            self._counters[0] += 1
            self.alpha_current = alpha_trial
            
        else:
            # move was rejected
            self._counters[1] += 1
            self._sim._operations.integrator.shape['A'] = verts_prev
            
    @hoomd.logging.log(category='scalar', requires_run=True)
    def alpha(self):
        return self.alpha_current

    @hoomd.logging.log(category='sequence', requires_run=True)
    def alchem_moves(self):
        return tuple(self._counters)
    
