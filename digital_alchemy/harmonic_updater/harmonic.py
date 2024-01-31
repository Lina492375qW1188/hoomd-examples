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
        
class HarmonicUpdater(CustomAction):
    """
    Custom alchemical updater for HPMC integrator. This updater performs trial moves on alchemical
    parameters.
    """
    def __init__(self, ref_pos, init_size):
        self.ref_pos = ref_pos
        self.init_size = init_size
        self.rescale = 1
        self.boxsize = self.init_size
    
    def act(self, timestep):
        
        self.boxsize = self._sim._state.get_snapshot().configuration.box[0]
        self.rescale = self._sim._state.get_snapshot().configuration.box[0]/self.init_size
        self._sim._operations._integrator._external_potential._param_dict['reference_positions']=self.rescale * self.ref_pos
        
    @hoomd.logging.log(category='scalar', requires_run=True)
    def rescale_factor(self):
        return self.rescale
    
    @hoomd.logging.log(category='scalar', requires_run=True)
    def boxsize_t(self):
        return self.boxsize
    
