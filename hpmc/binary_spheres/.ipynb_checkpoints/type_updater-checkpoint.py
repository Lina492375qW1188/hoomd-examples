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
        
class TypeUpdater(CustomAction):
    """
    Custom alchemical updater for HPMC integrator. This updater performs trial moves on alchemical
    parameters.
    """
    def __init__(self, rng):
        self._rng = rng
        self._counters = [0, 0]
    
    def act(self, timestep):
        
        # old snapshot
        old_snap = self._sim.state.get_snapshot()
        N_particles = old_snap.particles.N
        idx = np.random.choice(N_particles, 5)
        
        # trial
        with self._sim.state.cpu_local_snapshot as snapshot:
            idx0 = np.where(snapshot.particles.typeid[idx]==0)
            idx1 = np.where(snapshot.particles.typeid[idx]==1)

            snapshot.particles.typeid[idx[idx0]]=1
            snapshot.particles.typeid[idx[idx1]]=0

        overlaps = self._sim._operations.integrator.overlaps
        if  overlaps==0:
            # move was accepted
            self._counters[0] += 1
            
        else:
            # move was rejected
            self._counters[1] += 1
            self._sim.state.set_snapshot(old_snap)
            

    @hoomd.logging.log(category='sequence', requires_run=True)
    def moves(self):
        return tuple(self._counters)