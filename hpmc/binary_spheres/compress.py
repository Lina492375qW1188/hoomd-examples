import time
import numpy as np

import hoomd


#### initializing snapshot
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=59920)
sim.create_state_from_gsd(filename='./DATA/randomize.gsd', frame=-1)


#### integrator
def compute_d1d2(alpha):
        
    const = (4*np.pi/3)**(-1/3)
    volume_of_two_types = 2.0
    pre_factor = const * (volume_of_two_types)**(1/3)

    r1 = pre_factor * (1/(1+alpha**3))**(1/3)
    r2 = alpha * r1

    return 2*r1, 2*r2

alpha = 0.42
d1, d2 = compute_d1d2(alpha)

mc = hoomd.hpmc.integrate.Sphere(default_d=0.3, default_a=0.4)
mc.shape["A"] = dict(diameter=d1)
mc.shape["B"] = dict(diameter=d2)


#### quick compress
# calculate volume of each particle
r_particle = 0.5 * np.array(compute_d1d2(0.42))
V_particle = (4*np.pi/3) * r_particle**3
idx = sim.state.get_snapshot().particles.typeid
V = np.sum(V_particle[idx])

final_volume_fraction = 0.57

initial_box = sim.state.box
final_box = hoomd.Box.from_box(initial_box)
final_box.volume = V / final_volume_fraction
compress = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(100),
                                           target_box=final_box)

periodic = hoomd.trigger.Periodic(10)
tune = hoomd.hpmc.tune.MoveSize.scale_solver(moves=['a', 'd'],
                                             target=0.2,
                                             trigger=periodic,
                                             max_translation_move=0.2,
                                             max_rotation_move=0.2)


#### write
logger = hoomd.logging.Logger()
logger.add(mc, quantities=['type_shapes'])

gsd_writer = hoomd.write.GSD(filename='./DATA/compress.gsd',
                             trigger=hoomd.trigger.Periodic(100),
                             mode='wb',
                             filter=hoomd.filter.All(),
                             log=logger)


#### attaching operations
sim.operations.writers.append(gsd_writer)
sim.operations.integrator = mc
sim.operations.updaters.append(compress)
sim.operations.tuners.append(tune)


#### run simulation
while not compress.complete and sim.timestep < 1e6:
    sim.run(1000)
    
    
#### check compress
if not compress.complete:
    raise RuntimeError("Compression failed to complete")