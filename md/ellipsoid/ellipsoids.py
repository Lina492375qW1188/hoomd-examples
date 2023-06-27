#!/usr/bin/env python
# coding: utf-8
import gsd.hoomd
import hoomd
import matplotlib
import math
import numpy as np
import datetime
import itertools
import freud
import warnings
import fresnel
import IPython
import packaging.version
import coxeter


elle = coxeter.shapes.Ellipsoid(1.5, 0.5, 0.5)
sigma = elle.maximal_bounded_sphere_radius * 2
moment_inertia = elle.inertia_tensor.diagonal()

N = 10
spacing = 4.0
K = math.ceil(N**(1 / 3))
L = K * spacing
x = np.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))
print(position[0:4])

position = position[0:N]
orientation = [(1, 0, 0, 0)] * N

snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N
snapshot.particles.position = position
snapshot.particles.orientation = orientation
snapshot.particles.moment_inertia = [moment_inertia] * N
s
snapshot.particles.typeid = [0] * N
snapshot.particles.types = ['A']
snapshot.configuration.box = [L, L, L, 0, 0, 0]

initial_snapshot = snapshot


with gsd.hoomd.open(name='initial_lattice_ellipsoids.gsd', mode='wb') as f:
    f.append(snapshot)

    
    
    

    
dt = 0.0005
tau = 100*dt
tauS = 1000*dt
    
    
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=1)
sim.create_state_from_gsd(filename='initial_lattice_ellipsoids.gsd')
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    
nl = hoomd.md.nlist.Cell(buffer=0.4)
alj = hoomd.md.pair.aniso.ALJ(nl)
alj.r_cut[('A', 'A')] = 2 * elle.a + 0.15 * sigma

alj.params[("A", "A")] = dict(epsilon = 1.0,
                              sigma_i = sigma,
                              sigma_j = sigma,
                              alpha = 0
                              )
alj.shape["A"] = dict(rounding_radii = (elle.a, elle.b, elle.c), vertices = [], faces = [])

nvt = hoomd.md.methods.NVT(kT=1.0, filter=hoomd.filter.All(), tau=tau)

# npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), 
#                            tau=tau, 
#                            kT=1.0, 
#                            tauS=tauS, 
#                            S=1.0, 
#                            couple="xyz")

integrator = hoomd.md.Integrator(dt=dt, 
                                 methods=[nvt],
                                 forces=[alj],
                                 integrate_rotational_dof=True)


sim.operations.integrator = integrator

sim.run(0)

logger = hoomd.logging.Logger()
logger.add(alj, quantities=['type_shapes'])
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
logger.add(thermodynamic_properties)
    
gsd_writer = hoomd.write.GSD(filename='trajectory_ellipsoids.gsd',
                             trigger=hoomd.trigger.Periodic(10),
                             mode='wb',
                             filter=hoomd.filter.All(),
                             log=logger)
sim.operations.writers.append(gsd_writer)

sim.run(1e5)