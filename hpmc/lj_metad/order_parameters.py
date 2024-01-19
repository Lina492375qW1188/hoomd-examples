import numpy as np
import freud

qc = freud.order.Steinhardt(l=6, average=True)
def compute_qc(snapshot):
    """
    Q_l
    """
    box = snapshot.configuration.box
    points = snapshot.particles.position

    system = freud.AABBQuery(box, points)
    args = {"num_neighbors": 6, "exclude_ii": True}
    nlist = system.query(points, args).toNeighborList()

    qc.compute(system, neighbors=nlist)
    return qc.order

def compute_num_liq(snapshot):
    """
    Contiuous number of liquids
    """
    r_cut = 2.0
    box = snapshot.configuration.box
    points = snapshot.particles.position

    system = freud.AABBQuery(box, points)
    args = {"num_neighbors": 20, "exclude_ii": True}
    nlist = system.query(points, args).toNeighborList()
    d6 = (nlist.distances.reshape(20,-1)/r_cut)**6
    d12 = d6**2
    cij = (1-d6)/(1-d12+1e-10)
    ci = np.sum(cij, axis=0)
    
    c_l = 5
    clci6 = (c_l/ci)**6
    clci12 = clci6**2

    num_liq = np.sum( (1-clci6)/(1-clci12) )
    return num_liq