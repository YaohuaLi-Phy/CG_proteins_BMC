from __future__ import division
from NonBonded import *
import hoomd
import hoomd.md
hoomd.context.initialize("");
uc = hoomd.lattice.unitcell(N = 1,
                            a1 = [10.8, 0,   0],
                            a2 = [0,    1.2, 0],
                            a3 = [0,    0,   1.2],
                            dimensions = 3,
                            position = [[0,0,0]],
                            type_name = ['R'],
                            mass = [1.0],
                            moment_inertia = [[0,
                                               1/12*1.0*8**2,
                                               1/12*1.0*8**2]],
                            orientation = [[1, 0, 0, 0]]);
system = hoomd.init.create_lattice(unitcell=uc, n=[2,18,18]);
system.particles.types.add('A')
rigid = hoomd.md.constrain.rigid()
rigid.set_param('R',
                types=['A']*8,
                positions=[(-4,0,0),(-3,0,0),(-2,0,0),(-1,0,0),
                           (1,0,0),(2,0,0),(3,0,0),(4,0,0)]);

rigid.create_bodies()
nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(['R', 'A'], ['R', 'A'],func=normal_lj, rmin=0.001, rmax=3, coeff=dict(sigma=1.0, epsilon=5.0))
#lj = hoomd.md.pair.lj(r_cut=2**(1/6), nlist=nl)
#lj.set_params(mode='shift')
#lj.pair_coeff.set(['R', 'A'], ['R', 'A'], epsilon=1.0, sigma=1.0)
hoomd.md.integrate.mode_standard(dt=0.001);
rigid = hoomd.group.rigid_center();
integrator=hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=42);
hoomd.analyze.log(filename="log-output.log",
                  quantities=['potential_energy',
                              'translational_kinetic_energy',
                              'rotational_kinetic_energy'],
                  period=100,
                  overwrite=True,
                  phase=-1);
hoomd.run(1e3);
integrator.disable()
for i in range(10):
    integrator = hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=4)
    hoomd.run(1000)
    integrator.disable()
