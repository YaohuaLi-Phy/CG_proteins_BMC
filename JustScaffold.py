import sys
sys.path.append('/projects/b1030/hoomd/hoomd-2.2.0_cuda8.0/')
import hoomd
import hoomd.md
from Pentagon import PentagonBody
from PduABody import PduABody
from PduBBody import PduBBody
from NonBonded import LJ_attract
from NonBonded import *
from Solution import TestLattice
from SphericalTemplate import SphericalTemplate
from PeanutTemplate import PeanutTemplate
import os
# Place the type R central particles
hoomd.context.initialize("--mode=gpu");
#rseed=42

anneal = False
note = 'sc_charged'
rseed=os.environ['RSEED']
#mer_mer=os.environ['MERMER']
#mer_scaffold=os.environ['MER_TEMP']
scaffold_scaffold=os.environ['TEMP_TEMP']
BMC = type('BMC', (object,), {})()
edge_l = 2.5
#hexamer1 = PduABody(edge_length=edge_l)
#hexamer2 = PduBBody(edge_length=edge_l)
#pentamer = PentagonBody(edge_length=edge_l)
a = 2.5
template = SphericalTemplate(a)
#n_hex1=int(os.environ['N_HEX1'])
#n_scaf=int(os.environ['N_SCAF'])
BMC.filename=str(note)+str(scaffold_scaffold) + '_'+str(rseed)

sys = TestLattice(num_mers=1, mer_size=2*a)

uc = hoomd.lattice.unitcell(N=sys.num_body,
                            a1=[sys.box_l, 0, 0],
                            a2=[0, sys.box_l, 0],
                            a3=[0, 0, sys.box_l],
                            dimensions=3,
                            position=sys.position_list,
                            type_name=sys.type_name_list,
                            mass=sys.mass_list,
                            moment_inertia=sys.moment_inertias,
                            orientation=sys.orientation_list)
system = hoomd.init.create_lattice(unitcell=uc, n=[5, 5, 5])

# Add constituent particles of type A and create the rods

added_types = ['Sc']
system_types = ['R'] + added_types
for new_type in added_types:
    system.particles.types.add(new_type)
#system.particles.types.add('A');
rigid = hoomd.md.constrain.rigid()

rigid.set_param('R', types=template.type_list, positions=template.position)

rigid.create_bodies()

# forcefield and integration
kp = 1.0

nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(system_types, system_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1, epsilon=1.0))
table.pair_coeff.set('Sc', system_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1.5, epsilon=0.5))
table.pair_coeff.set('Sc', 'Sc', func=yukawa_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(scaffold_scaffold), A=1.20, kappa=kp))


hoomd.md.integrate.mode_standard(dt=0.004)
rigid = hoomd.group.rigid_center()
integrator = hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=int(rseed))

hoomd.analyze.log(filename=BMC.filename + ".log",
                  quantities=['temperature', 'potential_energy',
                              'translational_kinetic_energy',
                              'rotational_kinetic_energy'],
                  period=10000,
                  overwrite=True)

hoomd.dump.gsd(BMC.filename+".gsd",
               period=1e4,
               group=hoomd.group.all(),
               overwrite=True)

hoomd.run(1e6)

if anneal:
    temp_sequence=[1.1, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.2, 1.0]
    for temp in temp_sequence:
        integrator.disable()
        integrator = hoomd.md.integrate.langevin(group=rigid, kT=temp, seed=int(rseed))
        hoomd.run(2e6)

integrator.disable()
integrator = hoomd.md.integrate.npt(group=rigid, kT=1.0, tau=1.0, P=0.001, tauP=1.0)
hoomd.run(1e6)

hoomd.dump.gsd(BMC.filename+"final-frame.gsd", group=hoomd.group.all(), overwrite=True, period=None)
