import sys
#sys.path.append('/projects/b1030/hoomd/hoomd-2.2.0_cuda8.0/')
import hoomd
import hoomd.md
from Pentagon import PentagonBody
from PduABody import PduABody
from PduBBody import PduBBody
from NonBonded import LJ_attract
from NonBonded import *
from Solution import Lattice
from SphericalTemplate import SphericalTemplate
import os
# Place the type R central particles
hoomd.context.initialize("--mode=cpu");
#rseed=os.environ['RSEED']
#mer_mer=os.environ['MERMER']
#mer_scaffold=os.environ['MER_TEMP']
rseed=42
BMC = type('BMC', (object,), {})()

mer_mer = 4.0
mer_scaffold = 2.0
hexamer1 = PduABody()
hexamer2 = PduBBody()
pentamer = PentagonBody()
template = SphericalTemplate(1.0)
BMC.filename='Mer_' + str(mer_mer)+'_Scaf_'+str(mer_scaffold)+'_'+str(rseed)

sys = Lattice(pentamer, hexamer1, hexamer2, num_hex1=3)

uc = hoomd.lattice.unitcell(N=sys.num_body,
                            a1=[15, 0, 0],
                            a2=[0, 15, 0],
                            a3=[0, 0, sys.cell_height],
                            dimensions=3,
                            position=sys.position_list,
                            type_name=sys.type_name_list,
                            mass=sys.mass_list,
                            moment_inertia=sys.moment_inertias,
                            orientation=sys.orientation_list)
system = hoomd.init.create_lattice(unitcell=uc, n=[3, 3, 3])

# Add constituent particles of type A and create the rods

added_types = ['A', 'B', 'C', 'D', 'qP', 'qN', 'Sc', 'Ss']
system_types = ['R', 'P', 'H','R2'] + added_types
for new_type in added_types:
    system.particles.types.add(new_type)
#system.particles.types.add('A');
rigid = hoomd.md.constrain.rigid()

rigid.set_param('R', types=hexamer1.type_list, positions=hexamer1.all_sites)
rigid.set_param('P', types=pentamer.type_list, positions=pentamer.all_sites)
rigid.set_param('H', types=template.type_list, positions=template.position)
rigid.set_param('R2', types=hexamer2.type_list, positions=hexamer2.all_sites)

rigid.create_bodies()

# forcefield and integration
nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(system_types, system_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1, epsilon=1.0))
table.pair_coeff.set('C', 'D', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=5))
table.pair_coeff.set('qP', 'qP', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=0.1))
table.pair_coeff.set('qP', 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=-5,kappa=0.1))
table.pair_coeff.set('Sc', 'Sc', func=normal_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=mer_mer))
table.pair_coeff.set('Sc', 'Ss', func=normal_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=mer_scaffold))

hoomd.md.integrate.mode_standard(dt=0.004)
rigid = hoomd.group.rigid_center()
hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=rseed);

hoomd.analyze.log(filename=BMC.filename + ".log",
                  quantities=['potential_energy',
                              'translational_kinetic_energy',
                              'rotational_kinetic_energy'],
                  period=10000,
                  overwrite=True)

hoomd.dump.gsd(BMC.filename+".gsd",
               period=5e2,
               group=hoomd.group.all(),
               overwrite=True)

hoomd.run(1e6)
