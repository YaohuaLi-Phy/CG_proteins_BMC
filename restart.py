import sys
sys.path.append('/projects/b1030/hoomd/hoomd-2.2.0_cuda8.0/')
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
hoomd.context.initialize("--mode=gpu");
rseed=27154
mer_mer = 4.0
mer_scaffold = 6.0
anneal = False
#rseed=os.environ['RSEED']
#mer_mer=os.environ['MERMER']
#mer_scaffold=os.environ['MER_TEMP']
scaffold_scaffold=2.6  #os.environ['TEMP_TEMP']
BMC = type('BMC', (object,), {})()
edge_l = 2.5
hexamer1 = PduABody(edge_length=edge_l)
hexamer2 = PduBBody(edge_length=edge_l)
pentamer = PentagonBody(edge_length=edge_l)
a = 2.5
template = SphericalTemplate(a)

read_file = 'scan_Q1.2_ee_2.6_eh_6.0_hh_4.0_10222.gsd'
BMC.filename='continue'+'n_hex1_' + read_file
system = hoomd.init.read_gsd('scan_Q1.2_ee_2.6_eh_6.0_hh_4.0_10222.gsd', frame=759)

# forcefield and integration
kp = 1.1

added_types = ['A', 'B', 'C', 'D', 'C1', 'D1', 'qP', 'qN', 'Sc', 'Ss']
system_types = ['R', 'P', 'H', 'R2'] + added_types
for new_type in added_types:
    system.particles.types.add(new_type)
rigid = hoomd.md.constrain.rigid()
rigid.set_param('R', types=hexamer1.type_list, positions=hexamer1.all_sites)
rigid.set_param('P', types=pentamer.type_list, positions=pentamer.all_sites)
rigid.set_param('H', types=template.type_list, positions=template.position)
rigid.set_param('R2', types=hexamer2.type_list, positions=hexamer2.all_sites)


rigid.create_bodies()

nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(system_types, system_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1, epsilon=1.0))
table.pair_coeff.set('Sc', system_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1.5, epsilon=0.5))
table.pair_coeff.set('C', 'D', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_mer)))
table.pair_coeff.set('C', 'D1', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_mer)))
table.pair_coeff.set('C1', 'D', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_mer)))
table.pair_coeff.set(['qP', 'C'], ['qP', 'C'], func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
table.pair_coeff.set(['qP', 'C'], 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=-5, kappa=kp))
table.pair_coeff.set('qN', 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
# table.pair_coeff.set('H', 'H', func=Yukawa, rmin=0.01, rmax=4.5, coeff=dict(A=A_yuka, kappa=kp))
table.pair_coeff.set('Sc', 'Sc', func=yukawa_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(scaffold_scaffold), A=1.20, kappa=kp))
table.pair_coeff.set('Sc', 'Ss', func=normal_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_scaffold)))


hoomd.md.integrate.mode_standard(dt=0.003)
to_integrate = hoomd.group.rigid_center()
integrator = hoomd.md.integrate.langevin(group=to_integrate, kT=0.9, seed=int(rseed))


hoomd.dump.gsd(BMC.filename+"new.gsd",
               period=2e4,
               group=hoomd.group.all(), overwrite=True)

hoomd.run(5e6)
