import sys

sys.path.append('/projects/b1030/hoomd/hoomd-2.5.1/')
sys.path.append('/projects/b1021/Yaohua/cg_protein/new_rigid/')
import hoomd
import hoomd.md
from Pentagon import PentagonBody
from PduABody import PduABody
from PduBBody import PduBBody
from NonBonded import LJ_attract
from NonBonded import *
from Solution import Lattice
from SphericalTemplate import SphericalTemplate
from PeanutTemplate import PeanutTemplate
import os

# Place the type R central particles
hoomd.context.initialize("--mode=cpu");
# rseed=42
# mer_mer = 4.0
# mer_scaffold = 2.0
anneal = True
note = 'LJG_pent1.1'
rseed = 42#os.environ['RSEED']
mer_mer = 4#os.environ['MERMER']
mer_scaffold = 4#os.environ['MER_TEMP']
scaffold_scaffold = 4#os.environ['TEMP_TEMP']
BMC = type('BMC', (object,), {})()
edge_l = 2.5
BMC.angle = 15 * np.pi / 180
hexamer1 = PduABody(edge_length=edge_l, angle=BMC.angle)
hexamer2 = PduBBody(edge_length=edge_l, angle=BMC.angle)
pentamer = PentagonBody(edge_length=edge_l, angle=BMC.angle)
a = 2.5
template = SphericalTemplate(a)
n_hex1 = 3#int(os.environ['N_HEX1'])
n_scaf = 3#int(os.environ['N_SCAF'])

try:
    n_hex2 = int(os.environ['N_HEX2'])
except:
    n_hex2 = 1

BMC.filename = str(note) + '_ee_' + str(scaffold_scaffold) + '_eh_' + str(mer_scaffold) + '_hh_' + str(
    mer_mer) + '_' + str(rseed)

sys = Lattice(pentamer, hexamer1, hexamer2, template, num_hex1=n_hex1, num_hex2=n_hex2, num_scaffold=n_scaf)

uc = hoomd.lattice.unitcell(N=sys.num_body,
                            a1=[25, 0, 0],
                            a2=[0, 25, 0],
                            a3=[0, 0, sys.cell_height],
                            dimensions=3,
                            position=sys.position_list,
                            type_name=sys.type_name_list,
                            mass=sys.mass_list,
                            moment_inertia=sys.moment_inertias,
                            orientation=sys.orientation_list)
system = hoomd.init.create_lattice(unitcell=uc, n=[4, 4, 4])

# Add constituent particles and create the rigid bodies

added_types = ['A', 'B', 'C', 'D', 'qP', 'qN', 'C1', 'D1', 'Ss']
if n_scaf > 0:
    added_types += ['Sc']
system_types = ['R', 'P', 'R2'] + added_types
if n_scaf > 0:
    system_types += ['H']
#print system_types
for new_type in added_types:
    system.particles.types.add(new_type)
#print(system.particles.types)
rigid = hoomd.md.constrain.rigid()
if n_scaf > 0:
    rigid.set_param('H', types=template.type_list, positions=template.position)
rigid.set_param('R', types=hexamer1.type_list, positions=hexamer1.all_sites)
rigid.set_param('P', types=pentamer.type_list, positions=pentamer.all_sites)

rigid.set_param('R2', types=hexamer2.type_list, positions=hexamer2.all_sites)

rigid.create_bodies()

hoomd.dump.gsd(BMC.filename + "-initial.gsd", group=hoomd.group.all(), overwrite=True, period=None)


sn = system.take_snapshot()
num_particle = len(sn.particles.position)
print(num_particle)
sn.particles.resize(num_particle+1)

# forcefield and integration
lB = 1.0
kp = 1.1
z_q = 8.0
A_yuka = z_q ** 2 * lB * (np.exp(kp * a) / (1 + kp * a)) ** 2

nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(system_types, system_types, func=SoftRepulsive, rmin=0.01, rmax=3,
                     coeff=dict(sigma=1, epsilon=1.0))
table.pair_coeff.set('Sc', system_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1.5, epsilon=0.5))
table.pair_coeff.set('C', 'D', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_mer)))
table.pair_coeff.set('C', 'D1', func=LJ_attract, rmin=0.01, rmax=3,
                     coeff=dict(sigma=1.1, epsilon=(0.9 * float(mer_mer))))
table.pair_coeff.set('C1', 'D', func=LJ_attract, rmin=0.01, rmax=3,
                     coeff=dict(sigma=1.1, epsilon=(0.9 * float(mer_mer))))
table.pair_coeff.set(['qP', 'C'], ['qP', 'C'], func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
table.pair_coeff.set(['qP', 'C'], 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=-5, kappa=kp))
table.pair_coeff.set('qN', 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
# table.pair_coeff.set('H', 'H', func=Yukawa, rmin=0.01, rmax=4.5, coeff=dict(A=A_yuka, kappa=kp))
if n_scaf > 0:
    table.pair_coeff.set('Sc', 'Sc', func=yukawa_lj, rmin=0.01, rmax=3,
                         coeff=dict(sigma=1.0, epsilon=float(scaffold_scaffold), A=1.20, kappa=kp))
    table.pair_coeff.set('Sc', 'Ss', func=normal_lj, rmin=0.01, rmax=3,
                         coeff=dict(sigma=1.0, epsilon=float(mer_scaffold)))

hoomd.md.integrate.mode_standard(dt=0.004)
rigid = hoomd.group.rigid_center()
integrator = hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=int(rseed))

hoomd.analyze.log(filename=BMC.filename + ".log",
                  quantities=['temperature', 'potential_energy',
                              'translational_kinetic_energy',
                              'rotational_kinetic_energy'],
                  period=10000,
                  overwrite=True)

hoomd.dump.gsd(BMC.filename + ".gsd",
               period=5e4,
               group=hoomd.group.all(),
               overwrite=True)

hoomd.run(1e7)

if anneal:
    temp_sequence = [1.05, 1.1, 1.05, 1.0]
    for temp in temp_sequence:
        integrator.disable()
        integrator = hoomd.md.integrate.langevin(group=rigid, kT=temp, seed=int(rseed))
        hoomd.run(2e6)

hoomd.run(2e7)

hoomd.dump.gsd(BMC.filename + "final-frame.gsd", group=hoomd.group.all(), overwrite=True, period=None)
