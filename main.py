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
from SphericalTemplate import *
from PeanutTemplate import PeanutTemplate
import os
import hoomd.deprecated as hdepr

'''run options: 
no pentamer: set n_pent to 0, remove particle type P in particle type list and set param (2 lines)
no scaffold: set n_scaf to 0, it automatically takes care

'''

LOCAL = False
anneal = True

kp = 1.0
# Place the type R central particles
if LOCAL:
    hoomd.context.initialize("--mode=cpu")
    rseed=42
    mer_mer = 4.0
    mer_scaffold = 2.0
    scaffold_scaffold = 1.8
    n_hex1 = 1
    n_scaf = 1
    n_pent = 1
    pent_coeff = 0.85
    a = 2.5
    note = 'local_model'
    lB = 0.7

else:
    note = os.environ['NOTE']
    hoomd.context.initialize("--mode=gpu")
    rseed = os.environ['RSEED']
    mer_mer = float(os.environ['MERMER'])
    mer_scaffold = os.environ['MER_TEMP']
    scaffold_scaffold = os.environ['TEMP_TEMP']
    n_hex1 = int(os.environ['N_HEX1'])
    n_scaf = int(os.environ['N_SCAF'])
    n_pent = int(os.environ['N_pent'])
    pent_coeff = float(os.environ['pent_c'])
    a = float(os.environ['RADIUS'])
    angle = float(os.environ['angle'])
    lB = float(os.environ['lB'])
    hand_l=float(os.environ['r0'])
    try:
        coeff_b = float(os.environ['B_factor'])
    except:
        coeff_b = 1.0
BMC = type('BMC', (object,), {})()

edge_l = a
BMC.angle = angle * np.pi / 180
hexamer1 = PduABody(edge_length=edge_l, angle=BMC.angle)
hexamer2 = PduBBody(edge_length=edge_l, angle=BMC.angle)
pentamer = PentagonBody(edge_length=edge_l, angle=BMC.angle)

template = SphericalTemplate(a)

try:
    n_hex2 = int(os.environ['N_HEX2'])
except:
    n_hex2 = 1

BMC.filename = str(note) +'_n1_' + str(n_hex1) +'_nh2_' + str(n_hex2) +'_'+str(n_pent)+ '_ee_' + str(mer_scaffold) + '_hh_' + str(
    mer_mer) + 'ph_'+str(pent_coeff) + '_' + str(rseed)

sys = Lattice(pentamer, hexamer1, hexamer2, template, num_pen=n_pent, num_hex1=n_hex1, num_hex2=n_hex2, num_scaffold=n_scaf)

uc = hoomd.lattice.unitcell(N=sys.num_body,
                            a1=[(20+2.5*a), 0, 0],
                            a2=[0, (20+2.5*a), 0],
                            a3=[0, 0, sys.cell_height],
                            dimensions=3,
                            position=sys.position_list,
                            type_name=sys.type_name_list,
                            mass=sys.mass_list,
                            moment_inertia=sys.moment_inertias,
                            orientation=sys.orientation_list)
system = hoomd.init.create_lattice(unitcell=uc, n=[5, 5, 5])

# Add constituent particles and create the rigid bodies

added_types = ['A', 'B', 'C', 'D', 'qP', 'qN', 'C1', 'D1', 'Ss', 'C2', 'D2']
if n_scaf > 0:
    added_types += ['Sc']
system_types = added_types
if n_hex1 > 0:
    system_types += ['R']
if n_hex2 > 0:
    system_types += ['R2']
if n_pent > 0:
    system_types += ['P']
if n_scaf > 0:
    system_types += ['H']
#print system_types
for new_type in added_types:
    system.particles.types.add(new_type)
print(system.particles.types)
rigid = hoomd.md.constrain.rigid()

if n_hex1 > 0:
    rigid.set_param('R', types=hexamer1.type_list, positions=hexamer1.all_sites)
if n_scaf > 0:
    rigid.set_param('H', types=template.type_list, positions=template.position)

if n_pent > 0:
    rigid.set_param('P', types=pentamer.type_list, positions=pentamer.all_sites)

if n_hex2 > 0:
    rigid.set_param('R2', types=hexamer2.type_list, positions=hexamer2.all_sites)

rigid.create_bodies()

hoomd.dump.gsd(BMC.filename + "-initial.gsd", group=hoomd.group.all(), overwrite=True, period=None)


#sn = system.take_snapshot()
#num_particle = len(sn.particles.position)
#print(num_particle)
#sn.particles.resize(num_particle+1)

# forcefield and integration

z_q = 0.80
A_yuka = z_q ** 2 * lB * (np.exp(kp * a) / (1 + kp * a)) ** 2

nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(system_types, system_types, func=SoftRepulsive, rmin=0.01, rmax=3,
                     coeff=dict(sigma=1, epsilon=1.0))
table.pair_coeff.set('Sc', system_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=0.5))

table.pair_coeff.set('C', 'D', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_mer), r0=hand_l))
table.pair_coeff.set('C2', 'D2', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(coeff_b*float(mer_mer)), r0=hand_l))
table.pair_coeff.set('C2', 'D', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(0.5*(1.0+coeff_b)*(mer_mer)), r0=hand_l))
table.pair_coeff.set('C', 'D2', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(0.5*(1.0+coeff_b)*(mer_mer)), r0=hand_l))
table.pair_coeff.set(['C', 'C2'], 'D1', func=LJ_attract, rmin=0.01, rmax=3,
                     coeff=dict(sigma=1.0, epsilon=(pent_coeff * float(mer_mer)), r0=hand_l))
table.pair_coeff.set('C1', ['D', 'D2'], func=LJ_attract, rmin=0.01, rmax=3,
                     coeff=dict(sigma=1.0, epsilon=(pent_coeff * float(mer_mer)), r0=hand_l))
table.pair_coeff.set(['qP', 'C'], ['qP', 'C'], func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=A_yuka, kappa=kp))  # A=5 for previous data
table.pair_coeff.set(['qP', 'C'], 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=-A_yuka, kappa=kp))
table.pair_coeff.set('qN', 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=A_yuka, kappa=kp))
#table.pair_coeff.set('qN', 'qn', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=2, kappa=kp))
#table.pair_coeff.set(['qP', 'C'], 'qn', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=-2, kappa=kp))

if n_scaf > 0:
    table.pair_coeff.set('Sc', 'Sc', func=yukawa_lj, rmin=0.01, rmax=3,
                         coeff=dict(sigma=1.0, epsilon=float(scaffold_scaffold), A=A_yuka, kappa=kp))
    #table.pair_coeff.set('qn', 'qn', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=1, kappa=kp))
    table.pair_coeff.set('Sc', 'Ss', func=normal_lj, rmin=0.01, rmax=3,
                         coeff=dict(sigma=1.0, epsilon=float(mer_scaffold)))

hoomd.md.integrate.mode_standard(dt=0.003)
rigid = hoomd.group.rigid_center()
integrator = hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=int(rseed))

hoomd.analyze.log(filename=BMC.filename + ".log",
                  quantities=['temperature', 'potential_energy',
                              'translational_kinetic_energy',
                              'rotational_kinetic_energy','pair_table_energy'],
                  period=10000,
                  overwrite=True)

hoomd.dump.gsd(BMC.filename + ".gsd",
               period=2e5,
               group=hoomd.group.all(),
               overwrite=True)

hoomd.run(1e7)

hoomd.dump.gsd(BMC.filename + "mid1.gsd", group=hoomd.group.all(), overwrite=True, period=None)
if anneal:
    temp_sequence = [1.05, 1.1, 1.15, 1.1, 1.05, 1.0]
    for temp in temp_sequence:
        integrator.disable()
        integrator = hoomd.md.integrate.langevin(group=rigid, kT=temp, seed=int(rseed))
        hoomd.run(2e6)

hoomd.run(1e7)
hoomd.dump.gsd(BMC.filename + "mid2.gsd", group=hoomd.group.all(), overwrite=True, period=None)
if anneal:
    temp_sequence = [1.05, 1.1, 1.15, 1.1, 1.05, 1.0]
    for temp in temp_sequence:
        integrator.disable()
        integrator = hoomd.md.integrate.langevin(group=rigid, kT=temp, seed=int(rseed))
        hoomd.run(2e6)
hoomd.run(1e7)

#hoomd.dump.gsd(BMC.filename + "mid3.gsd", group=hoomd.group.all(), overwrite=True, period=None)
hoomd.dump.gsd(BMC.filename + "final-frame.gsd", group=hoomd.group.all(), overwrite=True, period=None)
