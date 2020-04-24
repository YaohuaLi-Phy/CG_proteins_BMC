import sys

sys.path.append('/projects/b1030/hoomd/hoomd-2.5.1/')
sys.path.append('/projects/b1021/Yaohua/cg_protein/new_rigid/')
import hoomd
import hoomd.md
from Pentagon import PentagonBody
from BondPduA import BondPduA
from PduBBody import PduBBody
from NonBonded import LJ_attract
from NonBonded import *
from BondSolution import UnitCell
from SphericalTemplate import *
from PeanutTemplate import PeanutTemplate
import os
import hoomd.deprecated as hdepr

'''run options: 
no pentamer: set n_pent to 0, remove particle type P in particle type list and set param (2 lines)
no scaffold: set n_scaf to 0, it automatically takes care

'''

try:
    LOCAL = os.environ['LOCAL']
except:
    LOCAL = True
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
    n_scaf = 0
    n_pent = 0
    pent_coeff = 0.85
    a = 2.5
    note = 'local_model'

else:
    note = os.environ['NOTE']
    hoomd.context.initialize("--mode=gpu")
    rseed = os.environ['RSEED']
    mer_mer = os.environ['MERMER']
    mer_scaffold = os.environ['MER_TEMP']
    scaffold_scaffold = os.environ['TEMP_TEMP']
    n_hex1 = int(os.environ['N_HEX1'])
    n_scaf = int(os.environ['N_SCAF'])
    n_pent = int(os.environ['N_pent'])
    pent_coeff = float(os.environ['pent_c'])
    a = float(os.environ['RADIUS'])
BMC = type('BMC', (object,), {})()

edge_l = a
BMC.angle = 15 * np.pi / 180
hexamer1 = BondPduA(edge_length=edge_l, angle=BMC.angle)

#pentamer = PentagonBody(edge_length=edge_l, angle=BMC.angle)
#template = SphericalTemplate(a)

try:
    n_hex2 = int(os.environ['N_HEX2'])
except:
    n_hex2 = 0

BMC.filename = str(note) +'_n1_' + str(n_hex1) +'_nh2_' + str(n_hex2) + '_ee_' + str(mer_scaffold) + '_hh_' + str(
    mer_mer) + 'ph_'+str(pent_coeff) + '_' + str(rseed)

u_cell = UnitCell( hexamer1, num_pen=n_pent, num_hex1=n_hex1, num_hex2=n_hex2, num_scaffold=n_scaf)

blx=10
bly=10
#snapshot = hoomd.data.make_snapshot(N=num_particles, particle_types=hexamer1.type_list, box=hoomd.data.boxdim(Lx=blx, Ly=bly, Lz=bly))
#snapshot.box = hoomd.data.boxdim(Lx=blx, Ly=bly, Lz=bly, xy=0.0, xz=0.0, yz=0.0)  # elongated in x direction

system = u_cell.create_system()
for type in u_cell.p_types:
    system.particles.types.add(type)

# Add constituent particles and create the spring connected protein

hoomd.dump.gsd(BMC.filename + "-initial.gsd", group=hoomd.group.all(), overwrite=True, period=None)

#sn = system.take_snapshot()
#num_particle = len(sn.particles.position)
#print(num_particle)
#sn.particles.resize(num_particle+1)

# forcefield and integration
lB = 1.0

z_q = 8.0
A_yuka = z_q ** 2 * lB * (np.exp(kp * a) / (1 + kp * a)) ** 2

nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(u_cell.p_types, u_cell.p_types, func=SoftRepulsive, rmin=0.01, rmax=3,
                     coeff=dict(sigma=1, epsilon=1.0))
table.pair_coeff.set('Sc', u_cell.p_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=0.5))
table.pair_coeff.set('C', 'D', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_mer)))
#table.pair_coeff.set('C', 'D1', func=LJ_attract, rmin=0.01, rmax=3,
#                     coeff=dict(sigma=1.0, epsilon=(pent_coeff * float(mer_mer))))
#table.pair_coeff.set('C1', 'D', func=LJ_attract, rmin=0.01, rmax=3,
#                     coeff=dict(sigma=1.0, epsilon=(pent_coeff * float(mer_mer))))
table.pair_coeff.set(['qP', 'C'], ['qP', 'C'], func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
table.pair_coeff.set(['qP', 'C'], 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=-5, kappa=kp))
table.pair_coeff.set('qN', 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
#table.pair_coeff.set('qN', 'qn', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=2, kappa=kp))
#table.pair_coeff.set(['qP', 'C'], 'qn', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=-2, kappa=kp))

if n_scaf > 0:
    table.pair_coeff.set('Sc', 'Sc', func=yukawa_lj, rmin=0.01, rmax=3,
                         coeff=dict(sigma=1.0, epsilon=float(scaffold_scaffold), A=1.20, kappa=kp))
    #table.pair_coeff.set('qn', 'qn', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=1, kappa=kp))
    table.pair_coeff.set('Sc', 'Ss', func=normal_lj, rmin=0.01, rmax=3,
                         coeff=dict(sigma=1.0, epsilon=float(mer_scaffold)))

hoomd.md.integrate.mode_standard(dt=0.003)
harmonic = hoomd.md.bond.harmonic(name='harmonic')
harmonic.bond_coeff.set('harmonic', k=200, r0=2.0)
ALL = hoomd.group.all()
integrator = hoomd.md.integrate.langevin(group=ALL, kT=1.0, seed=int(rseed))

hoomd.analyze.log(filename=BMC.filename + ".log",
                  quantities=['temperature', 'potential_energy',
                              'translational_kinetic_energy',
                              'rotational_kinetic_energy'],
                  period=10000,
                  overwrite=True)

hoomd.dump.gsd(BMC.filename + ".gsd",
               period=2e5,
               group=hoomd.group.all(),
               overwrite=True)

hoomd.run(1e7)

#hoomd.dump.gsd(BMC.filename + "mid1.gsd", group=hoomd.group.all(), overwrite=True, period=None)
if anneal:
    temp_sequence = [1.05, 1.1, 1.15, 1.1, 1.05, 1.0]
    for temp in temp_sequence:
        integrator.disable()
        integrator = hoomd.md.integrate.langevin(group=ALL, kT=temp, seed=int(rseed))
        hoomd.run(2e6)

hoomd.run(1e7)
hoomd.dump.gsd(BMC.filename + "mid2.gsd", group=hoomd.group.all(), overwrite=True, period=None)
if anneal:
    temp_sequence = [1.05, 1.1, 1.15, 1.1, 1.05, 1.0]
    for temp in temp_sequence:
        integrator.disable()
        integrator = hoomd.md.integrate.langevin(group=ALL, kT=temp, seed=int(rseed))
        hoomd.run(2e6)
hoomd.run(1e7)

hoomd.dump.gsd(BMC.filename + "mid3.gsd", group=ALL, overwrite=True, period=None)
hoomd.dump.gsd(BMC.filename + "final-frame.gsd", group=ALL, overwrite=True, period=None)
