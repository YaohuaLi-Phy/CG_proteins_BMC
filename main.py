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
#rseed=42
#mer_mer = 4.0
#mer_scaffold = 2.0
anneal = False
note='q8'
rseed=os.environ['RSEED']
mer_mer=os.environ['MERMER']
mer_scaffold=os.environ['MER_TEMP']
scaffold_scaffold=os.environ['TEMP_TEMP']
BMC = type('BMC', (object,), {})()
edge_l = 2.5
hexamer1 = PduABody(edge_length=edge_l)
hexamer2 = PduBBody(edge_length=edge_l)
pentamer = PentagonBody(edge_length=edge_l)
a = 2.5
template = SphericalTemplate(a)
n_hex1=int(os.environ['N_HEX1'])
n_scaf=int(os.environ['N_SCAF'])
BMC.filename=str(note)+'n_hex1_' + str(n_hex1)+'_Scaf_'+str(n_scaf)+'_'+str(rseed)

sys = Lattice(pentamer, hexamer1, hexamer2,template, num_hex1=n_hex1, num_scaffold=n_scaf)

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
system = hoomd.init.create_lattice(unitcell=uc, n=[4, 5, 5])

# Add constituent particles of type A and create the rods

added_types = ['A', 'B', 'C', 'D', 'C1', 'D1', 'qP', 'qN', 'Sc', 'Ss']
system_types = ['R', 'P', 'H', 'R2'] + added_types
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
lB = 1.0
kp = 1.1
z_q = 8.0
A_yuka = z_q**2 * lB * (np.exp(kp*a)/(1+kp*a))**2

nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(system_types, system_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1, epsilon=1.0))
table.pair_coeff.set('C', 'D', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_mer)))
table.pair_coeff.set('C', 'D1', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_mer)))
table.pair_coeff.set('C1', 'D', func=LJ_attract, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_mer)))
table.pair_coeff.set(['qP', 'C'], ['qP', 'C'], func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
table.pair_coeff.set(['qP', 'C'], 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=-5, kappa=kp))
table.pair_coeff.set('qN', 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
table.pair_coeff.set('H', 'H', func=Yukawa, rmin=0.01, rmax=4.5, coeff=dict(A=A_yuka, kappa=kp))
table.pair_coeff.set('Sc', 'Sc', func=normal_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(scaffold_scaffold)))
table.pair_coeff.set('Sc', 'Ss', func=normal_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_scaffold)))

hoomd.md.integrate.mode_standard(dt=0.004)
rigid = hoomd.group.rigid_center()
integrator = hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=int(rseed))

hoomd.analyze.log(filename=BMC.filename + ".log",
                  quantities=['temperature','potential_energy',
                              'translational_kinetic_energy',
                              'rotational_kinetic_energy'],
                  period=10000,
                  overwrite=True)

hoomd.dump.gsd(BMC.filename+".gsd",
               period=2e4,
               group=hoomd.group.all(),
               overwrite=True)

hoomd.run(5e6)

if anneal:
    temp_sequence=[1.25, 1.5, 1.25, 1.0, 1.25, 1.5, 1.25, 1.0]
    for temp in temp_sequence:
        integrator.disable()
        integrator = hoomd.md.integrate.langevin(group=rigid, kT=temp, seed=int(rseed))
        hoomd.run(2e6)

hoomd.run(2e7)
