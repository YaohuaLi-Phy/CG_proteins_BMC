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
rseed=27154
mer_mer = 4.0
mer_scaffold = 2.0
anneal = False
note='q10'
#rseed=os.environ['RSEED']
#mer_mer=os.environ['MERMER']
#mer_scaffold=os.environ['MER_TEMP']
scaffold_scaffold=3.0  #os.environ['TEMP_TEMP']
BMC = type('BMC', (object,), {})()
edge_l = 2.5
#hexamer1 = PduABody(edge_length=edge_l)
#hexamer2 = PduBBody(edge_length=edge_l)
#pentamer = PentagonBody(edge_length=edge_l)
a = 2.5
template = SphericalTemplate(a)
n_hex1=2 #int(os.environ['N_HEX1'])
n_scaf=2 #int(os.environ['N_SCAF'])
BMC.filename=str(note)+'n_hex1_' + str(n_hex1)+'_Scaf_'+str(n_scaf)+'_'+str(rseed) + '.gsd'
system = hoomd.init.read_gsd('small_q10n_hex1_2_Scaf_2_28595.gsd')

# forcefield and integration
lB = 1.0
kp = 1.1
z_q = 10.0
A_yuka = z_q**2 * lB * (np.exp(kp*a)/(1+kp*a))**2

added_types = ['A', 'B', 'C', 'D', 'C1', 'D1', 'qP', 'qN', 'Sc', 'Ss']
system_types = ['R', 'P', 'H', 'R2'] + added_types
rigid = hoomd.md.constrain.rigid()
#rigid.create_bodies()
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
to_integrate = hoomd.group.rigid_center()
integrator = hoomd.md.integrate.langevin(group=to_integrate, kT=1.0, seed=int(rseed))


hoomd.dump.gsd("new.gsd",
               period=100,
               group=hoomd.group.all())

hoomd.run(10000)
