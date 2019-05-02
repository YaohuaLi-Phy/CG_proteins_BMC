from NonBonded import LJ_attract
from NonBonded import *
from Solution import Lattice
from SphericalTemplate import SphericalTemplate
import os
import time
from Pentagon import PentagonBody
from PduABody import PduABody
from PduBBody import PduBBody
import sys
# installation dependent stuff
sys.path.append('/projects/b1030/hoomd/hoomd-2.5.1/')
sys.path.append('/projects/b1021/Yaohua/cg_protein/new_rigid/')
import hoomd
import hoomd.md
from mpi4py import MPI

mpi_rank = MPI.COMM_WORLD.Get_rank()

BMC = type('BMC', (object,), {})()
BMC.replica_exchange_time = 5e4
a=1.02
BMC.replicas = [1.0, a, a**2, a**3, a**4, a**5, a**6, a**7]
time.sleep(3 * mpi_rank)
print('context for rank: ' + str(mpi_rank))
hoomd.context.initialize("--mode=gpu --nrank=1")
BMC.W_time = 3e7
BMC.replica_stages = int(BMC.W_time / BMC.replica_exchange_time)
note = 'temper_all'
rseed = os.environ['RSEED']
try:
    mer_mer = os.environ['MERMER']
except:
    mer_mer = 4.0
mer_scaffold = os.environ['MER_TEMP']
scaffold_scaffold = os.environ['TEMP_TEMP']

edge_l = 2.5
BMC.angle = 15 * np.pi/180
hexamer1 = PduABody(edge_length=edge_l, angle=BMC.angle)
hexamer2 = PduBBody(edge_length=edge_l, angle=BMC.angle)
pentamer = PentagonBody(edge_length=edge_l, angle=BMC.angle)
a = 2.5
template = SphericalTemplate(a)
n_hex1 = int(os.environ['N_HEX1'])
n_scaf = int(os.environ['N_SCAF'])
BMC.filename = str(note) + '_ee_' + str(scaffold_scaffold) + '_eh_' + str(mer_scaffold) + '_hh_' + str(
    mer_mer) + '_' + str(rseed)

sys = Lattice(pentamer, hexamer1, hexamer2, template, num_hex1=n_hex1, num_scaffold=n_scaf)

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

# Add constituent particles of type A and create the rods

added_types = ['A', 'B', 'C', 'D', 'qP', 'qN', 'C1', 'D1', 'Ss'] # types to be added to the system
system_types = ['R', 'P', 'R2'] + added_types  # types to define forcefield
if n_scaf > 0:
    system_types += ['H', 'Sc']
    added_types += ['Sc']
for new_type in added_types:
    system.particles.types.add(new_type)

rigid = hoomd.md.constrain.rigid()

if n_scaf > 0:
    rigid.set_param('H', types=template.type_list, positions=template.position)
rigid.set_param('R', types=hexamer1.type_list, positions=hexamer1.all_sites)
rigid.set_param('P', types=pentamer.type_list, positions=pentamer.all_sites)

rigid.set_param('R2', types=hexamer2.type_list, positions=hexamer2.all_sites)

rigid.create_bodies()

# forcefield and integration
lB = 1.0
kp = 1.1
z_q = 8.0
A_yuka = z_q ** 2 * lB * (np.exp(kp * a) / (1 + kp * a)) ** 2

nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(system_types, system_types, func=SoftRepulsive, rmin=0.01, rmax=3,
                     coeff=dict(sigma=1, epsilon=1.0))
#table.pair_coeff.set('Sc', system_types, func=SoftRepulsive, rmin=0.01, rmax=3, coeff=dict(sigma=1.5, epsilon=0.5))
table.pair_coeff.set('C', 'D', func=normal_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.15, epsilon=float(mer_mer)))
table.pair_coeff.set('C', 'D1', func=normal_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.15, epsilon=4))
table.pair_coeff.set('C1', 'D', func=normal_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.15, epsilon=4))
table.pair_coeff.set(['qP', 'C'], ['qP', 'C'], func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
table.pair_coeff.set(['qP', 'C'], 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=-5, kappa=kp))
table.pair_coeff.set('qN', 'qN', func=Yukawa, rmin=0.01, rmax=3, coeff=dict(A=5, kappa=kp))
if n_scaf>0:
    table.pair_coeff.set('Sc', 'Sc', func=yukawa_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(scaffold_scaffold), A=1.20, kappa=kp))
    table.pair_coeff.set('Sc', 'Ss', func=normal_lj, rmin=0.01, rmax=3, coeff=dict(sigma=1.0, epsilon=float(mer_scaffold)))

hoomd.md.integrate.mode_standard(dt=0.0004)
rigid = hoomd.group.rigid_center()
integrator = hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=int(rseed))

logger = hoomd.analyze.log(filename=BMC.filename + '_rank_' + str(BMC.replicas[mpi_rank]) + ".log",
                           quantities=['temperature', 'potential_energy',
                                       'translational_kinetic_energy',
                                       'rotational_kinetic_energy'],
                           period=10000,
                           overwrite=True)
hoomd.run(5e5)  # equilibriate
integrator.disable()
num_replicas = len(BMC.replicas)
exchanges_prob = np.zeros(num_replicas - 1)
BMC.kT = BMC.replicas[mpi_rank]
track_fn = 'track_replica' + str(rseed) +'.txt'
for stage in range(BMC.replica_stages):
    integrator = hoomd.md.integrate.langevin(group=rigid, kT=BMC.kT, seed=int(rseed))
    dumper = hoomd.dump.gsd(BMC.filename + '_rank_' + str(BMC.replicas[mpi_rank]) + ".gsd",
                            period=50000, group=hoomd.group.all(), overwrite=False)
    hoomd.run(int(BMC.replica_exchange_time))
    BMC.potential = logger.query('potential_energy')
    dumper.disable()
    integrator.disable()
    MPI.COMM_WORLD.Barrier()
    BMC.potential_arr = MPI.COMM_WORLD.gather(BMC.potential)

    # decide whether to do the replica exchange
    if mpi_rank == 0:
        sorted_replicas_idx = np.argsort(BMC.replicas)
        with open(track_fn, 'a') as track_fp:
            for idx in list(sorted_replicas_idx):
                track_fp.write("%s\t" % idx)
            track_fp.write('\n')

        for i in range(num_replicas - 1):
            idx_i = sorted_replicas_idx[i]
            idx_j = sorted_replicas_idx[i + 1]
            beta_i = 1 / ( BMC.replicas[idx_i])
            beta_j = 1 / ( BMC.replicas[idx_j])
            de = (beta_j - beta_i) * (BMC.potential_arr[idx_i] - BMC.potential_arr[idx_j])
            exchanges_prob[i] = min([1.0, np.exp(-de)])

        idx = np.random.randint(0, num_replicas - 1)
        rnum = np.random.uniform(0.0, 1.0)
        swapA = sorted_replicas_idx[idx]
        swapB = sorted_replicas_idx[idx + 1]
        print 'probability of swapping ' + str(BMC.replicas[swapA]) + ' with ' + str(BMC.replicas[swapB]) \
              + ' is ' + str(exchanges_prob[idx])
        if rnum < exchanges_prob[idx]:
            print 'swapping ' + str(BMC.replicas[swapA]) + ' with ' + str(BMC.replicas[swapB])
            BMC.replicas[swapA], BMC.replicas[swapB] = BMC.replicas[swapB], BMC.replicas[swapA]

    BMC.replicas = MPI.COMM_WORLD.bcast(BMC.replicas, root=0)
    MPI.COMM_WORLD.Barrier()

if mpi_rank == 0:
    hoomd.dump.gsd(BMC.filename + "final-frame.gsd", group=hoomd.group.all(), overwrite=True, period=None)
