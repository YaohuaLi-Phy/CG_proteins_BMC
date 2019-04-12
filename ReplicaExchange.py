# coding: utf-8
from numpy import *

del angle
from math import *
import CenterFile
import GenShape
import Build
import LinearChain
import Colloid
import copy
import time

from Units import SimulationUnits as SimUnits
###########################
# installation dependant stuff
#############################
import sys
import gc  # force garbage collection due to weird file handle left open in loops for dumpers

sys.path.append('/projects/b1030/hoomd/hoomd-2.1.5-DNA-bias/')
sys.path.append('/projects/b1030/hoomd/hoomd-2.1.5-DNA-bias/hoomd/')
from hoomd import *
from hoomd import md
from hoomd import deprecated as hdepr
from mpi4py import MPI

mpi_rank = MPI.COMM_WORLD.Get_rank()
# mpi_rank = 0
seednum = int(os.environ['RSEED'])
W_idx_start = int(os.environ['WIDX'])

# this simulation parameters
PMF = type('PMF', (object,), {})()
PMF.max_center_dist = 400.0
PMF.long_center_dist = 360.0
PMF.min_center_dist = 300.0
PMF.K_bias = 0.25
PMF.N_windows = int(os.environ['NWINDOWS'])
PMF.L_N_windows = PMF.N_windows / int(os.environ['NJOBS'])
PMF.L_W_idx_start = PMF.L_N_windows * W_idx_start
PMF.W_time = 5e6
PMF.equil_window_time = 3e4

# replica exchange contexts
PMF.replicas = [0.0, 0.50, 1.00, 1.50]
# PMF.simulation_contexts = [context.SimulationContext() for c in PMF.replicas]
# for context_idx in range(PMF.simulation_contexts.__len__()):
#    PMF.simulation_contexts[context_idx].set_current()
time.sleep(3 * mpi_rank)
print 'context for rank : ' + str(mpi_rank)
context.initialize("--mode=gpu --nrank=1")
# PMF.simulation_contexts[mpi_rank].set_current()

PMF.replica_exchange_time = 5e4
PMF.replica_steps = int(PMF.W_time / PMF.replica_exchange_time)
PMF.potentials = [0.0 for c in PMF.replicas]
PMF.bias_potentials = [0.0 for c in PMF.replicas]
Bias_alpha = 0.33
Bias_rcut = 5.0 / Bias_alpha + 6.0
Bias_rel_strength = 0.125
Bias_drcut = 400
# option.set_notice_level(5)

try:
    PMF.Salt = float(os.environ['SALT'].strip())
except:
    PMF.Salt = 0.3
    print 'Did not find SALT env variable, using 0.3M of salt'
offset = 0 if 'W_OFFSET' not in os.environ else int(os.environ['W_OFFSET'])
PMF.window_position = linspace(PMF.long_center_dist, PMF.min_center_dist, PMF.N_windows)
PMF.init_center_dist = PMF.window_position[PMF.L_W_idx_start]

print 'initial distance between particles : ' + str(PMF.init_center_dist)

try:
    PMF.temperature = float(os.environ['TEMPERATURE'].strip()) + 273.0
except:
    PMF.temperature = 273.0 + 25.0
    print 'Did not find TEMPERATURE env variable, using 25C'

PMF.kt = PMF.temperature * 0.0083144621  # conversion to temperature in kJ/mol

PMF.filename = '3SPN_PMFC_HREX_T_' + str(PMF.temperature) + '_Salt_' + str(PMF.Salt) + '_WDIDX_' + str(
    W_idx_start) + '_RSEED_' + str(seednum) + '_'


def FSCoulomb(r, rmin, rmax, alpha, qiqj):
    V = qiqj * (erfc(alpha * r) / r - erfc(alpha * rmax) / rmax + (r - rmax) * erfc(
        alpha * rmax) / rmax ** 2.0 + 2.0 * alpha / pi ** 0.5 * (r - rmax) * exp(-alpha ** 2.0 * rmax ** 2.0) / rmax)
    F = qiqj * (erfc(alpha * r) / r ** 2.0 + 2.0 * alpha / pi ** 0.5 * exp(-alpha ** 2.0 * r ** 2.0) / r - erfc(
        alpha * rmax) / rmax ** 2.0 - 2.0 * alpha / pi ** 0.5 * exp(-alpha ** 2.0 * rmax ** 2.0) / rmax)
    return (V, F)


##########################
# call molecular builder
###########################

if mpi_rank == 0:

    SUnits = SimUnits()
    SUnits.set_mass('amu')
    SUnits.set_length('A')
    SUnits.set_energy('kJ/mol')
    ns = 402
    shapes = [GenShape.Sphere(Num=ns)]

    Seq1A = LinearChain.DNA3SPN_Chain(ss_Seq=list('AAAAACATCCATCCTTATCAACT'), ds_Seq=list(''), sticky_Seq=list(''))
    Seq2A = LinearChain.DNA3SPN_Chain(ss_Seq=list('AAAAAAACGACTCATACTCACCT'), ds_Seq=list(''), sticky_Seq=list(''))

    shapes[-1].set_properties(
        {'size': 3.75, 'ColloidType': Colloid.SimpleColloid, 'surf_type': 'P', 'mass': 31593.0, 'density': 14.29})
    Seq1 = LinearChain.DNA3SPN_Chain(ss_Seq=list('AAAAA'), ds_Seq=list('CATCCATCCTTATCAACT'),
                                     sticky_Seq=list('TAAGGAAA'))
    shapes[-1].set_ext_grafts(Seq1, num=139, linker_bond_type='NP-DNA')
    shapes[-1].set_ext_grafts(Seq1A, num=261, linker_bond_type='NP-DNA')
    shapes.append(GenShape.Sphere(Num=ns))
    shapes[-1].set_properties(
        {'size': 3.75, 'ColloidType': Colloid.SimpleColloid, 'surf_type': 'P', 'mass': 31593.0, 'density': 14.29})
    Seq2 = LinearChain.DNA3SPN_Chain(ss_Seq=list('AAAAA'), ds_Seq=list('AACGACTCATACTCACCT'),
                                     sticky_Seq=list('GTTTCCTT'))
    shapes[-1].set_ext_grafts(Seq2, num=139, linker_bond_type='NP-DNA')
    shapes[-1].set_ext_grafts(Seq2A, num=261, linker_bond_type='NP-DNA')

    grid = CenterFile.RandomPositions(system_size=[600.0, 600.0, 1200.0], particle_numbers=[0], units=SUnits)
    # grid.add_one_particle(position=[0.0, 0.0, PMF.max_center_dist / 2.0])
    # grid.add_one_particle(position=[0.0, 0.0, -PMF.max_center_dist / 2.0])
    grid.add_one_particle(position=[0.0, 0.0, PMF.init_center_dist / 2.0])
    grid.add_one_particle(position=[0.0, 0.0, -PMF.init_center_dist / 2.0])
    grid.expend_table()

    buildobj = Build.BuildHoomdXML(center_obj=grid, shapes=shapes, units=SUnits)
    # buildobj.impose_box = [450.0, 450.0, 950.0]
    buildobj.set_rotation_function(mode='random')
    buildobj.add_bonds_colloids()
    buildobj.set_charge_by_type('Phos', -1.0)
    buildobj.add_rho_molar_ions(PMF.Salt, qtype='Cl', ion_mass=35.453, q=-1.0, ion_diam=0.0)
    buildobj.add_rho_molar_ions(PMF.Salt, qtype='Na', ion_mass=22.98977, q=1.0, ion_diam=0.0)
    buildobj.fix_remaining_charge(ptype='Na', ntype='Cl', pion_mass=22.98977, nion_mass=35.453, qp=1.0, qn=-1.0,
                                  pion_diam=0.0, nion_diam=0.0)
    buildobj.set_eps_to_salt_DePablo(val=PMF.Salt, temp=PMF.temperature)
else:
    buildobj = None
buildobj = MPI.COMM_WORLD.bcast(buildobj, root=0)
MPI.COMM_WORLD.Barrier()

sysbox = data.boxdim(Lx=600.0, Ly=600.0,
                     Lz=1200.0)  # transverse size is ~particle diameter + 2*DNA length (~10 nm) + 20 nm = 45-50 nm, vertical is max(transverse, 2*max_dist)+15 = 95 nm

sn = data.make_snapshot(N=buildobj.num_beads, box=sysbox, particle_types=buildobj.bead_types)
buildobj.set_snapshot(sn)
system = init.read_snapshot(sn)

rigid_tuple_list = buildobj.aggregate_rigid_tuples()
rigid = md.constrain.rigid()
for cons in rigid_tuple_list:
    rigid.set_param(type_name=cons[0], types=cons[1], positions=cons[2])

rigid.create_bodies(create=False)

# for context_idx in range(1, PMF.replicas.__len__()):
#     PMF.simulation_contexts[context_idx].set_current()
#
#     system = init.read_snapshot(sn)
#
#     rigid = md.constrain.rigid()
#     for cons in rigid_tuple_list:
#         rigid.set_param(type_name=cons[0], types=cons[1], positions=cons[2])
#     rigid.create_bodies(create=False)

# BiasPotential = [None for c in PMF.replicas]
# Loggers = [None for c in PMF.replicas]
# AnBonded = [None for c in PMF.replicas]

# for context_idx in range(PMF.replicas.__len__()):
# PMF.simulation_contexts[context_idx].set_current()
syslist = md.nlist.cell(r_buff=0.6)
syslist.reset_exclusions(exclusions=['1-2', 'body'])
dna_nlist = md.nlist.tree(r_buff=0.6)
dna_nlist.reset_exclusions(exclusions=['1-2', 'body'])
charge_nlist = md.nlist.tree()
charge_nlist.reset_exclusions(exclusions=['1-2', 'body'])

####################
# Force field setup#
#####################
context_idx = mpi_rank
k2v = 0.6
k4v = 60.0
AnBonded = md.bond.anharmonic()
AnBonded.bond_coeff.set('Phos-Sug', K2=k2v, K3=0.0, K4=k4v, r0=3.899)
AnBonded.bond_coeff.set('Sug-Phos', K2=k2v, K3=0.0, K4=k4v, r0=3.559)
AnBonded.bond_coeff.set('NP-DNA', K2=k2v, K3=0.0, K4=k4v, r0=5.0)
AnBonded.bond_coeff.set('Sug-Ade', K2=k2v, K3=0.0, K4=k4v, r0=4.670)
AnBonded.bond_coeff.set('Sug-Thy', K2=k2v, K3=0.0, K4=k4v, r0=4.189)
AnBonded.bond_coeff.set('Sug-Gua', K2=k2v, K3=0.0, K4=k4v, r0=4.829)
AnBonded.bond_coeff.set('Sug-Cyt', K2=k2v, K3=0.0, K4=k4v, r0=4.112)
AnBonded.bond_coeff.set('dummy', K2=0.0, K3=0.0, K4=0.0, r0=0.0)
AnBonded.bond_coeff.set('Colloid-bond-0-1', K2=PMF.K_bias, K3=0.0, K4=0.0, r0=PMF.init_center_dist)

harm_angle = md.angle.harmonic()
harm_angle.angle_coeff.set('Sug-Phos-Sug', k=200.0, t0=94.49 * pi / 180.0)
harm_angle.angle_coeff.set('Phos-Sug-Phos', k=200.0, t0=120.15 * pi / 180.0)
harm_angle.angle_coeff.set('Phos-Sug-Ade', k=200.0, t0=103.53 * pi / 180.0)
harm_angle.angle_coeff.set('Phos-Sug-Thy', k=200.0, t0=92.06 * pi / 180.0)
harm_angle.angle_coeff.set('Phos-Sug-Gua', k=200.0, t0=107.40 * pi / 180.0)
harm_angle.angle_coeff.set('Phos-Sug-Cyt', k=200.0, t0=96.96 * pi / 180.0)
harm_angle.angle_coeff.set('Ade-Sug-Phos', k=200.0, t0=112.07 * pi / 180.0)
harm_angle.angle_coeff.set('Thy-Sug-Phos', k=200.0, t0=116.68 * pi / 180.0)
harm_angle.angle_coeff.set('Gua-Sug-Phos', k=200.0, t0=110.12 * pi / 180.0)
harm_angle.angle_coeff.set('Cyt-Sug-Phos', k=200.0, t0=114.34 * pi / 180.0)

bs_angle = md.angle.basestacking(K=6.0)
bs_angle.angle_coeff.set('Sug-Phos-Sug', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)
bs_angle.angle_coeff.set('Phos-Sug-Phos', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)
bs_angle.angle_coeff.set('Phos-Sug-Ade', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)
bs_angle.angle_coeff.set('Phos-Sug-Thy', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)
bs_angle.angle_coeff.set('Phos-Sug-Gua', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)
bs_angle.angle_coeff.set('Phos-Sug-Cyt', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)
bs_angle.angle_coeff.set('Ade-Sug-Phos', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)
bs_angle.angle_coeff.set('Thy-Sug-Phos', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)
bs_angle.angle_coeff.set('Gua-Sug-Phos', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)
bs_angle.angle_coeff.set('Cyt-Sug-Phos', epsilon=0.0, alpha=0.0, r0=0.0, t0=0.0)

bp = ['Ade', 'Thy', 'Gua', 'Cyt']

# base stacking params
e_tab = array([[14.39, 14.34, 13.25, 14.51], [10.37, 13.36, 10.34, 12.89], [14.81, 15.57, 14.93, 15.39],
               [11.42, 12.79, 10.52, 13.24]])
r0_tab = array([[3.716, 3.675, 3.827, 3.744], [4.238, 3.984, 4.416, 4.141], [3.576, 3.598, 3.664, 3.635],
                [4.038, 3.798, 4.208, 3.935]])
t0_tab = array([[101.15, 85.94, 105.26, 89.00], [101.59, 89.50, 104.31, 91.28], [100.89, 84.83, 105.48, 88.28],
                [106.49, 93.31, 109.54, 95.46]])
t0_tab *= pi / 180.0

# cross stacking params
e_tab_m = array([[2.186, 2.774, 2.833, 1.951], [2.774, 2.186, 2.539, 2.980], [2.833, 2.539, 3.774, 1.129],
                 [1.951, 2.980, 1.129, 4.802]])
e_tab_p = transpose(array([[2.186, 2.774, 2.980, 2.539], [2.774, 2.186, 1.951, 2.833], [2.980, 1.951, 4.802, 1.129],
                           [2.539, 2.833, 1.129, 3.774]]))

r0_tab_m = array([[6.208, 6.876, 6.072, 6.941], [6.876, 7.480, 6.771, 7.640], [6.072, 6.771, 5.921, 6.792],
                  [6.941, 7.640, 6.792, 7.698]])
r0_tab_p = transpose(array([[5.435, 6.295, 5.183, 5.965], [6.295, 7.195, 6.028, 6.868], [5.183, 6.028, 4.934, 5.684],
                            [5.965, 6.868, 5.684, 6.453]]))

t0_tab_m = array([[154.38, 159.10, 152.46, 157.58], [147.10, 153.79, 144.44, 148.59], [154.69, 157.83, 153.43, 158.60],
                  [160.37, 164.45, 158.62, 162.73]])
t0_tab_p = transpose(array(
    [[116.88, 121.74, 114.23, 114.58], [109.42, 112.95, 107.32, 106.41], [119.34, 124.72, 116.51, 117.49],
     [122.10, 125.80, 120.00, 119.67]]))
t0_tab_m *= pi / 180.0
t0_tab_p *= pi / 180.0

for i in range(4):
    for j in range(4):
        harm_angle.angle_coeff.set('Sug-' + bp[i] + '-' + bp[j], k=0.01, t0=0.01)
        bs_angle.angle_coeff.set('Sug-' + bp[i] + '-' + bp[j], epsilon=e_tab[i][j], alpha=3.0, r0=r0_tab[i][j],
                                 t0=t0_tab[i][j])


def gaussian(theta, k, phi0, sigma):
    lphi = [theta - phi0, theta - phi0 + 2.0 * pi, theta - phi0 - 2.0 * pi]
    lphi_abs = map(abs, lphi)
    dphi = min(lphi_abs)
    signdphi = sign(lphi[lphi_abs.index(dphi)])
    V = -k * exp(-(dphi) ** 2.0 / (2.0 * sigma ** 2.0))
    F = -k * (dphi) * signdphi * exp(-(dphi) ** 2.0 / (2.0 * sigma ** 2.0)) / sigma ** 2.0
    return (V, F)


dna_dihedral = md.dihedral.table(width=1024)
dna_dihedral.dihedral_coeff.set('Phos-Sug-Phos-Sug', func=gaussian,
                                coeff=dict(k=6.0, phi0=-154.79 * pi / 180.0, sigma=0.30))
dna_dihedral.dihedral_coeff.set('Sug-Phos-Sug-Phos', func=gaussian,
                                coeff=dict(k=6.0, phi0=-179.17 * pi / 180.0, sigma=0.30))

Kbp = 12.0
bp_rcut = 16.0

excl_vol = md.pair.lj(r_cut=18.0, nlist=syslist)
excl_vol.set_params(mode="shift")
dna_pairing = md.pair.DNABP(r_cut=bp_rcut, nlist=dna_nlist)
dna_xsM = md.pair.DNAXSM(r_cut=bp_rcut, nlist=dna_nlist)
dna_xsP = md.pair.DNAXSP(r_cut=bp_rcut, nlist=dna_nlist)
BiasPotential = md.pair.morseBias(r_cut=Bias_rcut, nlist=dna_nlist)

rel_entropy_potentials = md.pair.table(width=551, name='relative_entropy', nlist=charge_nlist)
rel_entropy_potentials.set_from_file('Na', 'Na', '../Na-Na.dat')
rel_entropy_potentials.set_from_file('Na', 'Cl', '../Na-Cl.dat')
rel_entropy_potentials.set_from_file('Cl', 'Cl', '../Cl-Cl.dat')
rel_entropy_potentials.set_from_file('Na', 'Phos', '../P-Na.dat')
rel_entropy_potentials.set_from_file('Cl', 'Phos', '../P-Cl.dat')
rel_entropy_potentials.set_from_file('Phos', 'Phos', '../P-P.dat')
non_charged = ['W', 'P', 'Ade', 'Thy', 'Gua', 'Cyt', 'Sug']
charged = ['Na', 'Cl', 'Phos']

rel_entropy_potentials.pair_coeff.set(non_charged, non_charged, func=FSCoulomb, rmin=0.001, rmax=0.002,
                                      coeff=dict(qiqj=0.0, alpha=1.0))
rel_entropy_potentials.pair_coeff.set(charged, non_charged, func=FSCoulomb, rmin=0.001, rmax=0.002,
                                      coeff=dict(qiqj=0.0, alpha=1.0))
# ch_group = group.charged()
# charge_potential = md.charge.pppm(group = ch_group, nlist = charge_nlist)
# charge_potential.set_params(Nx = 64, Ny = 64, Nz = 128, order=6, rcut=8.0)

# for idx1 in range(3):
#    for idx2 in range(idx1, 3):
#        charge_potential.ewald.pair_coeff.set(charged[idx1], charged[idx2], kappa = charge_potential.ew_kap,
#                                              alpha = charge_potential.ew_alp, r_cut = charge_potential.ew_rcut)
# charge_potential.ewald.pair_coeff.set(non_charged, non_charged, kappa = 0.1, alpha = 0.1, r_cut = 0.0)
# charge_potential.ewald.pair_coeff.set(non_charged, charged, kappa = 0.1, alpha = 0.1, r_cut = 0.0)

charge_potential = md.pair.table(width=1024, name='coulombic', nlist=charge_nlist)
for idx1 in range(3):
    for idx2 in range(idx1, 3):
        charge_potential.pair_coeff.set(charged[idx1], charged[idx2], func=FSCoulomb, rmin=0.3, rmax=30.0,
                                        coeff=dict(qiqj=buildobj.get_type_charge_product(charged[idx1], charged[idx2]),
                                                   alpha=0.01))
charge_potential.pair_coeff.set(non_charged, non_charged, func=FSCoulomb, rmin=0.001, rmax=0.002,
                                coeff=dict(qiqj=0.0, alpha=1.0))
charge_potential.pair_coeff.set(non_charged, charged, func=FSCoulomb, rmin=0.001, rmax=0.002,
                                coeff=dict(qiqj=0.0, alpha=1.0))

excl_loc_types = ['W', 'P', 'Na', 'Cl']
excl_loc_dia = {'W': 25.0, 'P': 1.0, 'Na': 2.494, 'Cl': 4.478}
dna_bp_no = dict(epsilon=-1.0, r0=0.0, K=0.0, alpha=0.0, phi0=0.0, theta10p=0.0, theta10m=0.0, theta30=0.0,
                 epsilon_3p=0.0, epsilon_3m=0.0, r0_3p=0.0, r0_3m=0.0, theta_cs3p_0=0.0, theta_cs3p_1=0.0,
                 theta_cs3m_0=0.0, theta_cs3m_1=0.0, r_cut=0.0)
bias_no = dict(D0=0.0, alpha=0.0, r0=0.0, dcut=0.0, r_cut=0.0)

for tuple in LinearChain.DNA3SPN_Chain.get_excl_sig():
    excl_vol.pair_coeff.set(tuple[0], tuple[1], epsilon=1.0, sigma=tuple[2], r_cut=tuple[2] * 2.0 ** (1.0 / 6.0))
    if not (tuple[0] in bp and tuple[1] in bp):
        dna_pairing.pair_coeff.set(tuple[0], tuple[1], **dna_bp_no)
        dna_xsM.pair_coeff.set(tuple[0], tuple[1], **dna_bp_no)
        dna_xsP.pair_coeff.set(tuple[0], tuple[1], **dna_bp_no)
        BiasPotential.pair_coeff.set(tuple[0], tuple[1], **bias_no)
        pass
dna_pairing.pair_coeff.set('Ade', 'Thy', epsilon=16.73, r0=5.941, K=Kbp, alpha=2.0, phi0=-38.35 * pi / 180.0,
                           theta10p=156.54 * pi / 180.0, theta10m=135.78 * pi / 180.0, theta30=116.09 * pi / 180.0,
                           epsilon_3p=e_tab_p[0][1], epsilon_3m=e_tab_m[0][1], r0_3p=r0_tab_p[0][1],
                           r0_3m=r0_tab_m[0][1], theta_cs3p_0=t0_tab_p[0][1], theta_cs3p_1=t0_tab_p[0][1],
                           theta_cs3m_0=t0_tab_m[0][1], theta_cs3m_1=t0_tab_m[0][1], r_cut=bp_rcut)
dna_pairing.pair_coeff.set('Gua', 'Cyt', epsilon=21.18, r0=5.530, K=Kbp, alpha=2.0, phi0=-42.98 * pi / 180.0,
                           theta10p=141.16 * pi / 180.0, theta10m=159.81 * pi / 180.0, theta30=124.94 * pi / 180.0,
                           epsilon_3p=e_tab_p[2][3], epsilon_3m=e_tab_m[2][3], r0_3p=r0_tab_p[2][3],
                           r0_3m=r0_tab_m[2][3], theta_cs3p_0=t0_tab_p[2][3], theta_cs3p_1=t0_tab_p[2][3],
                           theta_cs3m_0=t0_tab_m[2][3], theta_cs3m_1=t0_tab_m[2][3], r_cut=bp_rcut)

dna_xsM.pair_coeff.set('Ade', 'Thy', epsilon=16.73, r0=5.941, K=Kbp, alpha=2.0, phi0=-38.35 * pi / 180.0,
                       theta10p=156.54 * pi / 180.0, theta10m=135.78 * pi / 180.0, theta30=116.09 * pi / 180.0,
                       epsilon_3p=e_tab_p[0][1], epsilon_3m=e_tab_m[0][1], r0_3p=r0_tab_p[0][1], r0_3m=r0_tab_m[0][1],
                       theta_cs3p_0=t0_tab_p[0][1], theta_cs3p_1=t0_tab_p[0][1], theta_cs3m_0=t0_tab_m[0][1],
                       theta_cs3m_1=t0_tab_m[0][1], r_cut=bp_rcut)
dna_xsM.pair_coeff.set('Gua', 'Cyt', epsilon=21.18, r0=5.530, K=Kbp, alpha=2.0, phi0=-42.98 * pi / 180.0,
                       theta10p=141.16 * pi / 180.0, theta10m=159.81 * pi / 180.0, theta30=124.94 * pi / 180.0,
                       epsilon_3p=e_tab_p[2][3], epsilon_3m=e_tab_m[2][3], r0_3p=r0_tab_p[2][3], r0_3m=r0_tab_m[2][3],
                       theta_cs3p_0=t0_tab_p[2][3], theta_cs3p_1=t0_tab_p[2][3], theta_cs3m_0=t0_tab_m[2][3],
                       theta_cs3m_1=t0_tab_m[2][3], r_cut=bp_rcut)

dna_xsP.pair_coeff.set('Ade', 'Thy', epsilon=16.73, r0=5.941, K=Kbp, alpha=2.0, phi0=-38.35 * pi / 180.0,
                       theta10p=156.54 * pi / 180.0, theta10m=135.78 * pi / 180.0, theta30=116.09 * pi / 180.0,
                       epsilon_3p=e_tab_p[0][1], epsilon_3m=e_tab_m[0][1], r0_3p=r0_tab_p[0][1], r0_3m=r0_tab_m[0][1],
                       theta_cs3p_0=t0_tab_p[0][1], theta_cs3p_1=t0_tab_p[0][1], theta_cs3m_0=t0_tab_m[0][1],
                       theta_cs3m_1=t0_tab_m[0][1], r_cut=bp_rcut)
dna_xsP.pair_coeff.set('Gua', 'Cyt', epsilon=21.18, r0=5.530, K=Kbp, alpha=2.0, phi0=-42.98 * pi / 180.0,
                       theta10p=141.16 * pi / 180.0, theta10m=159.81 * pi / 180.0, theta30=124.94 * pi / 180.0,
                       epsilon_3p=e_tab_p[2][3], epsilon_3m=e_tab_m[2][3], r0_3p=r0_tab_p[2][3], r0_3m=r0_tab_m[2][3],
                       theta_cs3p_0=t0_tab_p[2][3], theta_cs3p_1=t0_tab_p[2][3], theta_cs3m_0=t0_tab_m[2][3],
                       theta_cs3m_1=t0_tab_m[2][3], r_cut=bp_rcut)

BiasPotential.pair_coeff.set('Ade', 'Thy', D0=0.0 * 16.73 * Bias_rel_strength, alpha=Bias_alpha, r0=5.941 / 2.0,
                             dcut=Bias_drcut)
BiasPotential.pair_coeff.set('Gua', 'Cyt', D0=0.0 * 21.18 * Bias_rel_strength, alpha=Bias_alpha, r0=5.941 / 2.0,
                             dcut=Bias_drcut)

for i in range(4):
    for j in range(i, 4):
        if ((i == 0 and j == 1) or (i == 2 and j == 3)):
            continue
        dna_pairing.pair_coeff.set(bp[i], bp[j], **dna_bp_no)
        dna_xsM.pair_coeff.set(bp[i], bp[j], epsilon=-1.0, r0=6.0, K=12.0, alpha=2.0, phi0=0.0, theta10p=0.0,
                               theta10m=0.0, theta30=0.0, epsilon_3p=e_tab_p[i][j], epsilon_3m=e_tab_m[i][j],
                               r0_3p=r0_tab_p[i][j], r0_3m=r0_tab_m[i][j], theta_cs3p_0=t0_tab_p[i][j],
                               theta_cs3p_1=t0_tab_p[j][i], theta_cs3m_0=t0_tab_m[i][j], theta_cs3m_1=t0_tab_m[j][i],
                               r_cut=0.0)
        dna_xsP.pair_coeff.set(bp[i], bp[j], epsilon=-1.0, r0=6.0, K=12.0, alpha=2.0, phi0=0.0, theta10p=0.0,
                               theta10m=0.0, theta30=0.0, epsilon_3p=e_tab_p[i][j], epsilon_3m=e_tab_m[i][j],
                               r0_3p=r0_tab_p[i][j], r0_3m=r0_tab_m[i][j], theta_cs3p_0=t0_tab_p[i][j],
                               theta_cs3p_1=t0_tab_p[j][i], theta_cs3m_0=t0_tab_m[i][j], theta_cs3m_1=t0_tab_m[j][i],
                               r_cut=0.0)
        BiasPotential.pair_coeff.set(bp[i], bp[j], **bias_no)

excl_vol.pair_coeff.set(['W', 'P'], ['W', 'P'], epsilon=0.0, sigma=0.0, r_cut=0.0)
excl_vol.pair_coeff.set(['Na', 'Cl'], ['Na', 'Cl'], epsilon=0.0, sigma=0.0, r_cut=0.0)
dna_pairing.pair_coeff.set(excl_loc_types, excl_loc_types, **dna_bp_no)
dna_xsM.pair_coeff.set(excl_loc_types, excl_loc_types, **dna_bp_no)
dna_xsP.pair_coeff.set(excl_loc_types, excl_loc_types, **dna_bp_no)
BiasPotential.pair_coeff.set(excl_loc_types, excl_loc_types, **bias_no)

for local_item in ['W', 'P']:
    for local_item2 in ['Na', 'Cl']:
        lsig = 0.5 * (excl_loc_dia[local_item] + excl_loc_dia[local_item2])
        excl_vol.pair_coeff.set(local_item, local_item2, epsilon=2.5, sigma=lsig, r_cut=lsig * 2.0 ** (1.0 / 6.0))

for dna_item in LinearChain.DNA3SPN_Chain._ALL_TYPES:
    for local_item in excl_loc_types:
        lsig = 0.5 * (excl_loc_dia[local_item] + LinearChain.DNA3SPN_Chain._EXCL_DIAM_LIST[dna_item])
        excl_vol.pair_coeff.set(dna_item, local_item, epsilon=1.0, sigma=lsig, r_cut=lsig * 2.0 ** (1.0 / 6.0))
        dna_pairing.pair_coeff.set(local_item, dna_item, **dna_bp_no)
        dna_xsM.pair_coeff.set(local_item, dna_item, **dna_bp_no)
        dna_xsP.pair_coeff.set(local_item, dna_item, **dna_bp_no)
        BiasPotential.pair_coeff.set(local_item, dna_item, **bias_no)


    ###########################################
    # Simulation setup
    ###########################################
    # center_groups = [None for c in PMF.replicas]
    # DCD_dumpers = [None for c in PMF.replicas]
    # DNA_thermos = [None for c in PMF.replicas]
    # DNA_loggers = [None for c in PMF.replicas]
    # for context_idx in range(PMF.simulation_contexts.__len__()):
    # PMF.simulation_contexts.set_current()
ALL = group.all()
BiasPotential.pair_coeff.set('Ade', 'Thy', D0=16.73 * Bias_rel_strength * PMF.replicas[mpi_rank])
BiasPotential.pair_coeff.set('Gua', 'Cyt', D0=21.18 * Bias_rel_strength * PMF.replicas[mpi_rank])

if PMF.replicas[mpi_rank] == 0:
    DCD_dumpers = dump.dcd(filename=PMF.filename + 'DCD.dcd', period=10000, overwrite=True)
else:
    DCD_dumpers = dump.dcd(filename=PMF.filename + '_bias_' + str(PMF.replicas[mpi_rank]) + '_DCD.dcd', period=10000,
                           overwrite=True)

if PMF.replicas[mpi_rank] == 0:
    xmld = hdepr.dump.xml(filename=PMF.filename + 'topo.xml', all=True, group=ALL)

md.integrate.mode_standard(dt=0.07, aniso=True)  # ~ 7fs

group_Na = group.type('Na')
group_Cl = group.type('Cl')
group_ions = group.union(a=group_Na, b=group_Cl, name='Ions')

grint = group.nonrigid()
gr_NI = group.difference(name='DNA', a=grint, b=group_ions)
DNA_thermos = compute.thermo(group=gr_NI)
DNA_loggers = analyze.log(filename=None, period=2000, quantities=['potential_energy_' + 'DNA'])
# ct = compute.thermo(group = gr_NI)
# restrict centers to move along Z
center_groups = group.rigid_center()
oneD_constrain = md.constrain.oneD(group=center_groups, constraint_vector=[0, 0, 1])
Loggers = analyze.log(filename=PMF.filename + '_idx_' + str(mpi_rank) + 'logALL.log', period=1000,
                      quantities=['temperature', 'temperature_DNA', 'potential_energy',
                                  'aniso_pair_dnabp_energy', 'aniso_pair_dnaxs5_energy',
                                  'aniso_pair_dnaxs3_energy', 'pair_lj_energy',
                                  'bond_anharmonic_energy', 'base_stacking_energy',
                                  'angle_harmonic_energy', 'pair_morseBias_energy',
                                  'dihedral_table_energy', 'pair_table_energy_coulombic',
                                  'pair_table_energy_relative_entropy', 'kinetic_energy',
                                  'translational_kinetic_energy', 'rotational_kinetic_energy'],
                      overwrite=True)
grALL = group.union(a=grint, b=center_groups, name='all_particles')

ion_nve = md.integrate.nve(group=group_ions, limit=0.005)
run(1e4)
ion_nve.disable()
ion_langevin_init = md.integrate.langevin(kT=PMF.kt, group=group_ions, seed=seednum)
run(2e6)

ion_langevin_init.disable()

# then equilibrate the whole thing
grintE = group.difference(name='BaseMov', a=gr_NI, b=group.type('Phos'))
intequil = md.integrate.nve(group=grintE, limit=0.05)
run(1e3)
# intequil.set_params(limit = 0.002)
# run(1e3)
intequil.disable()
intequil2 = md.integrate.nve(group=gr_NI, limit=0.05)
run(1e3)
# intequil2.set_params(limit = 0.002)
# run(1e3)
intequil2.disable()

# varT = variant.linear_interp(points=[(0, 0.1), (3e6, PMF.kt)])
integ = md.integrate.langevin(kT=PMF.kt, group=grint, seed=seednum,
                              noiseless_r=True)  # system temperature is (K_tran + K_rot and K_rot = 0)
integ2 = md.integrate.langevin(kT=PMF.kt, group=center_groups, seed=seednum)
for dna_item in LinearChain.DNA3SPN_Chain._ALL_TYPES:
    integ.set_gamma(dna_item, 0.5 * LinearChain.DNA3SPN_Chain._EXCL_DIAM_LIST[dna_item])
integ2.set_gamma('W', 0.5 * 75.0)
integ.set_gamma('Na', 0.5 * 2.494)
integ.set_gamma('Cl', 0.5 * 4.478)

run(2e6)
# Loggers.disable()

# create dump file
beta = 1.0 / PMF.kt
exchanges_prob = zeros(PMF.replicas.__len__() - 1)

for window_idx in range(PMF.L_N_windows - offset):

    if mpi_rank == 0:
        dumper = dump.gsd(filename=PMF.filename + 'window_' + str(
            PMF.window_position[PMF.L_W_idx_start + window_idx + offset]) + '.gsd',
                          group=center_groups, period=1000, overwrite=True)
        dumper.disable()
    MPI.COMM_WORLD.Barrier()

    AnBonded.bond_coeff.set('Colloid-bond-0-1', K2=PMF.K_bias, K3=0.0, K4=0.0,
                            r0=PMF.window_position[PMF.L_W_idx_start + window_idx + offset])
    BiasPotential.pair_coeff.set('Ade', 'Thy', D0=16.73 * Bias_rel_strength * PMF.replicas[mpi_rank])
    BiasPotential.pair_coeff.set('Gua', 'Cyt', D0=21.18 * Bias_rel_strength * PMF.replicas[mpi_rank])
    if PMF.replicas[mpi_rank] == 0:
        DCD_dumpers = dump.dcd(filename=PMF.filename + 'DCD.dcd', period=10000, overwrite=False)

    else:
        DCD_dumpers = dump.dcd(filename=PMF.filename + '_bias_' + str(PMF.replicas[mpi_rank]) + '_DCD.dcd',
                               period=10000, overwrite=False)

    run(PMF.equil_window_time)
    DCD_dumpers.disable()

    for step in range(PMF.replica_steps):
        # run the simulation for a bit
        gc.collect()  # collect garbage to free file handles, we don't want OS problems
        # for context_idx in range(PMF.replicas.__len__()):
        # PMF.simulation_contexts[context_idx].set_current()
        AnBonded.bond_coeff.set('Colloid-bond-0-1', K2=PMF.K_bias, K3=0.0, K4=0.0,
                                r0=PMF.window_position[PMF.L_W_idx_start + window_idx + offset])
        if not (PMF.replicas[mpi_rank] == 0):
            BiasPotential.enable()
            BiasPotential.pair_coeff.set('Ade', 'Thy', D0=16.73 * 2.0 * Bias_rel_strength * PMF.replicas[mpi_rank])
            BiasPotential.pair_coeff.set('Gua', 'Cyt', D0=21.18 * Bias_rel_strength * PMF.replicas[mpi_rank])
        else:
            BiasPotential.pair_coeff.set('Ade', 'Thy', D0=16.73 * 2.0 * Bias_rel_strength * 1.0)
            BiasPotential.pair_coeff.set('Gua', 'Cyt', D0=21.18 * Bias_rel_strength * 1.0)
            BiasPotential.disable(log=True)

        if PMF.replicas[mpi_rank] == 0:
            # dumper.enable()
            DCD_dumpers = dump.dcd(filename=PMF.filename + 'DCD.dcd', period=10000, overwrite=False)
            dumper = dump.gsd(filename=PMF.filename + 'window_' + str(
                PMF.window_position[PMF.L_W_idx_start + window_idx + offset]) + '.gsd',
                              group=center_groups, period=1000, overwrite=False)
        else:
            DCD_dumpers = dump.dcd(filename=PMF.filename + '_bias_' + str(PMF.replicas[mpi_rank]) + '_DCD.dcd',
                                   period=10000, overwrite=False)
            dumper = dump.gsd(filename=PMF.filename + '_bias_' + str(PMF.replicas[mpi_rank]) + '.gsd',
                              group=center_groups, period=1000, overwrite=False)

        run(PMF.replica_exchange_time)
        PMF.potentials = DNA_loggers.query('potential_energy_' + 'DNA')
        PMF.bias_potentials = Loggers.query('pair_morseBias_energy')
        dumper.disable()
        DCD_dumpers.disable()
        MPI.COMM_WORLD.Barrier()
        PMF.bias_potentials = MPI.COMM_WORLD.gather(PMF.bias_potentials, root=0)
        PMF.potentials = MPI.COMM_WORLD.gather(PMF.potentials, root=0)

        # decide whether we perform exchange of replicas. we only swap neighbours, since we have -1, 0, 1, we can only swap 0 with either
        if mpi_rank == 0:  # do the exchange on rank 0 then broadcast it
            zero_index = PMF.replicas.index(0.0)
            sorted_args_replicas = argsort(PMF.replicas)
            psum = 0.0

            for i in range(exchanges_prob.__len__()):
                idx_i = sorted_args_replicas[i]
                idx_j = sorted_args_replicas[i + 1]
                if idx_i == zero_index:
                    de = PMF.bias_potentials[idx_j] - PMF.bias_potentials[idx_i] * PMF.replicas[idx_j]

                elif idx_j == zero_index:
                    de = PMF.bias_potentials[idx_i] - PMF.bias_potentials[idx_j] * PMF.replicas[idx_i]
                else:
                    de = PMF.bias_potentials[idx_i] + PMF.bias_potentials[idx_j] \
                         - PMF.bias_potentials[idx_i] * PMF.replicas[idx_j] / PMF.replicas[idx_i] \
                         - PMF.bias_potentials[idx_j] * PMF.replicas[idx_i] / PMF.replicas[idx_j]
                # if not i == zero_index:
                #     de =  PMF.bias_potentials[i] - PMF.bias_potentials[zero_index] * PMF.replicas[i]
                exchanges_prob[i] = min([1.0, exp(-beta * de)])
                psum += exchanges_prob[i]
                # else:
                #     exchanges_prob[i] = 0.0

            idx = random.randint(exchanges_prob.__len__())
            rnum = random.uniform(0.0, 1.0)
            # for idx in range(exchanges_prob.__len__()):
            swapA = sorted_args_replicas[idx]
            swapB = sorted_args_replicas[idx + 1]
            print 'probability of swapping ' + str(PMF.replicas[swapA]) + ' with ' + str(
                PMF.replicas[swapB]) + ' is ' + str(exchanges_prob[idx])
            if rnum < exchanges_prob[idx]:
                print 'swapping ' + str(PMF.replicas[swapA]) + ' with ' + str(PMF.replicas[swapB])
                PMF.replicas[swapA], PMF.replicas[swapB] = PMF.replicas[swapB], PMF.replicas[swapA]

        PMF.replicas = MPI.COMM_WORLD.bcast(PMF.replicas, root=0)
        MPI.COMM_WORLD.Barrier()
