import hoomd
import hoomd.md
from Pentagon import PentagonBody
from PduABody import PduABody
from NonBonded import LJ_attract
from NonBonded import SoftRepulsive
from NonBonded import Yukawa
from Solution import Lattice
from SphericalTemplate import SphericalTemplate
# Place the type R central particles
hoomd.context.initialize("--mode=cpu");

hexamer1 = PduABody()
pentamer = PentagonBody()
template = SphericalTemplate(1.5)

sys = Lattice(pentamer, hexamer1, num_hex1=2)

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
system = hoomd.init.create_lattice(unitcell=uc, n=[4, 4, 3])

# Add constituent particles of type A and create the rods

added_types = ['A', 'B', 'C', 'D', 'qP', 'Sc', 'Ss']
system_types = ['R', 'P', 'H'] + added_types
for new_type in added_types:
    system.particles.types.add(new_type)
#system.particles.types.add('A');
rigid = hoomd.md.constrain.rigid()


rigid.set_param('R', types=hexamer1.type_list, positions=hexamer1.all_sites)
rigid.set_param('P', types=pentamer.type_list, positions=pentamer.all_sites)
rigid.set_param('H', types=template.type_list, positions=template.position)

rigid.create_bodies()

# forcefield and integration
nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=1000, nlist=nl)
table.pair_coeff.set(system_types, system_types, func=SoftRepulsive, rmin=0.01, rmax=2.5, coeff=dict(sigma=1, epsilon=0.5))
table.pair_coeff.set('C', 'D', func=LJ_attract, rmin=0.01, rmax=2.5, coeff=dict(sigma=1.0, epsilon=5))
table.pair_coeff.set('qP', 'qP', func=Yukawa, rmin=0.01, rmax=2.5, coeff=dict(A=5, kappa=0.1))
table.pair_coeff.set('Sc', 'Sc', func=LJ_attract, rmin=0.01, rmax=2.5, coeff=dict(sigma=1.0, epsilon=5))
table.pair_coeff.set('Sc', 'Ss', func=LJ_attract, rmin=0.01, rmax=2.5, coeff=dict(sigma=1.0, epsilon=5))
#lj = hoomd.md.pair.lj(r_cut=2.5, nlist=nl)
#lj.set_params(mode='shift')
#lj.pair_coeff.set(system_types, system_types, epsilon=0.0, sigma=1.0)
#lj.pair_coeff.set('C', 'D', epsilon=2.0, sigma=1.0)
hoomd.md.integrate.mode_standard(dt=0.004)
rigid = hoomd.group.rigid_center()
hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=42);

hoomd.analyze.log(filename="log-output.log",
                  quantities=['potential_energy',
                              'translational_kinetic_energy',
                              'rotational_kinetic_energy'],
                  period=100,
                  overwrite=True)

hoomd.dump.gsd("traj.gsd",
               period=5e2,
               group=hoomd.group.all(),
               overwrite=True)

hoomd.run(1e6)
