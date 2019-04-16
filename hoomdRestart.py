import sys
#sys.path.append('/projects/b1030/hoomd/hoomd-2.5.1/')
import hoomd, hoomd.md
hoomd.context.initialize()
uc = hoomd.lattice.unitcell(N = 1,
                            a1 = [10.8, 0,   0],
                            a2 = [0,    1.2, 0],
                            a3 = [0,    0,   1.2],
                            dimensions = 3,
                            position = [[0,0,0]],
                            type_name = ['R'],
                            mass = [1.0],
                            moment_inertia = [[0,
                                               1/12*1.0*8**2,
                                               1/12*1.0*8**2]],
                            orientation = [[1, 0, 0, 0]]);
system = hoomd.init.create_lattice(unitcell=uc, n=[2,18,18]);

# Add constituent particles of type A and create the rods
system.particles.types.add('A');
rigid = hoomd.md.constrain.rigid();
rigid.set_param('R',
                types=['A']*8,
                positions=[(-4,0,0),(-3,0,0),(-2,0,0),(-1,0,0),
                           (1,0,0),(2,0,0),(3,0,0),(4,0,0)]);

rigid.create_bodies()
rigid = hoomd.group.rigid_center();
integrator = hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=42);

hoomd.analyze.log(filename="restart_test.log",
                  quantities=['temperature', 'potential_energy',
                                       'translational_kinetic_energy',
                                       'rotational_kinetic_energy'],
                  period=100,
                  overwrite=True)

hoomd.run(1e4)
integrator.disable()
for i in range(10):
    integrator = hoomd.md.integrate.langevin(group=rigid, kT=1.2, seed=4)
    hoomd.run(1000)
    integrator.disable()
