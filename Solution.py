import numpy as np
class Lattice(object):

    def __init__(self, pentamer, hexamer, num_pen=1, num_hex1=2, num_scaffold=2):

        mass_pen = 5.0
        mass_hex1 = 6.0
        mass_scaffold = 1.0
        self.num_body = num_pen + num_hex1 + num_scaffold
        thickness = 3.0
        self.cell_height = num_pen * thickness + num_hex1 * thickness + num_scaffold + 2.0

        base1 = np.array([1, 0, 0])
        base3 = np.array([0, 0, 1])
        origin = np.array([0, 0, 0])
        self.mass_list = [mass_pen]*num_pen + [mass_hex1]*num_hex1 + [mass_scaffold]*num_scaffold

        self.position_list = []
        for i in range(num_pen):
            self.position_list.append(list(origin + base3*(i*thickness - self.cell_height/2.0)))

        for i in range(num_hex1):
            self.position_list.append(list(origin + base3*((i+num_pen)*thickness - self.cell_height/2.0)))

        for idx in range(int(num_scaffold/2)):
            self.position_list.append(list(origin + base1 * 5 + (idx-self.cell_height/2.0) * base3 * 4))
            self.position_list.append(list(origin - base1 * 5 + (idx-self.cell_height/2.0) * base3 * 4))

        self.moment_inertias = []
        for i in range(num_pen):
            self.moment_inertias.append(pentamer.moment_of_inertia)
        for i in range(num_hex1):
            self.moment_inertias.append(hexamer.moment_of_inertia)
        for i in range(num_scaffold):
            self.moment_inertias.append([2, 1, 1])

        self.type_name_list = ['P']*num_pen + ['R']*num_hex1 + ['H']*num_scaffold
        self.orientation_list = []
        for i in range(self.num_body):
            self.orientation_list.append([1, 0, 0, 0])
        self.num_stacks = num_hex1*2 + 1
        print self.orientation_list
        print self.moment_inertias
