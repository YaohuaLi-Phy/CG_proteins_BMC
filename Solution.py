import numpy as np
class Lattice(object):

    def __init__(self, pentamer, hexamer, hexamer2,scaffold, num_pen=1, num_hex1=2, num_hex2=1, num_scaffold=2):

        mass_pen = 5.0
        mass_hex1 = 6.0
        mass_hex2 = 10.0
        mass_scaffold = 2.0
        self.num_body = num_pen + num_hex1 + num_scaffold + num_hex2
        thickness = 3.0/2*hexamer.edge_l
        self.cell_height = num_pen * thickness + (num_hex1 + num_hex2) * thickness + num_scaffold*2 + 2.0

        base1 = np.array([1, 0, 0])
        base3 = np.array([0, 0, 1])
        origin = np.array([0, 0, 0])
        self.mass_list = [mass_pen]*num_pen + [mass_hex1]*num_hex1 + [mass_hex2]*num_hex2 + [mass_scaffold]*num_scaffold

        self.position_list = []
        for i in range(num_pen):
            self.position_list.append(list(origin + base3*(i*thickness - self.cell_height/2.0)))

        for i in range(num_hex1):
            self.position_list.append(list(origin + base3*((i+num_pen)*thickness - self.cell_height/2.0)))

        for i in range(num_hex2):
            self.position_list.append(list(origin + base3*((i+num_pen+num_hex1)*thickness - self.cell_height/2.0)))

        if num_scaffold % 2 == 0:
            for idx in range(int(num_scaffold/2)):
                self.position_list.append(list(origin + base1 * 2.5*hexamer.edge_l + (idx-self.cell_height/2.0) * base3 * 1.3*scaffold.diameter))
                self.position_list.append(list(origin - base1 * 2.5*hexamer.edge_l + (idx-self.cell_height/2.0) * base3 * 1.3*scaffold.diameter))
        else:
            for idx in range(int(num_scaffold)):
                self.position_list.append(list(origin + base1 * 2.5 * hexamer.edge_l + (
                idx - self.cell_height / 2.0) * base3 * 1.3 * scaffold.diameter))

        self.moment_inertias = []
        for i in range(num_pen):
            self.moment_inertias.append(pentamer.moment_of_inertia)
        for i in range(num_hex1):
            self.moment_inertias.append(hexamer.moment_of_inertia)
        for i in range(num_hex2):
            self.moment_inertias.append(hexamer2.moment_of_inertia)
        for i in range(num_scaffold):
            self.moment_inertias.append([1, 1, 1])

        self.type_name_list = ['P']*num_pen + ['R']*num_hex1 + ['R2']*num_hex2 + ['H']*num_scaffold
        self.orientation_list = []
        for i in range(self.num_body):
            self.orientation_list.append([1, 0, 0, 0])
        self.num_stacks = num_hex1*2 + 1 + num_hex2

