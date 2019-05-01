from __future__ import division
import numpy as np
from Body import Body


class PduBBody(Body):
    """ PduA protein"""

    def __init__(self, edge_length=2.0, angle=0):
        super(PduBBody, self).__init__()
        l_spacing = edge_length / 2.0
        height = 0.3 * edge_length
        base1 = l_spacing * np.array([0, 1, 0])
        base2 = l_spacing * np.array([np.sqrt(3) / 2.0, -0.5, 0])
        base3 = height * np.array([0, 0, 1])
        radius_factor = 1 - angle * 0.8 / np.sqrt(3)
        for i in range(-2, 3):
            for j in range(-2, 3):
                position = base1 * i + base2 * j
                if -2.01 * l_spacing < position[1] - 0.5 * position[0] < 2.01 * l_spacing:
                    self.body_sites.append(tuple(position + base3))
                    self.body_sites.append(tuple(position))
                    self.body_sites.append(tuple(radius_factor * position - base3))

        self.hand_sites = []

        vertices = []
        vertices.append(base1)
        vertices.append((base1 + base2))
        vertices.append(base2)
        vertices.append(-base1)
        vertices.append(-(base2 + base1))
        vertices.append(-base2)
        for i in range(len(vertices)):
            vector = (vertices[i] - vertices[i - 1])
            self.binding_sites.append(list(2.1 * (vertices[i - 1] + vector * 0.3)))
            self.hand_sites.append(list(2.1 * (vertices[i - 1] + vector * 0.7)))
            self.n_charge_sites.append(list(vertices[i])-0.5*base3)
            self.scaffold_sites.append(list(vertices[i]-1.5*base3))

        self.all_sites = self.body_sites + self.hand_sites + self.binding_sites + self.n_charge_sites + self.scaffold_sites
        self.type_list = ['B'] * len(self.body_sites) + \
                         ['C'] * len(self.hand_sites) + \
                         ['D'] * len(self.binding_sites) + \
                         ['qN'] * len(self.n_charge_sites) + \
                        ['Ss'] * len(self.scaffold_sites)
        self.types = ['B', 'C', 'D', 'qN', 'Ss']
        self.moment_of_inertia = [2, 2, 3]
