from __future__ import division
import numpy as np
from Body import Body


class PentagonBody(Body):
    """PduN pentemer"""

    def __init__(self, edge_length=2.0):
        super(PentagonBody, self).__init__()
        self.body_sites.append([0, 0, 0])
        edge = edge_length
        height = 0.3*edge_length
        radius = edge / (2 * np.sin(np.pi / 5))
        pts = np.array([[0, radius, 0], [edge * np.cos(np.pi / 5), radius - edge * np.sin(np.pi / 5), 0],
                        [radius * np.sin(np.pi / 5), -radius * np.cos(np.pi / 5), 0],
                        [-radius * np.sin(np.pi / 5), -radius * np.cos(np.pi / 5), 0],
                        [-edge * np.cos(np.pi / 5), radius - edge * np.sin(np.pi / 5), 0]])
        # for i in range(len(pts)):
        #    self.body_sites.append(list(pts[i]))
        num_bead = 2
        num_stack = 2
        z_unit_vector = np.array([0, 0, -1])
        for j in range(5):
            vect = pts[j] - pts[j - 1]
            self.binding_sites.append(list(1.1*(pts[j] - 0.3*vect)))

            for i in range(num_bead):
                l_scale = (i + 1.0) / (float(num_bead) + 1.0)

                if j != 0:
                    pts = np.vstack((pts, pts[j-1] + l_scale * vect))
                else:
                    pts = np.vstack([pts, pts[4] + l_scale * vect])

        num_layers = 2
        for j in range(num_layers):
            r_scale = 1.0 / (j + 1.0)
            for i in range(len(pts)):
                self.body_sites.append(list(r_scale * pts[i] + z_unit_vector*height))
                self.body_sites.append(list(r_scale * pts[i] - z_unit_vector*height))

        self.hand_sites = []
        self.all_sites = self.body_sites + self.hand_sites + self.binding_sites
        self.type_list = ['B'] * len(self.body_sites) + ['C'] * len(self.hand_sites) + ['D'] * len(self.binding_sites)
        self.moment_of_inertia = [1, 1, 2]

