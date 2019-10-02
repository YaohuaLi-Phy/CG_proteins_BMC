from __future__ import division
import numpy as np
from numpy import linalg as la
import hoomd
import copy as cp


class UnitCell(object):
    def __init__(self, pentamer, hexamer, hexamer2, scaffold, num_pen=1, num_hex1=2, num_hex2=1, num_scaffold=2):
        self.dump_context = None
        self.num_particles = hexamer.num_of_sites * num_hex1
        base1 = np.array([1, 0, 0])
        base3 = np.array([0, 0, 1])
        origin = np.array([0, 0, 0])
        self.box_length = num_hex1 * hexamer.edge_l
        self.hexamer = cp.deepcopy(hexamer)
        body_shift = base3 * hexamer.edge_l
        self.position_list = []  # a list of all positions in one unit cell
        for i in range(num_hex1):
            self.position_list.append(hexamer.all_sites + body_shift * i)
        self.num_hex1 = num_hex1
        self.num_pen = num_pen

    def create_system(self):
        if self.dump_context is None:
            self.dump_context = hoomd.context.initialize("")

        hex1_sites = self.hexamer.num_of_sites

        snapshot = hoomd.data.make_snapshot(N=self.num_particles, box=hoomd.data.boxdim(L=self.box_length))
        snapshot.bonds.resize(0)
        for k in range(self.num_hex1):  # repeat pduA
            for j in range(self.hexamer.num_body):
                snapshot.particles.position[k * hex1_sites + j] = self.hexamer.body_sites[j]
                snapshot.particles.mass[k * hex1_sites + j] = 1.0
                snapshot.particles.typeid[k * hex1_sites + j] = 0
                snapshot.particles.charge[k * hex1_sites + j] = 0

            level = k * hex1_sites + self.hexamer.num_body
            for j in range(self.hexamer.num_hand):
                snapshot.particles.position[level + j] = self.hexamer.hand_sites[j]
                snapshot.particles.mass[level + j] = 1.0
                snapshot.particles.typeid[level + j] = 1
                snapshot.particles.charge[level + j] = 0
            level = k*hex1_sites + self.hexamer.num_body + self.hexamer.num_hand
            for j in range(self.hexamer.num_bind):
                snapshot.particles.position[level + j] = self.hexamer.binding_sites[j]
                snapshot.particles.mass[level + j] = 1.0