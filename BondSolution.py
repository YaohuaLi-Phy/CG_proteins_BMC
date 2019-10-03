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
        #for i in range(num_hex1):
        #    self.position_list.append(hexamer.all_sites + body_shift * i)
        self.num_hex1 = num_hex1
        self.num_pen = num_pen

    def create_system(self):
        if self.dump_context is None:
            self.dump_context = hoomd.context.initialize("")
        base1 = np.array([1, 0, 0])
        base3 = np.array([0, 0, 1])
        origin = np.array([0, 0, 0])
        hex1_sites = self.hexamer.num_of_sites

        snapshot = hoomd.data.make_snapshot(N=self.num_particles, box=hoomd.data.boxdim(L=self.box_length))
        snapshot.bonds.resize(0)
        for k in range(self.num_hex1):  # repeat pduA
            level = k * hex1_sites
            tag = 0
            for j in range(self.hexamer.num_body):
                snapshot.particles.position[level+tag+j] = self.hexamer.body_sites[j] + base3 * self.hexamer.edge_l * k
                snapshot.particles.mass[k * hex1_sites + j] = 1.0
                snapshot.particles.typeid[k * hex1_sites + j] = 0
                snapshot.particles.charge[k * hex1_sites + j] = 0
            tag += self.hexamer.num_body
            for j in range(self.hexamer.num_hand):
                snapshot.particles.position[level + tag+j] = self.hexamer.hand_sites[j] + base3 * self.hexamer.edge_l * k
                snapshot.particles.mass[level+tag+j] = 1.0
                snapshot.particles.typeid[level+tag+j] = 1
                snapshot.particles.charge[level+tag+j] = 0
            tag += self.hexamer.num_hand
            for j in range(self.hexamer.num_bind):
                snapshot.particles.position[level + tag+j] = self.hexamer.binding_sites[j] + base3 * self.hexamer.edge_l * k
                snapshot.particles.mass[level+tag+j] = 1.0
                snapshot.particles.typeid[level+tag+j] = 2
            tag += self.hexamer.num_bind
            for j in range(self.hexamer.num_posi):
                snapshot.particles.position[level+tag+j] = self.hexamer.p_charge_sites[j] + base3 * self.hexamer.edge_l * k
                snapshot.particles.mass[level+tag+j] = 1.0
                snapshot.particles.typeid[level+tag+j] = 3
                snapshot.particles.charge[level+tag+j] = 1.0
            tag += self.hexamer.num_posi
            for j in range(self.hexamer.num_nega):
                snapshot.particles.position[level + tag+j] = self.hexamer.n_charge_sites[j] + base3 * self.hexamer.edge_l * k
                snapshot.particles.mass[level+tag+j] = 1.0
                snapshot.particles.typeid = 4
                snapshot.particles.charge = -1.0

        system = hoomd.init.read_snapshot(snapshot)

