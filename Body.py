from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Body(object):

    def __init__(self):
        self.type_list = []
        self.body_sites = []
        self.binding_sites = []
        self.p_charge_sites = []
        self.n_charge_sites = []
        self.scaffold_sites = []


    def plot_position(self):
        coord = np.array(self.body_sites)
        sites = np.array(self.binding_sites)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax = fig.add_subplot(111)
        xs = coord[:, 0]
        ys = coord[:, 1]
        zs = coord[:, 2]
        x1 = sites[:, 0]
        y1 = sites[:, 1]
        z1 = sites[:, 2]
        ax.scatter(xs, ys, zs)
        ax.scatter(x1, y1, z1)
        plt.show()