from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class PeanutTemplate(object):
    def __init__(self, radius):
        sa = 4 * np.pi * radius * radius
        area = .25 * np.pi
        length = int(sa / area)
        length2 = 2* length
        self.position = [[] for i in range(length2)]
        self.rigid_count = 1
        self.type_list = ['Sc'] * length2
        self.diameter = 2 * radius
        left_half = np.multiply(self.unit_sphere(length, -1.0), radius)

        right_half = np.multiply(self.unit_sphere(length, 1.0), radius)
        points = np.concatenate((left_half, right_half), axis=0)


        for ind, point in enumerate(points):
            self.position[ind] = point
            # self.mass[ind] = 100
            # self.type[ind] = 'qPm'

    def unit_sphere(self, n, shift):

        points = []
        offset = 2. / n
        increment = np.pi * (3. - np.sqrt(5.))

        for i in range(n):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - pow(y, 2))

            phi = i * increment
            #print 'shift='
            #print shift
            x = np.cos(phi) * r
            z = np.sin(phi) * r

            points.append([x, y + shift, z])

        return points

    def plot_position(self):
        coord = np.array(self.position)
        #print coord
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax = fig.add_subplot(111)
        xs = coord[:, 0]
        ys = coord[:, 1]
        zs = coord[:, 2]
        ax.scatter(xs, ys, zs)
        plt.show()
