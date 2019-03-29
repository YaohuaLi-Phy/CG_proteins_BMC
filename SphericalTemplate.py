from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class SphericalTemplate(object):

    def __init__(self, radius):
        sa = 4 * np.pi * radius * radius
        area = .25 * np.pi
        length = int(sa / area)
        self.position = [[] for i in range(length)]
        self.rigid_count = 1
        self.type_list = ['Sc']*length
	self.diameter = 2*radius
        points = np.multiply(self.unit_sphere(length), radius)

        for ind, point in enumerate(points):
            self.position[ind] = point
            #self.mass[ind] = 100
            #self.type[ind] = 'qPm'

        print self.position

    def unit_sphere(self, n):

        points = []
        offset = 2. / n
        increment = np.pi * (3. - np.sqrt(5.));

        for i in range(n):
            y = ((i * offset) - 1) + (offset / 2);
            r = np.sqrt(1 - pow(y, 2))

            phi = i * increment

            x = np.cos(phi) * r
            z = np.sin(phi) * r

            points.append([x, y, z])

        return points

    def plot_position(self):
        coord = np.array(self.position)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax = fig.add_subplot(111)
        xs = coord[:, 0]
        ys = coord[:, 1]
        zs = coord[:, 2]
        ax.scatter(xs, ys, zs)
        plt.show()
