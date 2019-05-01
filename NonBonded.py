import numpy as np
import matplotlib.pyplot as plt


def SoftRepulsive(r,rmin, rmax, sigma, epsilon):

    V = epsilon * ((sigma / r) ** 12 - 0.5 ** 12)
    F = epsilon/sigma * 12 * (sigma/r) ** 13
    return (V, F)


def LJ_attract(r, rmin, rmax, sigma, epsilon):
    x6 = (float(sigma)/r)**6
    xmax6 = (sigma/float(rmax))**6
    mod_factor = 0.8
    V = epsilon * ((x6**2-x6) - (xmax6**2-xmax6) - mod_factor * np.exp(-(r-1.2*sigma)**2/(0.3*sigma**2)))
    F = epsilon/r * (12 * x6**2 - 6 * x6) - epsilon * mod_factor * np.exp(-(r-1.2*sigma)**2/(0.3*sigma**2)) *2*(r-1.2*sigma)/(0.3*sigma**2)
    return (V, F)


def normal_lj(r, rmin, rmax, sigma, epsilon):
    x6 = (float(sigma) / r) ** 6
    xmax6 = (sigma / float(rmax)) ** 6
    V = epsilon * ((x6 ** 2 - x6) - (xmax6 ** 2 - xmax6))
    F = epsilon / r * (12 * x6 ** 2 - 6 * x6)
    return (V, F)


def Yukawa(r, rmin, rmax, kappa, A):
    V = A*np.exp(-kappa*r)/r
    F = V*(kappa*r)/r
    return (V, F)

def yukawa_lj(r, rmin, rmax, sigma, epsilon, kappa, A):
    x6 = (float(sigma) / r) ** 6
    xmax6 = (sigma / float(rmax)) ** 6

    V = epsilon * ((x6 ** 2 - x6) - (xmax6 ** 2 - xmax6)) + A*np.exp(-kappa*r)/r
    F = epsilon / r * (12 * x6 ** 2 - 6 * x6) + A*np.exp(-kappa*r)/r*(kappa*r)/r
    return (V, F)

class PotentialTest(object):

    def __init__(self, param):
        self.plot_range = 8
        self.x = np.arange(0.2, self.plot_range, 0.02)
        self.param = param
    def plot(self, funct):
        vf = funct(self.x, 2, self.plot_range, 1.15, self.param)
        y = vf[0]
        plt.plot(self.x, y)
        plt.plot(self.x, vf[1])
        plt.ylim(-10, 10)
        plt.grid()
        plt.show()


