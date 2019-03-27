import numpy as np
import matplotlib.pyplot as plt

def SoftRepulsive(r,rmin, rmax, sigma, epsilon):

    if r < 2 * sigma:
        V = epsilon * ((sigma / r) ** 12 - 0.5 ** 12)
        F = epsilon/sigma * 12 * (sigma/r) ** 13
    else:
        V = 0
        F = 0
    return (V, F)


def LJ_attract(r,rmin, rmax, sigma, epsilon):
    x6 = (float(sigma)/r)**6
    xmax6 = (sigma/float(rmax))**6
    V = epsilon * ((x6**2-x6) - (xmax6**2-xmax6) - np.exp(-(r-2.2*sigma)**2/(sigma**2)))
    F = epsilon/r * (12 * x6**2 - 6 * x6) + epsilon * np.exp(-(r-2.2*sigma)**2/(2*sigma**2)) *2*(r-2.2*sigma)/(sigma**2)
    return (V, F)


def Yukawa(r, rmin, rmax, kappa, A):
    V = A*np.exp(-kappa*r)/r
    F = V*(kappa*r)/r
    return (V, F)


class potential_test(object):

    def __init__(self):
        self.plot_range = 3
        self.x = np.arange(0.3, self.plot_range, 0.02)

    def plot(self, funct):
        vf = funct(self.x, 0.01, self.plot_range, 1, 5.0)
        y = vf[1]
        plt.plot(self.x, y)
        plt.ylim(-10, 10)
        plt.show()


