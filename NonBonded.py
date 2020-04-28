import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 10
# comment the below line because it also change the circle edge width
#mpl.rcParams['lines.markeredgewidth'] = 3 # plus cross marker
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18


def SoftRepulsive(r,rmin, rmax, sigma, epsilon):

    V = epsilon * ((sigma / r) ** 12 - 0.5 ** 12)
    F = epsilon/sigma * 12 * (sigma/r) ** 13
    return (V, F)

def RepulsiveLJ(r, rmin, rmax, sigma, epsilon):
    V = epsilon * ((sigma/r)**12 - (sigma/r)**6)
    F = epsilon / sigma * 12 * (sigma / r) ** 13
    return (V, F)

def LJ_attract(r, rmin, rmax, epsilon, sigma, r0):
    x6 = (float(sigma)/r)**6
    xmax6 = (sigma/float(rmax))**6
    mod_factor = 0.8
    V = epsilon * ((x6**2-x6) - (xmax6**2-xmax6) - mod_factor * np.exp(-(r-r0*sigma)**2/(0.3*sigma**2)))
    F = epsilon/r * (12 * x6**2 - 6 * x6) - epsilon * mod_factor * np.exp(-(r-r0*sigma)**2/(0.3*sigma**2)) *2*(r-r0*sigma)/(0.3*sigma**2)
    return (V, F)


def normal_lj(r, rmin, rmax, epsilon, sigma):
    x6 = (float(sigma) / r) ** 6
    xmax6 = (sigma / float(rmax)) ** 6
    V = epsilon * ((x6 ** 2 - x6) - (xmax6 ** 2 - xmax6))
    F = epsilon / r * (12 * x6 ** 2 - 6 * x6)
    return (V, F)


def Yukawa(r, rmin, rmax, kappa, A):
    V = A*np.exp(-kappa*r)/r + (1.0/r)**12
    F = V*(kappa*r)/r + 12 * (1.0 / r) ** 13
    return (V, F)

def yukawa_lj(r, rmin, rmax, sigma, epsilon, kappa, A):
    x6 = (float(sigma) / r) ** 6
    xmax6 = (sigma / float(rmax)) ** 6

    V = epsilon * ((x6 ** 2 - x6) - (xmax6 ** 2 - xmax6)) + A*np.exp(-kappa*r)/r
    F = epsilon / r * (12 * x6 ** 2 - 6 * x6) + A*np.exp(-kappa*r)/r*(kappa*r)/r
    return (V, F)


def LJ_finite(r, rmin, rmax, epsilon, sigma):
    sigmoid = 1/(1+np.exp(-20*(r-0.8*sigma)))
    ri2 = (sigma / r) * (sigma / r)
    sig_r_delta6 = ri2 * ri2 * ri2
    shift = 4 * epsilon * ((sigma/rmax)**12 - (sigma/rmax)**6)/(1+np.exp(-10*(rmax-0.9)))
    potential = 4 * epsilon * (sig_r_delta6 * sig_r_delta6 - sig_r_delta6)*sigmoid-shift
    f = 4 * epsilon / r * (12 * sig_r_delta6 * (sig_r_delta6 - 0.5))*sigmoid

    return (potential, f)

class PotentialTest(object):

    def __init__(self):
        self.plot_range = 1.12
        self.x = np.arange(0.2, self.plot_range, 0.01)

    def plot(self, funct):
        vf = funct(self.x, 0.1, self.plot_range, 1.0, 1.0)
        y = vf[0]
        plt.plot(self.x, y)
        #plt.plot(self.x, vf[1])
        plt.ylim(-1, 1000)
        plt.xlabel('distance (nm)')
        plt.ylabel('potential (kTCal/mol)')
        #plt.grid()
        plt.show()
