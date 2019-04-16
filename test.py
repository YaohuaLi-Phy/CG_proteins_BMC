from  __future__ import division
import hoomd
import hoomd.md
from Pentagon import PentagonBody
from PduABody import PduABody
from NonBonded import LJ_attract
from NonBonded import SoftRepulsive
from NonBonded import Yukawa
from Solution import Lattice
from PeanutTemplate import PeanutTemplate
import sys

from PduABody import PduABody
from Pentagon import PentagonBody
from NonBonded import *


#test_object = PentagonBody()
#test_object.plot_position()
a=0.5
lB = 1.0
kp = 1.1
z_q = 1.0
A_yuka = z_q**2 * lB * (np.exp(kp*a)/(1+kp*a))**2

#temp = PeanutTemplate(2.5)
#temp.plot_position()
test=PotentialTest(10.0)
test.plot(LJ_attract)