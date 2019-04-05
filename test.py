from  __future__ import division
import hoomd
import hoomd.md
from Pentagon import PentagonBody
from PduABody import PduABody
from NonBonded import LJ_attract
from NonBonded import SoftRepulsive
from NonBonded import Yukawa
from Solution import Lattice
from SphericalTemplate import SphericalTemplate
import sys

from PduABody import PduABody
from Pentagon import PentagonBody
from NonBonded import *


#test_object = PentagonBody()
#test_object.plot_position()


test=PotentialTest(4.0)
test.plot(yukawa_lj)