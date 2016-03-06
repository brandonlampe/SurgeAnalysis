"""
Results from this analysis show that the wave speed in brine is essentially
 constant over the pressures observed during typical storage operations.

avg SG = 1.1766
speed of sound = 4624.8 ft/sec
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/Lampe/PyScripts/surgeAnalysis')
import SurgeFunc as sf

PSI = np.linspace(100, 3000)
# sf.matprint("PSI", PSI)

COMP = sf.CompBrine(100, PSI)
BULK = 1/COMP
DEN_GCC = sf.DBrine(100, PSI)  # OR SPECIFIC GRAVITY
# sf.matprint("SG", DEN_GCC)
SG = np.mean(DEN_GCC)
sf.valprint("Avg SG", SG)

DEN_SLUG = 1.94 * DEN_GCC
QUOT = (144.0)/(COMP*DEN_SLUG)
# sf.matprint("QUOT", QUOT)
SPEED = np.sqrt(QUOT)  # FT/SEC
sf.matprint("SPEED OF SOUND", SPEED)

PATH = '/Users/Lampe/Documents/PB/SurgeAnalysis/Results/'
FIG2, AX2 = plt.subplots(figsize=(12, 8))
AX2.plot(PSI, SPEED, 'o--')
AX2.grid(True)
AX2.set_xlabel("Pressure (psi)")
AX2.set_ylabel("Wave Speed (ft/sec)")
# AX2.legend(LBL, frameon=1, framealpha=1, loc=1)
FIG2_NAME = 'brine_speed.pdf'
FIG2.savefig(PATH + FIG2_NAME)
plt.show()
