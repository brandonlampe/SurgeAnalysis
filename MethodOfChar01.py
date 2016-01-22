""" Method of Characteristics
-Derivation from Bruce Thomson class notes (UNM CE540)
-upstream (EOT) is always assumed to be the problem datum (elev=0)

"""

import numpy as np
import matplotlib.pyplot as plt
# import sys
# import pdb

# RUNS ON PYTHON VERSION 2.7+
# print(sys.version_info)
np.set_printoptions(precision=4)


def valprint(string, value):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .6e}".format(string, value))


def matprint(string, value):
    """ Ensure uniform formatting of matrix value outputs. """
    print("{0}:".format(string))
    print(value)

# Problem Data
TIME_CLOSE = 6.0  # SECONDS (VALVE CLOSURE DURATION)
TIME = 45.0 # SECONDS OF ANALYSIS
LENGTH = 11.1e3  # FEET (PIPE LENGTH)
DIA = 2  # FEET (DIAMETER)
WAVE_SPEED = 3.7E3  # FEET/SEC (SPEED OF SOUND IN FLUID)
VEL0 = 6.0  # FEET/SEC (INITIAL FLUID VELOCITY)
HEAD_US = 115.0  # FEET (ELEV + PRESSURE HEADS = TOTAL HEAD AT UPSTREAM DATUM)
THETA = 0.0  # ORIENTATION OF PIPE (RADIANS: 0=HORIZ, np.pi/2=VERT)
FRIC = 0.014  # DIMENSIONLESS DARCY-WIESBACH FRICTION FACTOR

# GRID SPACING
NELEM = 4
NNODE = NELEM + 1
DELX = LENGTH/NELEM  # FEET (NODE SPACING)
DELT = DELX/WAVE_SPEED  # SECOND (TIME STEP)
XGRID = np.linspace(start=0, stop=LENGTH, num=NNODE)  # SPATIAL DISC
TGRID = np.linspace(start=0, stop=TIME, num=TIME/DELT+1)  # TEMPORAL DISC

# CONSTANTS
GRAV = 32.2  # FT/SEC2

# INITIAL CONDITIONS
HEAD_EL = np.sin(THETA)*XGRID  # ELEVATION HEAD
HEAD_PR = HEAD_US - FRIC*XGRID/(2*DIA*GRAV)*VEL0**2  # PRESSURE HEAD
HEAD0 = HEAD_EL + HEAD_PR  # INITIAL CONDITIONS - TOTAL HEAD

# BOUNDARY CONDITIONS
BC_US = HEAD_US  # CONSTANT HEAD AT UPSTREAM RESERVOIR

VEL_DS = np.zeros(len(TGRID))
for i in xrange(len(VEL_DS)):
    frac_open = 1 - TGRID[i]/TIME_CLOSE
    if frac_open > 0:
        VEL_DS[i] = VEL0*frac_open  # PRESCRIBED FLUX AT DOWNSTREAM

# DEFINE ARRAYS TO STORE THE SOLUTION
HEAD_ARR = np.zeros((len(XGRID), len(TGRID)))
VEL_ARR = np.zeros((len(XGRID), len(TGRID)))

# SET INITIAL CONDITIONS
HEAD_ARR[:, 0] = HEAD0
VEL_ARR[:, 0] = VEL0

# SET BCTS
HEAD_ARR[0, :] = HEAD_US  # UPSTREAM HEAD
VEL_ARR[-1, :] = VEL_DS  # DOWNSTREAM VELOCITY

BTERM = WAVE_SPEED/GRAV
RTERM = FRIC*DELX/(2*GRAV*DIA)
XROW = len(XGRID)-2  # INTERNAL SPATIAL GRID POINTS
TCOL = len(TGRID)-1  # INTERNAL TEMPORAL GRID POINTS

# BDRY TERMS AT THE FIRST TIME STEP
bneg = BTERM + RTERM*np.abs(VEL_ARR[1, 0])
cneg = HEAD_ARR[1, 0] - BTERM*VEL_ARR[1, 0]
VEL_ARR[0, 1] = (HEAD_US - cneg)/bneg

bpos = BTERM + RTERM*np.abs(VEL_ARR[-2, 0])
cpos = HEAD_ARR[-2, 0] + BTERM*VEL_ARR[-2, 0]
HEAD_ARR[-1, 1] = cpos - bpos*VEL_ARR[-1, 1]

for j in xrange(TCOL):
    for i in xrange(XROW):  # INTERIOR NODE CALCULATIONS
        cpos = HEAD_ARR[i, j] + BTERM*VEL_ARR[i, j]
        cneg = HEAD_ARR[i+2, j] - BTERM*VEL_ARR[i+2, j]
        bpos = BTERM + RTERM*np.abs(VEL_ARR[i, j])
        bneg = BTERM + RTERM*np.abs(VEL_ARR[i+2, j])
        HEAD_ARR[i+1, j+1] = (cpos*bneg + cneg*bpos)/(bpos+bneg)
        VEL_ARR[i+1, j+1] = (cpos - cneg)/(bpos + bneg)
        if i == 0:  # UPSTREAM BOUNDARY NODE - VELOCITY
            cneg = HEAD_ARR[i+1, j] - BTERM*VEL_ARR[i+1, j]
            bneg = BTERM + RTERM*np.abs(VEL_ARR[i+1, j])
            VEL_ARR[i, j+1] = (HEAD_US - cneg)/bneg
        if i == XROW-1:  # DOWNSTREAM BOUNDARY NODE - HEAD
            cpos = HEAD_ARR[i+1, j] + BTERM*VEL_ARR[i+1, j]
            bpos = BTERM + RTERM*np.abs(VEL_ARR[i+1, j])
            HEAD_ARR[-1, j+1] = cpos - bpos*VEL_ARR[-1, j+1]

time_plot = np.tile(TGRID, (NNODE, 1)).T
x_plot = np.tile(XGRID, (len(HEAD_ARR[0, :]), 1)).T

# PLOT RESULTS
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(time_plot, HEAD_ARR.T, 'o--')
# ax.legend(lbl, frameon=1, loc=0)
ax.grid(True)
plt.show()