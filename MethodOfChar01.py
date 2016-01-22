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
TIME_CLOSE = 5  # SECONDS (VALVE CLOSURE DURATION)
TIME = 90  # SECONDS OF ANALYSIS
LENGTH = 3.e3  # FEET (PIPE LENGTH)
DIA = 2  # FEET (DIAMETER)
WAVE_SPEED = 3.7E3  # FEET/SEC (SPEED OF SOUND IN FLUID)
VEL0 = 15.0  # FEET/SEC (INITIAL FLUID VELOCITY)
HEAD_US = 4000.0  # FEET (ELEV + PRESSURE HEADS = TOTAL HEAD AT UPSTREAM DATUM)
THETA = np.pi/2  # ORIENTATION OF PIPE (RADIANS: 0=HORIZ, np.pi/2=VERT)
FRIC = 0.1  # DIMENSIONLESS DARCY-WIESBACH FRICTION FACTOR

# GRID SPACING
NELEM = 4
NNODE = NELEM + 1
DELX = LENGTH/NELEM  # FEET (NODE SPACING)
DELT = DELX/WAVE_SPEED  # SECOND (TIME STEP)
XGRID = np.linspace(start=0, stop=LENGTH, num=NNODE)  # SPATIAL DISC
TGRID = np.linspace(start=0, stop=TIME, num=TIME/DELT+1)  # TEMPORAL DISC

# CONSTANTS
GRAV = 32.2  # FT/SEC2
RHO = 1.94  # SLUGS/FT3
GAMMA = RHO * GRAV  # LB/FT3

# INITIAL CONDITIONS
HEAD_EL = np.sin(THETA)*XGRID  # ELEVATION HEAD
HEAD_V = VEL0**2/(2*GRAV)  # VELOCITY HEAD
HEAD_LS = FRIC*XGRID/(2*DIA*GRAV)*VEL0**2 # HEAD LOSS FROM FRICTION
HEAD0 = HEAD_US - HEAD_LS  # INITIAL CONDITIONS - TOTAL HEAD
DEPTH = HEAD_EL[::-1]

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

# CALCULATE PRESSURE
ROW, COL = HEAD_ARR.shape
PRESS = np.zeros((ROW, COL))
for i in xrange(COL):
    PRESS[:, i] = (HEAD_ARR[:, i] - HEAD_EL - VEL_ARR[:, i]**2/(2*GRAV))\
                 * GAMMA/144.0


# BUILD ARRAYS FOR PLOTTING
time_plot = np.tile(TGRID, (NNODE, 1)).T
x_plot = np.tile(XGRID, (COL, 1)).T
PATH = '/Users/Lampe/Documents/PB/SurgeAnalysis/Results/'

LBL = [None]*ROW
for i in xrange(ROW):
    LBL[i] = "Depth: {:,.0f} ft".format(DEPTH[i])

FIG1, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(time_plot, VEL_ARR.T, 'o--')
AX1.grid(True)
AX1.set_xlabel("Time (sec)")
AX1.set_ylabel("Fluid Velocity (ft/sec)")
AX1.legend(LBL, frameon=1, framealpha=1, loc=0)
FIG1_NAME = 'fig_velocity.pdf'
# FIG1.savefig(PATH + FIG1_NAME)

FIG2, AX2 = plt.subplots(figsize=(12, 8))
AX2.plot(time_plot, PRESS.T, 'o--')
AX2.grid(True)
AX2.set_xlabel("Time (sec)")
AX2.set_ylabel("Pressure (psi)")
AX2.legend(LBL, frameon=1, framealpha=1, loc=1)
FIG2_NAME = 'fig_head.pdf'
# FIG2.savefig(PATH + FIG2_NAME)
