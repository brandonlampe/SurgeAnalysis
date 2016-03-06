""" Analytical solution to 1D wave equation
    -   semi-inifinite bar
    -   takes form provided in ME504 (Buck Schreyer)

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/Lampe/PyScripts/surgeAnalysis')
import SurgeFunc as sf


T_TOT = 25.0  # [seconds] total analysis time duration
T_DUR = 5.0  # [SEC] DURATION OF TRACTION PULSE
EMOD = 2.2E9  # [PA] ELASTIC MODULUS (water)
RHO = 1000.0  # [KG/M3] DENSITY (water)

AREA = 1.0  # [M2] CROSS-SECTIONAL AREA
# SIG_YIELD = 20.0E9  # [PA] YIELD STRESS
SIG_ZERO = 100.0E3  # [PA] AMPLITUDE OF PULSE

# DISCRETIZATION
T_INC = 200  # NUMBER OF TIME INCREMENTS
X_INC = 100  # NUMBER OF SPATIAL INCREMENTS

# CALCULATED PARAMETERS
CEL = np.sqrt(EMOD/RHO)  # [M/SEC]
LWAVE = CEL*T_DUR  # [M]
sf.valprint("wavelength", LWAVE)
LEN = LWAVE*4  # [M] LENGTH OF DOMAIN: 4 WAVELENGTHS

T_NNODE = T_INC + 1  # NUMBER OF TIME NODES
X_NNODE = X_INC + 1  # NUMBER OF SPATIAL NODES
T_ELEM = T_TOT/T_INC  # SIZE OF EACH TIME STEP
X_ELEM = LEN/X_INC  # SIZE OF EACH SPATIAL STEP


# CREATE VECTORS
TVECT = np.linspace(0, T_TOT, T_NNODE)  # TIME VECTOR
XVECT = np.linspace(0, LEN, X_NNODE)  # SPACE VECTOR
TRAC_VECT = np.zeros(len(TVECT))  # EMPTY TRACTION VECTOR
# ETA_POS = TVECT - XVECT/CEL  # [SEC] IND VAR FOR WAVE TRAVELING IN POSITIVE X
# ETA_NEG = TVECT + XVECT/CEL  # [SEC] IND VAR FOR WAVE TRAVELING IN NEGATIVE X

# CREATE THE TRACTION VECTOR (PULSE)
for i in xrange(len(TVECT)):
    if TVECT[i] <= T_DUR:
        TRAC_VECT[i] = -SIG_ZERO / 2.0 * (1 - np.cos(2 *
                                          np.pi * TVECT[i] / T_DUR))

# REFERENCE VALUES
T_REF = T_DUR  # [SEC]
X_REF = LWAVE  # [M]
SIG_REF = SIG_ZERO  # [PA]
VEL_REF = SIG_ZERO / np.sqrt(RHO*EMOD)
DISP_REF = (SIG_ZERO*T_DUR) / np.sqrt(RHO*EMOD)

# DIMENSIONLESS SCALARS
SIG_BAR = SIG_ZERO/SIG_REF

# # DIMENSIONLESS VECTORS
TBAR_VECT = TVECT/T_REF
XBAR_VECT = XVECT/X_REF

# CREATE 2D ARRAYS: [X DIST, TIME]
DISP_BAR_ARR = np.zeros((len(XVECT), len(TVECT)))  # STRESS ARRAY
VEL_BAR_ARR = np.zeros((len(XVECT), len(TVECT)))  # VELOCITY ARRAY
SIG_BAR_ARR = np.zeros((len(XVECT), len(TVECT)))  # STRESS ARRAY
STOR_ETA_PLUS = np.zeros((len(XVECT), len(TVECT)))  # STRESS ARRAY

for i in xrange(len(TBAR_VECT)):
    for j in xrange(len(XBAR_VECT)):
        # ETA_POS = TVECT[i] - XVECT[j]/CEL
        # ETA_POS_BAR = ETA_POS/T_REF
        ETA_POS_BAR = TBAR_VECT[i] - XBAR_VECT[j]
        STOR_ETA_PLUS[j, i] = ETA_POS_BAR
        ETA_NEG_BAR = TBAR_VECT[i] + XBAR_VECT[j]

        if ETA_POS_BAR >= 0.0 and ETA_POS_BAR <= 1.0:
            DISP_BAR_ARR[j, i] = 0.5*(ETA_POS_BAR-np.sin(2*np.pi*ETA_POS_BAR) /
                                      (2 * np.pi))
            VEL_BAR_ARR[j, i] = 0.5*(1-np.cos(2*np.pi*ETA_POS_BAR))
            SIG_BAR_ARR[j, i] = -0.5*(1-np.cos(2*np.pi*ETA_POS_BAR))
        if ETA_POS_BAR >= 1.0:
            DISP_BAR_ARR[j, i] = 0.5
            # VEL_BAR_ARR[j, i] = 0.5*(1-np.cos(2*np.pi*ETA_POS_BAR))
            # SIG_BAR_ARR[j, i] = -0.5*(1-np.cos(2*np.pi*ETA_POS_BAR))

# BUILD ARRAYS FOR PLOTTING
TIME_PLOT = np.tile(XBAR_VECT, (T_NNODE, 1)).T  # POINTS IN TIME TO BE PLOTTED
X_PLOT = np.tile(XBAR_VECT, (T_NNODE, 1)).T  # POINTS IN SPACE TO BE PLOTTED
# PATH = '/Users/Lampe/Documents/PB/SurgeAnalysis/Results/'

TSTART = 0
TSTOP = len(TVECT)
TINC = len(TVECT)/5
LBL_TALL = [None]*T_NNODE
for i in xrange(T_NNODE):
    LBL_TALL[i] = "Time: {:,.3f}".format(TBAR_VECT[i])
LBL_T = LBL_TALL[TSTART:TSTOP:TINC]

FIG_DISP, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(TIME_PLOT[:,TSTART:TSTOP:TINC], DISP_BAR_ARR[:, TSTART:TSTOP:TINC], 'o-')
AX1.grid(True)
AX1.set_xlabel("X Location")
AX1.set_ylabel("Displacement")
AX1.legend(LBL_T, frameon=1, framealpha=1, loc=0)
FIG_DISP_NAME = 'fig_disp.pdf'
# FIG_DISP.savefig(PATH + FIG_DISP_NAME)

FIG_VEL, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(TIME_PLOT[:,TSTART:TSTOP:TINC], VEL_BAR_ARR[:, TSTART:TSTOP:TINC], 'o-')
AX1.grid(True)
AX1.set_xlabel("X Location")
AX1.set_ylabel("Velocity")
AX1.legend(LBL_T, frameon=1, framealpha=1, loc=0)
FIG_VEL_NAME = 'fig_velocity.pdf'
# FIG_VEL.savefig(PATH + FIG_VEL_NAME)

FIG_STRS, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(TIME_PLOT[:,TSTART:TSTOP:TINC], SIG_BAR_ARR[:, TSTART:TSTOP:TINC], 'o-')
AX1.grid(True)
AX1.set_xlabel("X Location")
AX1.set_ylabel("Stress")
AX1.legend(LBL_T, frameon=1, framealpha=1, loc=0)
FIG_STRS_NAME = 'fig_stress.pdf'
# FIG_STRS.savefig(PATH + FIG_STRS_NAME)

# FIG_TRACTION, AX1 = plt.subplots(figsize=(12, 8))
# AX1.plot(TVECT,TRAC_VECT)

# if PLOT != 0:
plt.show()
