""" Analytical solution to 1D wave equation
    -   semi-inifinite bar
    -   takes form provided in ME504 (Buck Schreyer)

"""
import numpy as np
import matplotlib as mpl
# mpl.use('MacOSX')
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
T_INC = 300  # NUMBER OF TIME INCREMENTS
X_INC = 300  # NUMBER OF SPATIAL INCREMENTS

# CALCULATED PARAMETERS
CEL = np.sqrt(EMOD/RHO)  # [M/SEC]
LWAVE = CEL*T_DUR  # [M]
sf.valprint("wavelength", LWAVE)
LEN = LWAVE*5  # [M] LENGTH OF DOMAIN: 4 WAVELENGTHS

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
X_PLOT = np.tile(TBAR_VECT, (X_NNODE, 1))  # POINTS IN SPACE TO BE PLOTTED
# PATH = '/Users/Lampe/Documents/PB/SurgeAnalysis/Results/'

TSTART = 0
TSTOP = len(TVECT)
TINC = len(TVECT)/5
LBL_TALL = [None]*T_NNODE
for i in xrange(T_NNODE):
    LBL_TALL[i] = "Time: {:,.3f}".format(TBAR_VECT[i])
LBL_T = LBL_TALL[TSTART:TSTOP:TINC]

XSTART = 0
XSTOP = len(XVECT)
XINC = len(XVECT)/5
LBL_XALL = [None]*X_NNODE
for i in xrange(X_NNODE):
    LBL_XALL[i] = "X Dist: {:,.3f}".format(XBAR_VECT[i])
LBL_X = LBL_XALL[XSTART:XSTOP:XINC]

# PLOT VELOCITY
FIG_VELT, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(TIME_PLOT[:,TSTART:TSTOP:TINC], VEL_BAR_ARR[:, TSTART:TSTOP:TINC], '-o')
AX1.grid(True)
AX1.set_xlabel("X Location")
AX1.set_ylabel("Velocity")
AX1.legend(LBL_T, frameon=1, framealpha=1, loc=0)
FIG_VELT_NAME = 'fig_velocity.pdf'
# # FIG_VELT.savefig(PATH + FIG_VELT_NAME)

FIG_VELX, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(X_PLOT[XSTART:XSTOP:XINC,:].T, VEL_BAR_ARR[XSTART:XSTOP:XINC,:].T, '-o')
AX1.grid(True)
AX1.set_xlabel("Time")
AX1.set_ylabel("Velocity")
AX1.legend(LBL_X, frameon=1, framealpha=1, loc=0)
FIG_VELX_NAME = 'fig_velocity.pdf'
# # FIG_VELX.savefig(PATH + FIG_VELX_NAME)

# PLOT DISPLACEMENT
# FIG_DISP, AX1 = plt.subplots(figsize=(12, 8))
# AX1.plot(TIME_PLOT[:,TSTART:TSTOP:TINC], DISP_BAR_ARR[:, TSTART:TSTOP:TINC], 'o-')
# AX1.grid(True)
# AX1.set_xlabel("X Location")
# AX1.set_ylabel("Displacement")
# AX1.legend(LBL_T, frameon=1, framealpha=1, loc=0)
# FIG_DISP_NAME = 'fig_disp.pdf'
# # FIG_DISP.savefig(PATH + FIG_DISP_NAME)

# PLOT STRESS
# FIG_STRS, AX1 = plt.subplots(figsize=(12, 8))
# AX1.plot(TIME_PLOT[:,TSTART:TSTOP:TINC], SIG_BAR_ARR[:, TSTART:TSTOP:TINC], 'o-')
# AX1.grid(True)
# AX1.set_xlabel("X Location")
# AX1.set_ylabel("Stress")
# AX1.legend(LBL_T, frameon=1, framealpha=1, loc=0)
# FIG_STRS_NAME = 'fig_stress.pdf'
# # FIG_STRS.savefig(PATH + FIG_STRS_NAME)

# PLOT TRACTION AB X=0
# FIG_TRACTION, AX1 = plt.subplots(figsize=(12, 8))
# AX1.plot(TVECT,TRAC_VECT)

# if PLOT != 0:
plt.show()
# plt.show()
