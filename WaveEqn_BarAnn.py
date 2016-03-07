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

PLOT = 1  # 0=RESULTS NOT PLOTTED, 1=PLOT RESULTS
DIM = 1  # 0=PLOTS IN DIMENSIONLESS FOR, 1=USES DIMENSIONS
# PATH = '/Users/Lampe/Documents/PB/SurgeAnalysis/Results/'

T_TOT = 20.0  # [seconds] total analysis time duration
T_DUR = 5.0  # [SEC] DURATION OF TRACTION PULSE
EMOD = 2.2E9  # [PA] ELASTIC MODULUS (water)
RHO = 1000.0  # [KG/M3] DENSITY (water)

AREA = 1.0  # [M2] CROSS-SECTIONAL AREA
# SIG_YIELD = 20.0E9  # [PA] YIELD STRESS
SIG_ZERO = 100.0E3  # [PA] AMPLITUDE OF PULSE

# DISCRETIZATION
T_INC = 200  # NUMBER OF TIME INCREMENTS
X_INC = 200  # NUMBER OF SPATIAL INCREMENTS

# CALCULATED PARAMETERS
CEL = np.sqrt(EMOD/RHO)  # [M/SEC]
LWAVE = CEL*T_DUR  # [M]
sf.valprint("wavelength", LWAVE)
LEN_REAL = LWAVE*4  # [M] LENGTH OF REAL DOMAIN, 1/2 the modeled domain

LEN = LEN_REAL*2  # MODELED SPATIAL DOMAIN IS TWICE THAT OF THE REAL DOMAIN
T_NNODE = T_INC + 1  # NUMBER OF TIME NODES
X_NNODE = X_INC + 1  # NUMBER OF SPATIAL NODES
T_ELEM = T_TOT/T_INC  # SIZE OF EACH TIME STEP
X_ELEM = LEN/X_INC  # SIZE OF EACH SPATIAL STEP

# CREATE VECTORS
TVECT = np.linspace(0, T_TOT, T_NNODE)  # TIME VECTOR
XVECT = np.linspace(0, LEN, X_NNODE)  # SPACE VECTOR
TRAC_VECT = np.zeros(len(TVECT))  # EMPTY TRACTION VECTOR

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
DISP_BAR_POS_ARR = np.zeros((len(XVECT), len(TVECT)))  # STRESS ARRAY
VEL_BAR_POS_ARR = np.zeros((len(XVECT), len(TVECT)))  # VELOCITY ARRAY
SIG_BAR_POS_ARR = np.zeros((len(XVECT), len(TVECT)))  # STRESS ARRAY

DISP_BAR_NEG_ARR = np.zeros((len(XVECT), len(TVECT)))  # STRESS ARRAY
VEL_BAR_NEG_ARR = np.zeros((len(XVECT), len(TVECT)))  # VELOCITY ARRAY
SIG_BAR_NEG_ARR = np.zeros((len(XVECT), len(TVECT)))  # STRESS ARRAY

for i in xrange(len(TBAR_VECT)):
    # CALCULATIONS FOR WAVE TRAVELING IN -X (<-) ARE PERFORMED STARTING AT
    # THE END OF THE VECTOR
    REV_INC = -1
    for j in xrange(len(XBAR_VECT)):
        ETA_POS_BAR = TBAR_VECT[i] - XBAR_VECT[j]
        ETA_NEG_BAR = TBAR_VECT[i] + XBAR_VECT[j+REV_INC] - LEN/X_REF
        # CALCULATIONS FOR WAVE TRAVELING IN POSITIVE X ->
        if ETA_POS_BAR >= 0.0 and ETA_POS_BAR <= 1.0:
            DISP_BAR_POS_ARR[j, i] = 0.5*(ETA_POS_BAR -
                                          np.sin(2*np.pi*ETA_POS_BAR) /
                                          (2*np.pi))
            VEL_BAR_POS_ARR[j, i] = 0.5*(1-np.cos(2*np.pi*ETA_POS_BAR))
            SIG_BAR_POS_ARR[j, i] = -0.5*(1-np.cos(2*np.pi*ETA_POS_BAR))
        if ETA_POS_BAR > 1.0:
            DISP_BAR_POS_ARR[j, i] = 0.5

        # CALCULATIONS FOR WAVE TRAVELING IN NEGATIVE X <-
        if ETA_NEG_BAR >= 0.0 and ETA_NEG_BAR <= 1.0:
            DISP_BAR_NEG_ARR[j+REV_INC, i] = -0.5 *\
                                            (ETA_NEG_BAR -
                                             np.sin(2*np.pi * ETA_NEG_BAR) /
                                             (2*np.pi))
            VEL_BAR_NEG_ARR[j+REV_INC, i] = -0.5*(1 -
                                                  np.cos(2*np.pi*ETA_NEG_BAR))
            SIG_BAR_NEG_ARR[j+REV_INC, i] = -0.5*(1 -
                                                  np.cos(2*np.pi*ETA_NEG_BAR))
        if ETA_NEG_BAR > 1.0:
            DISP_BAR_NEG_ARR[j+REV_INC, i] = -0.5

        REV_INC = REV_INC - 2

# IDENTIFY THE REAL DOMAIN, THEN GET THE VECTOR INDEX OF THE REALY RIGHT BDRY
IDX_REAL = np.where(XVECT == LEN_REAL)
IDX_REAL = IDX_REAL[0][0]

# SUM UP INFLUENCE OF BOTH WAVES OVER THE REAL DOMAIN
DISP_ARR = DISP_BAR_POS_ARR[0:IDX_REAL, :] +\
               DISP_BAR_NEG_ARR[0:IDX_REAL, :]
VEL_ARR = VEL_BAR_POS_ARR[0:IDX_REAL, :] +\
              VEL_BAR_NEG_ARR[0:IDX_REAL, :]
SIG_ARR = SIG_BAR_POS_ARR[0:IDX_REAL, :] +\
              SIG_BAR_NEG_ARR[0:IDX_REAL, :]

if DIM == 1:  # DIMENSIONAL FORM
    DISP_ARR = DISP_ARR * DISP_REF
    VEL_ARR = VEL_ARR * VEL_REF
    SIG_ARR = SIG_ARR * SIG_REF

if DIM == 0:  # DIMENSIONLESS FORM
    TVECT = TBAR_VECT
    XVECT = XBAR_VECT

# BUILD ARRAYS FOR PLOTTING
TIME_PLOT = np.tile(XVECT[0:IDX_REAL], (T_NNODE, 1)).T
X_PLOT = np.tile(TVECT, (IDX_REAL+1, 1))  # POINTS IN SPACE TO BE PLOTTED

TSTART = 0
TSTOP = len(TVECT)
TINC = len(TVECT)/8
LBL_TALL = [None]*T_NNODE
for i in xrange(T_NNODE):
    LBL_TALL[i] = "Time: {:,.3f}".format(TVECT[i])
LBL_T = LBL_TALL[TSTART:TSTOP:TINC]

XSTART = 0
XSTOP = IDX_REAL
XINC = IDX_REAL/5
LBL_XALL = [None]*(IDX_REAL + 1)
for i in xrange(IDX_REAL+1):
    LBL_XALL[i] = "X Dist: {:,.3f}".format(XVECT[i])
LBL_X = LBL_XALL[XSTART:XSTOP:XINC]

# # PLOT VELOCITY
FIG_VELT, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(TIME_PLOT[:, TSTART:TSTOP:TINC], VEL_ARR[:, TSTART:TSTOP:TINC],
         '-o')
AX1.grid(True)
AX1.set_xlabel("X Location")
AX1.set_ylabel("Velocity")
AX1.legend(LBL_T, frameon=1, framealpha=1, loc=0)
FIG_VELT_NAME = 'fig_velocity_T.pdf'
# # FIG_VELT.savefig(PATH + FIG_VELT_NAME)

FIG_VELX, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(X_PLOT[XSTART:XSTOP:XINC, :].T, VEL_ARR[XSTART:XSTOP:XINC, :].T,
         '-o')
AX1.grid(True)
AX1.set_xlabel("Time")
AX1.set_ylabel("Velocity")
AX1.legend(LBL_X, frameon=1, framealpha=1, loc=0)
FIG_VELX_NAME = 'fig_velocity_X.pdf'
# # FIG_VELX.savefig(PATH + FIG_VELX_NAME)


# PLOT DISPLACEMENT
FIG_DISPT, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(TIME_PLOT[:, TSTART:TSTOP:TINC], DISP_ARR[:, TSTART:TSTOP:TINC],
         '-o')
AX1.grid(True)
AX1.set_xlabel("X Location")
AX1.set_ylabel("Displacement")
AX1.legend(LBL_T, frameon=1, framealpha=1, loc=0)
FIG_DISPT_NAME = 'fig_disp_T.pdf'
# # FIG_DISPT.savefig(PATH + FIG_DISPT_NAME)

FIG_DISPX, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(X_PLOT[XSTART:XSTOP:XINC, :].T, DISP_ARR[XSTART:XSTOP:XINC, :].T,
         '-o')
AX1.grid(True)
AX1.set_xlabel("Time")
AX1.set_ylabel("Displacement")
AX1.legend(LBL_X, frameon=1, framealpha=1, loc=0)
FIG_VELX_NAME = 'fig_disp_X.pdf'
# # FIG_DISPX.savefig(PATH + FIG_DISPX_NAME)

# PLOT STRESS
FIG_STRST, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(TIME_PLOT[:, TSTART:TSTOP:TINC], SIG_ARR[:, TSTART:TSTOP:TINC],
         'o-')
AX1.grid(True)
AX1.set_xlabel("X Location")
AX1.set_ylabel("Stress")
AX1.legend(LBL_T, frameon=1, framealpha=1, loc=0)
FIG_STRST_NAME = 'fig_stress_T.pdf'
# # FIG_STRS.savefig(PATH + FIG_STRS_NAME)

FIG_STRSX, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(X_PLOT[XSTART:XSTOP:XINC, :].T, SIG_ARR[XSTART:XSTOP:XINC, :].T,
         '-o')
AX1.grid(True)
AX1.set_xlabel("Time")
AX1.set_ylabel("Stress")
AX1.legend(LBL_X, frameon=1, framealpha=1, loc=0)
FIG_STRSX_NAME = 'fig_stress_X.pdf'
# # FIG_STRSX.savefig(PATH + FIG_STRSX_NAME)

# PLOT TRACTION AB X=0
# FIG_TRACTION, AX1 = plt.subplots(figsize=(12, 8))
# AX1.plot(TVECT,TRAC_VECT)

if PLOT != 0:
    plt.show()
