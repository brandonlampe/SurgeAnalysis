""" Method of Characteristics
-Derivation from Bruce Thomson class notes (UNM CE540)
-upstream (EOT) is always assumed to be the problem datum (elev=0)

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/Lampe/PyScripts/surgeAnalysis')
import SurgeFunc as sf

# import pdb

# RUNS ON PYTHON VERSION 2.7+
# print(sys.version_info)
np.set_printoptions(precision=4)

PLOT = 0
# INPUT DATA ###
TIME_CLOSE = 10  # SECONDS (VALVE CLOSURE DURATION)
TIME = 60  # SECONDS OF ANALYSIS
VEL0 = 15.0  # FEET/SEC (INITIAL FLUID VELOCITY)
THETA = np.pi/2  # ORIENTATION OF PIPE (RADIANS: 0=HORIZ, np.pi/2=VERT)

LENGTH = 3.5e3  # FEET (PIPE LENGTH)
DIA_IN = 24.0  # INCHES (INNER DIAMETER OF PIPE)
ROUGH = 0.06  # INCHES (ROUGHNESS OF LEACHING STRING (0.018 TO 0.06))
WALL_THICK = 0.75  # INCH (PIPE WALL THICKNESS)
MOD_PIPE = 29E6  # PSI (MODULUS OF ELASTICITY FOR PIPE)
POISSON = 0.27  # POISSION'S RATIO FOR STEEL

PRESS_US = 100 + 0.52*LENGTH  # PSI (PRESSURE AT EOT)
# CONSTANTS
GRAV = 32.2  # FT/SEC2

# CONDITIONS FOR FLUID PROPERTIES
FLUID = "Water"
# FLUID = "Brine"
FLUID_TEMP = 105.0  # DEG F
FLUID_PRES = PRESS_US  # PSIG
if FLUID == "Brine":
    SPG = sf.DBrine(FLUID_TEMP, FLUID_PRES)  # SPECIFIC GRAVITY OF FLUID
    FLUID_WEIGHT = SPG * 62.4  # LB/FT3
    FLUID_RHO = FLUID_WEIGHT / GRAV  # SLUG/FT3
else:
    FLUID_WEIGHT = sf.unit_weight(FLUID_TEMP, FLUID_PRES, FLUID)  # LB/FT3
    SPG = FLUID_WEIGHT / 62.4
    FLUID_RHO = FLUID_WEIGHT / GRAV  # SLUG/FT3

# PRODUCT PROPERTIES
PROD = "n-Propane"
PROD_TEMP = 105.0
PROD_PRES = PRESS_US  # PSIG
PROD_WEIGHT = sf.unit_weight(PROD_TEMP, PROD_PRES, PROD)  # LB/FT3
PROD_RHO = PROD_WEIGHT / GRAV  # SLUG/FT3

# DEFINE DISCRETIZATION
NELEM = 4

# CALCULATE WAVE SPEED IN SYSTEM
sf.valprint("FLUID GRAVITY", SPG)
FLUID_COMP = sf.CompBrine(FLUID_TEMP, FLUID_PRES)  # 1/PSI
sf.valprint("FLUID COMPRESSIBILITY", FLUID_COMP)

DEN_SLUG = 1.94 * SPG
QUOT = (144.0)/(FLUID_COMP*DEN_SLUG)
WAVE_SPEED_FLUID = np.sqrt(QUOT)  # FT/SEC
WAVE_SPEED_PROD = sf.speedOfSound(PROD_TEMP, PROD_PRES, PROD)  # FT/SEC
sf.valprint("SPEED OF SOUND IN FLUID ONLY", WAVE_SPEED_FLUID)
sf.valprint("SPEED OF SOUND IN PRODUCT ONLY", WAVE_SPEED_PROD)

NUM = 144.0/(FLUID_COMP * FLUID_RHO)  # FT2/SEC2
DEN = 1 + (DIA_IN/(MOD_PIPE * WALL_THICK * FLUID_COMP))*(1-POISSON/2.0)
WAVE_SPEED = np.sqrt(NUM/DEN)
sf.valprint("SPEED OF SOUND IN SYSTEM", WAVE_SPEED)
TIME_CRIT = LENGTH/WAVE_SPEED  # TIME FOR PRESSURE WAVE TO TRAVEL LENGTH
sf.valprint("CRIT TIME", TIME_CRIT)

CHECK = DIA_IN/(MOD_PIPE * WALL_THICK) * (1 - POISSON/2) * (1/FLUID_COMP)
sf.valprint("PIPE CHECK", CHECK)

# UPSTREAM BOUNDARY CONDITION
# FEET (ELEV + PRESSURE + VELOCITY HEADS = TOTAL HEAD AT UPSTREAM DATUM)
HEAD_US = 0.0 + PRESS_US/FLUID_WEIGHT*144.0 + VEL0**2/(2*GRAV)
sf.valprint("UPSTREAM BC (FT)", HEAD_US)

# END INPUT DATA ###
NNODE = NELEM + 1
DELX = LENGTH/NELEM  # FEET (NODE SPACING)
DELT = DELX/WAVE_SPEED  # SECOND (TIME STEP)
XGRID = np.linspace(start=0, stop=LENGTH, num=NNODE)  # SPATIAL DISC
TGRID = np.linspace(start=0, stop=TIME, num=TIME/DELT+1)  # TEMPORAL DISC
COURANT = WAVE_SPEED * DELT/DELX  # THE COURANT NUMBER <= 0.5 FOR STABILITY
sf.valprint("COURANT", COURANT)
sf.valprint("Delta Time", DELT)
sf.valprint("Delta X", DELX)
CHECK = WAVE_SPEED * DELT
sf.valprint("a * dt", CHECK)


# CALCULATIONS TO SET UP PROBLEM
DIA = DIA_IN/12.0  # FEET (DIAMETER)
AREA = np.pi * DIA**2/4.0  # FT2 (AREA OF PIPE FLOW)
RHOW = 1.94  # SLUGS/FT3 (DENSITY OF WATER)
RHO = FLUID_RHO  # SLUGS/FT3 (DENSITY OF FLUID)
GAMMA = RHO * GRAV  # LB/FT3
DEL_WEIGHT = (FLUID_WEIGHT - PROD_WEIGHT)  # LB/FT3 (DIFF IN FLUID WEIGHTS)

# CALCULATE FRICTION FACTOR
VISC = sf.visc(90, 2000, "Water")
# DIMENSIONLESS DARCY-WIESBACH FRICTION FACTOR
FRIC = sf.FrictionFact_jit(VEL0, VISC, SPG, DIA_IN, ROUGH, 0)
sf.valprint("FRICTION FACTOR", FRIC)

# INITIAL CONDITIONS (TOTAL HEAD EVERYWHERE IN SYSTEM)
HEAD_EL = np.sin(THETA)*XGRID  # ELEVATION HEAD
HEAD_V = VEL0**2/(2*GRAV)  # VELOCITY HEAD
HEAD_LS = FRIC*XGRID/(2*DIA*GRAV)*VEL0**2  # HEAD LOSS FROM FRICTION
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
FRIC_ARR = np.zeros((len(XGRID), len(TGRID)))
INT = np.zeros(len(TGRID))  # TRACKS INTERFACE DEPTH
int_pass = np.zeros(len(TGRID)) 

# SET INITIAL CONDITIONS
HEAD_ARR[:, 0] = HEAD0
VEL_ARR[:, 0] = VEL0

# SET BCTS
HEAD_ARR[0, :] = HEAD_US  # UPSTREAM HEAD
VEL_ARR[-1, :] = VEL_DS  # DOWNSTREAM VELOCITY
FRIC_ARR[:, 0] = FRIC

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

sf.valprint("DEL_WEIGHT", DEL_WEIGHT)
sf.valprint("PROD_RHO", PROD_RHO)

for j in xrange(TCOL):
    # MODIFY UPSTREAM BC TO ACCOUNT FOR PRODUCT IN STRING
    INT[j+1] = DELT * VEL_ARR[-1, j] + INT[j]
    DEL_HEAD = (DEL_WEIGHT * INT[j+1])/PROD_WEIGHT
    HEAD_ARR[0, j+1] = HEAD_US + DEL_HEAD  # INCREASE IN HEAD FROM INT MOVEMENT
    for i in xrange(XROW):  # INTERIOR NODE CALCULATIONS
        fric = sf.FrictionFact_jit(VEL_ARR[i, j], VISC, SPG, DIA_IN, ROUGH, 0)
        # fric = FRIC
        rterm = fric*DELX/(2*GRAV*DIA)

        cpos = HEAD_ARR[i, j] + BTERM*VEL_ARR[i, j]
        cneg = HEAD_ARR[i+2, j] - BTERM*VEL_ARR[i+2, j]
        bpos = BTERM + rterm*np.abs(VEL_ARR[i, j])
        bneg = BTERM + rterm*np.abs(VEL_ARR[i+2, j])
        HEAD_ARR[i+1, j+1] = (cpos*bneg + cneg*bpos)/(bpos+bneg)
        VEL_ARR[i+1, j+1] = (cpos - cneg)/(bpos + bneg)
        FRIC_ARR[i+1, j+1] = fric
        if i == 0:  # UPSTREAM BOUNDARY NODE - VELOCITY
            cneg = HEAD_ARR[i+1, j] - BTERM*VEL_ARR[i+1, j]
            bneg = BTERM + rterm*np.abs(VEL_ARR[i+1, j])
            VEL_ARR[i, j+1] = (HEAD_ARR[i, j+1] - cneg)/bneg
        if i == XROW-1:  # DOWNSTREAM BOUNDARY NODE - HEAD
            cpos = HEAD_ARR[i+1, j] + BTERM*VEL_ARR[i+1, j]
            bpos = BTERM + rterm*np.abs(VEL_ARR[i+1, j])
            HEAD_ARR[-1, j+1] = cpos - bpos*VEL_ARR[-1, j+1]

# CALCULATE PRESSURE
ROW, COL = HEAD_ARR.shape
PRESS = np.zeros((ROW, COL))
for i in xrange(COL):
    PRESS[:, i] = (HEAD_ARR[:, i] - HEAD_EL - VEL_ARR[:, i]**2/(2*GRAV))\
                 * GAMMA/144.0

IDX_TIME_4 = PRESS[-1, np.where(TGRID == 4.0)]  # SAVES PRESSURE AT T=4 SEC
# sf.matprint("PRESSURE AT T=4", IDX_TIME_4)

# SAVES TIME AT MAX P
IDX_MAX_PRESS = TGRID[np.where(PRESS[-1, :] == max(PRESS[-1, :]))]
# sf.matprint("TIME AT MAX P", IDX_MAX_PRESS)

# sf.matprint("FRIC ARRAY", FRIC_ARR)
sf.valprint("INTERFACE RISE (FT)", max(INT))

# BUILD ARRAYS FOR PLOTTING
time_plot = np.tile(TGRID, (NNODE, 1)).T
x_plot = np.tile(XGRID, (COL, 1)).T
PATH = '/Users/Lampe/Documents/PB/SurgeAnalysis/Results/'

START = 0
STOP = HEAD_ARR.shape[0]
INC = HEAD_ARR.shape[0]/2

LBL_ALL = [None]*ROW
for i in xrange(ROW):
    LBL_ALL[i] = "Depth: {:,.0f} ft".format(DEPTH[i])

LBL = LBL_ALL[START:STOP:INC]

FIG1, AX1 = plt.subplots(figsize=(12, 8))
AX1.plot(time_plot[:, START:STOP:INC], VEL_ARR.T[:, START:STOP:INC], 'o--')
AX1.grid(True)
AX1.set_xlabel("Time (sec)")
AX1.set_ylabel("Fluid Velocity (ft/sec)")
AX1.legend(LBL, frameon=1, framealpha=1, loc=0)
FIG1_NAME = 'fig_velocity.pdf'
# FIG1.savefig(PATH + FIG1_NAME)

FIG2, AX2 = plt.subplots(figsize=(12, 8))
AX2.plot(time_plot[:, START:STOP:INC], PRESS.T[:, START:STOP:INC], 'o--')
AX2.grid(True)
AX2.set_xlabel("Time (sec)")
AX2.set_ylabel("Pressure (psi)")
AX2.legend(LBL, frameon=1, framealpha=1, loc=1)
FIG2_NAME = 'fig_head.pdf'
# FIG2.savefig(PATH + FIG2_NAME)

if PLOT != 0:
    plt.show()
