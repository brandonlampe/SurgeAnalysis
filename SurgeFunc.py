""" 
    Module to be loaded for the transient surge analysis.
	Content includes:
		def valprint(string, value): function for printing numeric values with a label
		def matprint(string, value): function for printing arrays with a label
        def valvePercOpen
		class SS_DeltaH(): general function class for calcuation SS pressure loss
"""

import numpy as np
from scipy.interpolate import interp1d
from numba import jit

def valprint(string, value):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .6e}".format(string, value))

def matprint(string, value):
    """ Ensure uniform formatting of matrix value outputs. """
    print("{0}:".format(string))
    print(value)

@jit(["float32(float32, float32, float32, float32, float32, int32)"], nopython=True)
def FrictionFact_jit(flowRate, mu, spGrav, ID, roughness, units = 0):
    """
        All calculations will be performed in consistent SI units.  
        Therefore, if units are input in US terms (units=0), they will be converted to SI for calculation.

        Units for Input (must maintain a constant system of units):

                  |   DESCRIPTION      | SI   |
        units     |system of units     | !=0  |
        ------------------------------------------------
        flowRate  |volumetric flow rate|m3/sec|
        dynVisc   |dynamic viscosity   |Pa-s  |
        spGrav    |specific gravity    |none  |
        ID        |internal diameter   |m     |
        roughness |pipe roughness      |m     |

        Output:
        -------------------------------------------------
        f: darcy-weisbach friction factor (dimensionless)
    """
    rho = spGrav * 1000. # kg/m3
    area = np.pi * ID **2 / 4.# m2
    v = flowRate/area # velocity (m/sec)
    Re = rho * v * ID / mu #reynolds number
    Re_lim = 2000. #transition point between laminara and turbulent flow
    f_lim = Re_lim / 64. #maximum friction factor value occurs at Re_lim/64
    trans = 0.75 #transition range (region of smooth transition)
    Re_trans = Re_lim*(1+trans) # the transitional zone
    n = 1.0 #shape factor for lower side of transition (Re < 2000)
    slope = 10. #shape factor for upper side of transition (Re > 2000)
    
    if Re <= Re_lim:
        f = f_lim * np.sin((np.pi / 2. * (Re / Re_lim))**n)
        
    elif Re > Re_lim and Re <= Re_trans: 
        # transition between laminar and turbulent flow
        sqrt_f_trans = 10.0 # initial guess
        loop_max = 20 # max. number of loops
        inc = 1 # increment count
        res = 10.0 # initial value
        while res > 10**-8 and inc < loop_max:
            LHS = -2.0 * np.log10(roughness / (3.7 * ID) + 2.51 / (Re_trans * sqrt_f_trans))
            res = (LHS - 1./sqrt_f_trans)**2 #squared error
            sqrt_f_trans = 1./LHS
            inc = inc + 1
        f_trans = sqrt_f_trans**2
        Re_star = (Re - Re_lim) / Re_lim
        f = f_trans + (f_lim - f_trans)*(1. + slope * Re_star) * np.exp(-slope * Re_star)
                                       
    else:
        sqrt_f = 10.0 # initial guess
        loop_max = 20 # max. number of loops
        inc = 1 # increment count
        res = 10.0 # initial value

        while res > 10**-8 and inc < loop_max:
            LHS = -2.0 * np.log10(roughness / (3.7 * ID) + 2.51 / (Re * sqrt_f))
            res = (LHS - 1./sqrt_f)**2 #squared error
            sqrt_f = 1./LHS
            inc = inc + 1
        f = sqrt_f**2
        
    return f

@jit
def MethodChar01_jit(x, t, q0_arr, h0_arr, p0_arr, ca, cf, cb, idx, sg, ID, mu, rough, s, row, col, qp, hp, h, q, p, rho, area, v, Re, f, Tc, tc, y):
    ID = ID/1000. # convert from mm to m
    rough = rough/1000. #convert form mm to m
    area = np.pi * ID **2 / 4.# m2
    
    # define initial conditions
    for i in range(row):
        q[i,0] = q0_arr[i]
        h[i,0] = h0_arr[i]
        p[i,0] = p0_arr[i]
    
    # initial values at upstream reservoir
    # neg characteristic (m3)
    cn = q[1, 0] - ca[0]*h[1, 0] - cb - cf * q[1, 0] * np.abs(q[1, 0])
    qp[0] = cn + ca[0]*h[0, 0]
    
    # interior nodes
    for j in range(col - 1):
        for i in range(row - 2):
            i = i + 1 # skip first value (BCT)
            ip1 = i + 1
            im1 = i - 1
            ######################## Friction Factor Calc ########################
            f = FrictionFact_jit(q[ip1,j], mu[idx[i]], sg[idx[i]], ID, rough, units=0)
            cf = s * f/(2*(ID)*area) #(sec/m3)
            ########################################################################
            cn = q[ip1, j] - ca[idx[i]]*h[ip1,j] - cb - cf*q[ip1,j]*np.abs(q[ip1,j]) # negative characteristic
            cp = q[im1, j] + ca[idx[i]]*h[im1,j] - cb - cf*q[im1,j]*np.abs(q[im1,j]) # positive characteristic
            qp[i] = 0.5 * (cn + cp) # flow rate at future time
            hp[i] = (cp - qp[i])/ ca[idx[i]] # head at future time
            if i == 1: # calculations for upstream reservoir BCTs
                hp[0] = h[0,0] # Constant upstream reservoir head (m)
                qp[0] = cn + ca[idx[i]]*hp[0]
            if i == row - 2: # calculations for downstream valve BCTs
                #interpolate valve 
                interp = t[j]
                if interp >= Tc:
                    tau = 0
                else:
                    I = 0
                    while interp >= tc[I]:
                        I = I + 1
                    frac = (interp - tc[I-1]) / (tc[I] - tc[I-1])
                    tau = y[I-1] + frac * (y[I] - y[I-1])
        
                cv = (tau * q[row-1,0] * q[row-1,0]) / (ca[idx[i]] * h[row-1,0])
                qp[i+1] = 0.5 * (-cv + ((cv*cv) + 4 * cp * cv)**(0.5))
                hp[i+1] = (cp - qp[i])/ ca[idx[i]]
        q[:,j+1] = qp
        h[:,j+1] = hp
    return q, h

class SS_DeltaH():
    """ Doc String for General Class should go here"""
    
    def __init__(self, flowRate, dynVisc, spGrav, ID, roughness, length, degTheta = 90., units = 0):
        """
            General class for calculating the Darcy-Weisbach Friction Factor.
            
            All calculations will be performed in consisten SI units.  
            Therefore, if units are input in US terms (units=0), they will be converted to SI for calculation.
            
            Units for Input (must maintain a constant system of units):
            
                      |   DESCRIPTION      | US     | SI   |
            units     |system of units     | =0     | !=0  |
            ------------------------------------------------
            flowRate  |volumetric flow rate|gal/min |m3/hr |
            dynVisc   |dynamic viscosity   |cP      |Pa-s  |
            spGrav    |specific gravity    |none    |none  |
            ID        |internal diameter   |inch    |mm    |
            roughness |pipe roughness      |inch    |mm    |
            length    |length of flow      |feet    |m     |
            degTheta  |CCW Angle From Horiz|degree  |degree|
                      |in + flow direction |

            Note: to obtain results call the method ".Calc()" followed by desired parameter
            
            Example: Calculate friciton factor, Reynolds No., head loss, pressure loss, and velocity

                f = SS_DeltaH(q, mu, sg, d, rough, L, units=0).Calc().f
                Re = SS_DeltaH(q, mu, sg, d, rough, L, units=0).Calc().Re
                dh = SS_DeltaH(q, mu, sg, d, rough, L, units=0).Calc().dh
                dp = SS_DeltaH(q, mu, sg, d, rough, L, units=0).Calc().dp
                v = SS_DeltaH(q, mu, sg, d, rough, L, units=0).Calc().v

        """
        self.Q = flowRate
        self.mu = dynVisc
        self.sg = spGrav
        self.ID = ID
        self.roughness = roughness
        self.L = length
        self.units = units
        self.theta = np.pi / 180. * degTheta
        
        
    def ConvertIn(self):
        """
        Calculations will all be performed in SI units; therefore, this method converts from US to SI units
        """
        self.rho = self.sg * 1000 #kg/m3
        if self.units == 0:
            #convert from US to SI units
            self.L = self.L * 0.3048 #m
            self.Q = self.Q *3.7853/1000. / 60. # m3/sec
            self.ID = self.ID * 2.54 / 100. # m
            self.area = np.pi * self.ID **2 / 4.# m2
            self.v = self.Q / self.area # m/sec
            self.mu = self.mu * 10**-3 # Pa-s
            self.roughness = self.roughness * 2.54 / 100. # m
            textOut = "Input Was US Units"
        else:
            #convert to consistent SI units
            self.Q = self.Q / 3600. # m3/sec
            self.ID = self.ID / 1000. # m
            self.area = np.pi * self.ID **2 / 4.# m2
            self.v = self.Q / self.area # m/sec
            self.roughness = self.roughness / 1000. #m
            textOut = "Input Was SI Units"
        return self
    
    def Re(self):
        """
        Calculates Reynolds Number
         - this method may be called to check the Reynolds Number
        """
        self.Re = self.rho * self.v * self.ID / self.mu
        return self
        
    def FricFact(self):
        """
        Function to obtain the Darcy-Weisbach friction factor:
            Laminar Flow (Re <= 2000) -> Poiseuille's relationship
            Turbulent Flow (Re > 2000) -> evaluate the colebrrok-white equation 

        Input:
        -------------------------------------------------
        Re (required): reynolds number (dimensionless)
        d  (required): hydraulic diameter (length)
        epsilon (required): conduit roughness (length)

        Output:
        -------------------------------------------------
        f: darcy-weisbach friction factor (dimensionless)
        """
        if self.Re <= 2000.:
            f = self.Re / 64.
        else:
            sqrt_f = 10.0 # initial guess
            loop_max = 20 # max. number of loops
            inc = 1 # increment count
            res = 10.0 # initial value

            while res > 10**-8 and inc < loop_max:
                LHS = -2.0 * np.log10(self.roughness / (3.7 * self.ID) + 2.51 / (self.Re * sqrt_f))
                res = (LHS - 1./sqrt_f)**2 #squared error
                sqrt_f = 1./LHS
                inc = inc + 1
            f = sqrt_f**2

            if inc == loop_max:
                outText = "Friction Factor Did NOT Converge! Printed Value Should = 1.0"
                print(outText)
                print(-2.0 * np.log10(self.roughness / (3.7 * self.ID) + 2.51 / (self.Re * f**0.5)) * f**0.5)
                
        self.f = f
        return self
    
    def HeadLoss(self):
        """
        Calculates the change in piezometric head for steady-state flow
        accounts for head loss resulting from:
            -friction
            -change in elevation
        """
        g = 9.806 #m/s2
        self.dh_m = -self.f * (self.Q)**2 * self.L / (2 * self.ID * self.area**2*g) - np.sin(self.theta)*self.L
        self.dp_Pa = self.rho * g * self.dh_m
        return self
    
    def ConvertOut(self):
        """
        Converts units for return to user
        US Units:
            - head loss: feet
            - pressure loss: psi
            - velocity: ft/sec
        SI Units:
            - head loss: meter
            - pressure loss: MPa
            - velocity: m/sec
        """
        if self.units == 0:
            self.dh = self.dh_m / 0.3048 # feet
            self.dp = self.dp_Pa * 1.4504 * 10**-4 # psi
            self.v = self.v /0.3048 # ft/sec
        else:
            self.dh = self.dh_m # m
            self.dp = self.dp_Pa / 10**6 # MPa
            self.v = self.v # m/sec
        return self
                       
    def Calc(self):
        """
        This method must be called to obtain any needed parameters
        """
        self.ConvertIn()
        self.Re()
        self.FricFact()
        self.HeadLoss()
        self.ConvertOut()
        return self