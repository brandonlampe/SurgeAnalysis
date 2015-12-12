""" 
    Module to be loaded for the transient surge analysis.
	Content includes:
		def valprint(string, value): function for printing numeric values with a label
		def matprint(string, value): function for printing arrays with a label
		class SS_DeltaH(): general function class for calcuation SS pressure loss
"""

import numpy as np

def valprint(string, value):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .6e}".format(string, value))

def matprint(string, value):
    """ Ensure uniform formatting of matrix value outputs. """
    print("{0}:".format(string))
    print(value)

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