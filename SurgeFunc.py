"""
    Module to be loaded for the transient surge analysis.
    Content includes:
        def valprint(string, value): function for printing numeric values
        def matprint(string, value): function for printing arrays

"""

import numpy as np
import numba
import CoolProp.CoolProp as CP
# from CoolProp.CoolProp import FluidsList


def valprint(string, value):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .6e}".format(string, value))


def matprint(string, value):
    """ Ensure uniform formatting of matrix value outputs. """
    print("{0}:".format(string))
    print(value)


# @numba.jit(["float32(float32, float32, float32, float32, float32, int32)"],
#            nopython=True)
@numba.jit(nopython=True)
def FrictionFact_jit(vel, visc, spgrav, dia_in, roughness, units):
    """
        All calculations will be performed in consistent SI units.
        Therefore, if units are input in US terms (units=0),
        they will be converted to SI for calculation.

        Units for Input (must maintain a constant system of units):

                  |   DESCRIPTION      | SI   | US
        units     |system of units     | !=0  | =0
        ------------------------------------------------
        VELOCITY  |fliud velocity      |m/sec | ft/sec
        dynVisc   |dynamic viscosity   |Pa-s  | cP
        spGrav    |specific gravity    |none  | none
        ID        |internal diameter   |m     | in
        roughness |pipe roughness      |m     | in

        Output:
        -------------------------------------------------
        f: darcy-weisbach friction factor (dimensionless)
    """

    if units == 0:  # CONVERT TO SI UNITS
        visc = visc * 10**-3  # Pa-s
        vel = vel * 0.3048  # m/sec
        dia_in = dia_in / 12 * 0.3048  # m
        roughness = roughness / 12 * 0.3048  # m

    rho = spgrav * 1000.  # kg/m3
    Re = rho * vel * dia_in / visc  # reynolds number
    Re_lim = 2000.  # transition point between laminara and turbulent flow
    f_lim = Re_lim / 64.  # maximum friction factor value occurs at Re_lim/64
    trans = 0.75  # transition range (region of smooth transition)
    Re_trans = Re_lim*(1+trans)  # the transitional zone
    n = 1.0  # shape factor for lower side of transition (Re < 2000)
    slope = 10.  # shape factor for upper side of transition (Re > 2000)

    if Re <= Re_lim:
        f = f_lim * np.sin((np.pi / 2. * (Re / Re_lim))**n)

    elif Re > Re_lim and Re <= Re_trans:
        # transition between laminar and turbulent flow
        sqrt_f_trans = 10.0  # initial guess
        loop_max = 20  # max. number of loops
        inc = 1  # increment count
        res = 10.0  # initial value
        while res > 10**-8 and inc < loop_max:
            LHS = -2.0 * np.log10(roughness / (3.7 * dia_in) +
                                  2.51 / (Re_trans * sqrt_f_trans))
            res = (LHS - 1./sqrt_f_trans)**2  # squared error
            sqrt_f_trans = 1./LHS
            inc = inc + 1
        f_trans = sqrt_f_trans**2
        Re_star = (Re - Re_lim) / Re_lim
        f = f_trans + ((f_lim - f_trans) *
                       (1. + slope * Re_star) * np.exp(-slope * Re_star))

    else:
        sqrt_f = 10.0  # initial guess
        loop_max = 20  # max. number of loops
        inc = 1  # increment count
        res = 10.0  # initial value

        while res > 10**-8 and inc < loop_max:
            LHS = -2.0 * np.log10(roughness / (3.7 * dia_in) +
                                  2.51 / (Re * sqrt_f))
            res = (LHS - 1./sqrt_f)**2  # squared error
            sqrt_f = 1./LHS
            inc = inc + 1
        f = sqrt_f**2

    return np.abs(f)


"""
FluidsList()
common fluids:
    Air
    Argon
    Benzene
    n-Butane
    IsoButane
    IsoButene
    Ethane
    Ethanol
    Ethylene
    Methane
    Nitrogen
    n-Propane
    n-Pentane
    Water
"""


def visc(temp_f, press_psig, fluid):
    """
    Function used to call CoolProp to obtain fluid viscosity (cP).
    Input Arguments:
        temp_f = temperature of fluid (F)
        press_psig = pressure of fluid (psig)
        fluid = fluid from CoolProp list of fluids,
            argument must be a string

    list of predefined fluids may be found at:
    http://www.coolprop.org/fluid_properties/PurePseudoPure.html

    list of properties may be found at:
    http://www.coolprop.org/v4/apidoc/CoolProp.html
    """

    temp = (temp_f - 32)*5/9 + 273.15  # kelvin
    p_atm = 101325.0  # pascal (atmospheric pressure at sea level)
    press = press_psig * (101325/14.6959) + p_atm  # pascal abs
    fluid = str(fluid)
    visc_pas = CP.PropsSI('viscosity', 'T', temp, 'P', press, fluid)  # Pa-s
    visc_cp = visc_pas * 10**3  # centipoise
    return visc_cp


def unit_weight(temp_f, press_psig, fluid):
    """
    temp_f = temperature of fluid (F)
    press_psig = pressure of fluid (psig)
    fluid = fluid from CoolProp list of fluids,
        argument must be a string

    list of predefined fluids may be found at:
    http://www.coolprop.org/fluid_properties/PurePseudoPure.html
    """

    temp = (temp_f - 32)*5/9 + 273.15  # kelvin
    p_atm = 101325.0  # pascal (atmospheric pressure at sea level)
    press = press_psig * (101325/14.6959) + p_atm  # pascal abs
    fluid = str(fluid)
    den_kgm3 = CP.PropsSI('D', 'T', temp, 'P', press, fluid)  # kg/m3
    unit_lbft3 = den_kgm3/1000 * 62.4  # lb/ft3
    return unit_lbft3


def speedOfSound(temp_f, press_psig, fluid):
    """
    temp_f = temperature of fluid (F)
    press_psig = pressure of fluid (psig)
    fluid = fluid from CoolProp list of fluids,
        argument must be a string

    list of predefined fluids may be found at:
    http://www.coolprop.org/fluid_properties/PurePseudoPure.html
    """
    temp = (temp_f - 32)*5/9 + 273.15  # kelvin
    p_atm = 101325.0  # pascal (atmospheric pressure at sea level)
    press = press_psig * (101325/14.6959) + p_atm  # pascal abs
    fluid = str(fluid)
    a_mps = CP.PropsSI('A', 'T', temp, 'P', press, fluid)  # m/sec
    a_fps = a_mps / 0.3048  # feet per sec
    return a_fps


def compressibility(temp_f, press_psig, fluid):
    """
    temp_f = temperature of fluid (F)
    press_psig = pressure of fluid (psig)
    fluid = fluid from CoolProp list of fluids,
        argument must be a string

    list of predefined fluids may be found at:
    http://www.coolprop.org/fluid_properties/PurePseudoPure.html
    """

    temp = (temp_f - 32)*5/9 + 273.15  # kelvin
    p_atm = 101325.0  # Pa (atmospheric pressure at sea level)
    press = press_psig * (101325/14.6959) + p_atm  # pascal abs
    fluid = str(fluid)
    comp_kpa = CP.PropsSI('isothermal_compressibility', 'T', temp,
                          'P', press, fluid)  # 1/Pa
    comp_psi = comp_kpa * (101325.0/14.6959)  # 1/psi
    return comp_psi


