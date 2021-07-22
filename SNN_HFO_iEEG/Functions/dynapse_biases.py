import numpy as np


def getTau(i):
    '''
    Compute DPI time constant tau in seconds for a given current value:
    c_p : DPI capacitance [F]
    u_t : Thermal voltage [V]
    i : Current, must be given in [A]
    tau : (c_p*u_t)/(k*i) [sec]
    '''
    c_p = 1.5*1e-12  # Farads
    u_t = 25*1e-3  # V
    k = 0.7

    return (c_p*u_t)/(k*i)


def getTauCurrent(tau, vector=False):
    '''
    Compute the current in Amperes necessary to get a desired time constant:
    tau : Time constant, must be given in [sec]
    c_p : DPI capacitance [F]
    u_t : Thermal voltage [V]
    i : (c_p*u_t)/(k*tau) [A]
    '''
    c_p = 1.5*1e-12  # Farads
    u_t = 25*1e-3  # V
    k = 0.7

    if vector == False:
        if tau == 0:
            return 2.390625e-05

    currents = (c_p*u_t)/(k*tau)

    if vector == True:
        zeros = np.where(tau == 0)
        currents[zeros] = 0

    return currents
