import numpy as np

def getTau(I):

    '''
    Compute DPI time constant tau in seconds for a given current value:
    C_p : DPI capacitance [F]
    U_t : Thermal voltage [V]
    I : Current, must be given in [A]
    tau : (C_p*U_t)/(k*I) [sec]
    '''
    C_p = 1.5*1e-12 #Farads 
    U_t = 25*1e-3 # V
    k = 0.7

    return (C_p*U_t)/(k*I)

def getTauCurrent(tau, vector = False):

    '''
    Compute the current in Amperes necessary to get a desired time constant:
    tau : Time constant, must be given in [sec]
    C_p : DPI capacitance [F]
    U_t : Thermal voltage [V]
    I : (C_p*U_t)/(k*tau) [A]
    '''
    C_p = 1.5*1e-12 #Farads 
    U_t = 25*1e-3 # V
    k = 0.7
    
    if vector == False:
        if tau == 0:
            return 2.390625e-05
     
    currents = (C_p*U_t)/(k*tau)
    
    if vector == True:
        zeros = np.where(tau == 0)
        currents[zeros] = 0
        
    return currents
