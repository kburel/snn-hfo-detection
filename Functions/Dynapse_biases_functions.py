#from brian2 import *
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
    return (C_p*U_t)/(k*tau)

def get_mean_std_currents(mean_tau, std_tau):

    '''
    Take the mean and standard deviation values for the time constants that are required on the network 
    and give back the mean and standard deviation that need to be set in amperes. This when the DPI neuron
    and synapse equations are used.
    :param mean_tau (float): mean time constant [sec]
    :param std_tau (float):  starndard deviation [sec]
    '''

    mean_current = getTauCurrent(mean_tau * 1e-3)
    max_current = getTauCurrent((mean_tau + std_tau) * 1e-3)
    min_current = getTauCurrent((mean_tau - std_tau) * 1e-3)
    std = (np.abs(mean_current - max_current) + np.abs(mean_current - min_current))/2

    return mean_current, std, max_current, min_current

def updateCurrent(index_coarse,fine):

    """
    Convert caer bias I_TAU into current value:
    index_coarse: coarse value in range(8)
    fine: fine value in range(255)
    """   
    coarse = ['24u', '3.2u', '0.4u', '50n', '6.5n', '820p', '105p', '15p']
    unit = coarse[index_coarse]
    maxCurrent = float(unit.split(unit[-1])[0])
    current = fine * maxCurrent / 256
    if(unit[-1] == 'u'):
        multiplier = 1e-6
    if(unit[-1] == 'p'):
        multiplier = 1e-12
    if(unit[-1] == 'n'):
        multiplier = 1e-9
    current_final = current * multiplier
    
    return current_final






