def _convert_tau_and_current_to_each_other(tau_or_current):
    # DPI capacitance [F]
    c_p = 1.5*1e-12
    # Thermal voltage [V]
    u_t = 25*1e-3  # V
    k = 0.7
    if isinstance(tau_or_current, list):
        return [(c_p*u_t)/(k*x) for x in tau_or_current]
    return (c_p*u_t)/(k*tau_or_current)


def get_tau(i):
    '''
    Compute DPI time constant tau in seconds for a given current value:

    Parameters
    -----
    i : Current, must be given in [A]
    '''
    return _convert_tau_and_current_to_each_other(i)


def get_current(tau):
    '''
    Compute the current in Amperes necessary to get a desired time constant:

    Parameters
    -----
    tau : Time constant, must be given in [sec]
    '''
    return _convert_tau_and_current_to_each_other(tau)
