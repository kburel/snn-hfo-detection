from scipy.signal import butter, lfilter

# ========================================================================================
# Butterworth filter coefficients
# ========================================================================================


def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    This function is used to generate the coefficients for lowpass, highpass and bandpass
    filtering for Butterworth filters.

    :cutOff (int): either the lowpass or highpass cutoff frequency
    :lowcut, highcut (int): cutoff frequencies for the bandpass filter
    :fs (float): sampling frequency of the wideband signal
    :order (int): filter order 
    :return b, a (float): filtering coefficients that will be applied on the wideband signal
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# ========================================================================================
# Butterworth filters
# ========================================================================================


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    This function applies the filtering coefficients calculated above to the wideband signal.

    :data (array): vector with the amplitude values of the wideband signal 
    :cutOff (int): either the lowpass or highpass cutoff frequency
    :lowcut, highcut (int): cutoff frequencies for the bandpass filter
    :fs (float): sampling frequency of the wideband signal
    :order (int): filter order 
    :return y (array): vector with amplitude of the filtered signal
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
