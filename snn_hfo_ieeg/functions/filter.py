from scipy.signal import butter, lfilter

# ========================================================================================
# Butterworth filter coefficients
# ========================================================================================


def butter_bandpass(lowcut, highcut, sampling_frequency, order=5):
    '''
    This function is used to generate the coefficients for lowpass, highpass and bandpass
    filtering for Butterworth filters.

    :cutOff (int): either the lowpass or highpass cutoff frequency
    :lowcut, highcut (int): cutoff frequencies for the bandpass filter
    :sampling_frequency (float): sampling_frequency frequency of the wideband signal
    :order (int): filter order 
    :return b, a (float): filtering coefficients that will be applied on the wideband signal
    '''
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')


# ========================================================================================
# Butterworth filters
# ========================================================================================


def butter_bandpass_filter(data, lowcut, highcut, sampling_frequency, order=5):
    '''
    This function applies the filtering coefficients calculated above to the wideband signal.

    :data (array): vector with the amplitude values of the wideband signal 
    :cutOff (int): either the lowpass or highpass cutoff frequency
    :lowcut, highcut (int): cutoff frequencies for the bandpass filter
    :sampling (float): sampling frequency of the wideband signal
    :order (int): filter order 
    :return y (array): vector with amplitude of the filtered signal
    '''
    coefficient_b, coefficient_a = butter_bandpass(
        lowcut, highcut, sampling_frequency, order=order)
    return lfilter(coefficient_b, coefficient_a, data)
