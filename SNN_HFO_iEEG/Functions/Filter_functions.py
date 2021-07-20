import numpy as np
from scipy.signal import butter, lfilter,  filtfilt

#========================================================================================
# Butterworth filter coefficients
#========================================================================================   
'''
These functions are used to generate the coefficients for lowpass, highpass and bandpass
filtering for Butterworth filters.

:cutOff (int): either the lowpass or highpass cutoff frequency
:lowcut, highcut (int): cutoff frequencies for the bandpass filter
:fs (float): sampling frequency of the wideband signal
:order (int): filter order 
:return b, a (float): filtering coefficients that will be applied on the wideband signal

'''

def butter_lowpass(cutOff, fs, order=5):
    normalCutoff = 2 * cutOff / fs
    b, a = butter(order, normalCutoff)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#========================================================================================
# Butterworth filters
#========================================================================================   
'''
These functions apply the filtering coefficients calculated above to the wideband signal.

:data (array): vector with the amplitude values of the wideband signal 
:cutOff (int): either the lowpass or highpass cutoff frequency
:lowcut, highcut (int): cutoff frequencies for the bandpass filter
:fs (float): sampling frequency of the wideband signal
:order (int): filter order 
:return y (array): vector with amplitude of the filtered signal

'''
def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

