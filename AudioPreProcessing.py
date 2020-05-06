# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 18:34:26 2019

@author: renan
"""

#pre-processing to the recording: remove DC and LPF

def AudioPreProcessing(sample_rate, samples):
    import scipy
    from scipy.signal import butter, lfilter
    
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    #parameter for filter
    fs=sample_rate
    data=samples
    order = 6
    cutoff = 120*10**3 # desired cutoff frequency of the filter, Hz
    
    # Get the filter coefficients
    b, a = butter_lowpass(cutoff, fs, order)
    y = butter_lowpass_filter(data, cutoff, fs, order)
    
    #remove DC
    y-=scipy.mean(y)
    
    return(y)
   
