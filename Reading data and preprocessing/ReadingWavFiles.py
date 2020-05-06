# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:32:56 2019

@author: omrim
"""

def ReadingWavFiles(path):
    import os
    from scipy.io import wavfile
    import sys

    x = list(os.walk(path)) # put all files and folders in a list
    wavList = []
    for line in range(len(x)): # iterate every folder
        if x[line][2]: # [2] is the place in the list of the files only
            for i in x[line][2]:
                if i.lower().endswith('.wav'): 
                    fs, data = wavfile.read(x[line][0]+'\\'+ i) # read the fs and data of a wav file from the correct path
                    wavList.append(data)
    return wavList,fs

    
    
    
    
    
    