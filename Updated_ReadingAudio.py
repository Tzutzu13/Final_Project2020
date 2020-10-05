# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:35:31 2020

@author: yaniv
"""

#read from Excel the syllables, their type and their scores.
#The function performs pre-processing to get the same spectrogram as the matlab output
# make sure to change line 27 to the wanted path
# make sure the E folder is not empty (meaning that the Hard disk is connected)
# Returns: reshaped data to wanted dimention, labels of each syllable

def Updated_ReadingAudio(img_rows=64, img_cols=64, path = r"C:\Users\yaniv\Desktop\classification app\goldStandarts_2020_united.xlsx" ):
    import scipy
    from scipy import signal
    from scipy.io import wavfile
    from pathlib import Path
    import numpy as np
    from scipy.misc import imresize
    import xlrd
    # from AudioPreProcessing import AudioPreProcessing
    from skimage import img_as_ubyte
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib import mlab
    # Open the gold standard xl
    xl_workbook = xlrd.open_workbook(path)
    xl_sheet = xl_workbook.sheet_by_index(0)
    num_rows = xl_sheet.nrows   # Number of rows    
    Image = [None] * num_rows
    TrueLabels = [None] * num_rows
    Data = [None] * num_rows
    Precentages = [None] * num_rows
    for row_idx in range(0,num_rows): # Iterate through rows
        current_row = xl_sheet.row_values(row_idx) 
        data_folder = current_row[0]
        file = "%s.wav" %(current_row[1]) #audiofile
        file_to_open = data_folder + file #audiofile
        Fs, samples = wavfile.read(file_to_open)
        times1 = int(round(current_row[2],4)*Fs) #taking relevant times of the syllable from the audio file
        times2 = int(round(current_row[3],4)*Fs) #taking relevant times of the syllable from the audio file
        Signal = samples[times1:times2] #taking relevant times of the syllable from the audio file
        spectrogram,frequencies,times = mlab.specgram(Signal, NFFT=256, Fs=Fs, noverlap=120) # same spectrogram as the matlab
        # dBS =  20*np.log10(spectrogram) # convert to dB
        # plt.figure()
        # plt.pcolormesh(dBS)
        # spectrogram(sig,256,120,256,Fs,'MinThreshold',-110,'yaxis')
        # data = PreProcessedAudio #Dummy data. Just for testing
        width = img_rows # define the new spectrogram shape
        height = img_cols
        dim = (width, height)
        resized = cv2.resize(spectrogram, dim, interpolation = cv2.INTER_AREA)
            # plt.figure()
            # plt.pcolormesh(20*np.log10(resized))
            # plt.colorbar() 
        # img = resized/np.amax(resized)
        #     plt.figure()
        #     plt.pcolormesh(20*np.log10(img))
        #     plt.colorbar() 
        Data[row_idx] = resized
        TrueLabels[row_idx] = current_row[8]
    return(Data,TrueLabels)

# [Data,TrueLabels] = Updated_ReadingAudio()
# import numpy as np
# np.save('Data_united', Data)
# np.save('Labels_united', TrueLabels)
