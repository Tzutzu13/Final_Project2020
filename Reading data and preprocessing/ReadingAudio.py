  # -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:58:07 2019

@author: Shachar
"""

#read from Excel the syllables, their type and their scores.
#The function performs pre-processing: remove DC, padding zeros, decimation and normalization.
# Returns: image, score, and true value
#Pay attention to line 27 and 36 (path of file)



def ReadingAudio():
    import scipy
    from scipy import signal
    from scipy.io import wavfile
    from pathlib import Path
    import numpy as np
    from scipy.misc import imresize
    import xlrd
    from AudioPreProcessing import AudioPreProcessing
    from skimage import img_as_ubyte
    
    # Open the gold standard xl
    xl_workbook = xlrd.open_workbook(r"C:\Users\yaniv\Desktop\project2019\May\gold standars.xlsx")
    xl_sheet = xl_workbook.sheet_by_index(0)
    
    num_rows = xl_sheet.nrows   # Number of rows
    
    Rec_path = 'E:/Recordings/Adult/2017' #yaniv
    
    Image = [None] * num_rows
    TrueLabels = [None] * num_rows
    Data = [None] * num_rows
    Precentages = [None] * num_rows
    for row_idx in range(1,num_rows): # Iterate through rows
        current_row = xl_sheet.row_values(row_idx)
        data_folder = Path(Rec_path + "/%s relevant/%s/ch1" %(current_row[2], current_row[1]))
        file = "%s.wav" %(current_row[3])
        file_to_open = data_folder / file
        sample_rate, samples = wavfile.read(file_to_open)
        PreProcessedAudio=AudioPreProcessing(sample_rate, samples)
        frequencies, times, spectrogram = signal.spectrogram(PreProcessedAudio, sample_rate)
        list_times=list( times)
        first_num = times[times>=current_row[4]][0]
        first_ind = list_times.index(first_num)
        sec_num = times[times>=current_row[5]][0]
        sect_ind = list_times.index(sec_num)
        x=spectrogram[:,first_ind:sect_ind]
        
         # DC removal
        x-=scipy.mean(x)
         
         # zero padding to size 200
        size=400
        c,r=np.shape(x)
        padr1=int((size-r)/2)
        padr2=size-r-padr1
        padc1=int((size-c)/2)
        padc2=size-c-padc1
        Padded=np.pad(x, [(padc1, padc2), (padr1, padr2)], mode='constant')
        
         # Decimation to a size of (32,32)
        DecIm=imresize(Padded, (32,32), interp='bilinear', mode=None)
        # from PIL import Image

        # DecIm = np.array(Image.fromarray(Padded).resize([32,32],resample = Image.BILINEAR)) 

         # normalization
        img=DecIm/np.amax(DecIm)
        # Data[row_idx] = img

        Data[row_idx] = img_as_ubyte(img)
        TrueLabels[row_idx] = current_row[6]
        Precentages[row_idx] = current_row[7]
        Image[row_idx] = x
        
    return(Data,TrueLabels,Precentages,Image)
