# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 05:03:33 2017

@author: HP-PC
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import time

#import sys
#print(sys.executable)

#open the annotations file
df = pd.read_csv('C:/Users/HP-PC/Documents/fran/Masters/Comp5703 Capstone Project/Physionet-sleepedf/csv files/SC4012EC-Hypnogram_annotations.csv')

#create new dataframe to store modified annotation information
annote_df = pd.DataFrame(columns=())

#deal with the 1st row - shorten to 30 min of wake at the start
INITAL_WAKE_STAGES = 60   #30 minutes of wake stages at 30sec each
onset_wake_duration = df.iloc[0][1]   #get the very first sleep stage onset length
#set the very beginning of the modified dataset for evaluation
onset_sleep_start = onset_wake_duration - INITAL_WAKE_STAGES*30   
#add in the first 30 minutes of Wake stage
for stage_time in range(onset_sleep_start, onset_wake_duration-1, 30):
    annote_df = annote_df.append(pd.DataFrame({'begin stage time':stage_time, 'sleep stage':'Sleep stage W'}, index=[0]), ignore_index=True)

#locate the first unknown '?' stage
last_row = df[df.values  == "Sleep stage ?"].index.item()

#deal with all other rows in between 1st and 2nd last
for i in range(1, last_row-1):
    num_epoch = df.iloc[i][1] / 30   #find how many epochs does this stage have
    for j in range(0, int(num_epoch)):     #for each epoch add a row into new_df
        annote_df = annote_df.append(pd.DataFrame({'begin stage time':df.iloc[i][0]+j*30, 'sleep stage':df.iloc[i][2]}, index=[0]), ignore_index=True)
    
#deal with the last row, 30 min of wake at the end    
FINAL_WAKE_STAGES = 60
term_wake_duration = df.iloc[last_row-1][1]   #duration of final Wake stage
term_wake_begin = int(df.iloc[last_row-1][0])    #beginning of final Wake stage
term_wake_end = int(term_wake_begin + FINAL_WAKE_STAGES*30)    #calculate the last Wake stage to be included
for stage_time in range(term_wake_begin, term_wake_end, 30):
    annote_df = annote_df.append(pd.DataFrame({'begin stage time':stage_time, 'sleep stage':'Sleep stage W'}, index=[0]), ignore_index=True)

#remove movement time and unknown    
annote_df = annote_df.drop(annote_df[annote_df.values == "Sleep stage ?"].index)
annote_df = annote_df.drop(annote_df[annote_df.values == "Movement time"].index)
annote_df = annote_df.reset_index(drop=True) 
    
#combine stages 3 and 4 into 3
annote_df["combined sleep stage"]= annote_df["sleep stage"]  #copy column sleep stage
row_index = annote_df['combined sleep stage'] == "Sleep stage 4"
annote_df.loc[row_index, 'combined sleep stage'] = "Sleep stage 3"

#open the eeg file
df = pd.read_csv('C:/Users/HP-PC/Documents/fran/Masters/Comp5703 Capstone Project/Physionet-sleepedf/csv files/SC4012E0-PSG_data.txt')

#create empty eeg dataframe
eeg_np = np.empty([len(annote_df),3000])

#start = time.time()
for i in range(len(annote_df)):
    start_time = annote_df.iloc[i][0]     #locate the start time in annotation
    df_st_idx = df[df["Time"]==start_time].index.item()   #locate corresponding start time in eeg
    #Change df["1"],df["2"],df["3"] depending on FpCz, PzOz or Horizontal(EMG) signals
    epoch = np.array(df["2"].iloc[df_st_idx:df_st_idx+3000]) #slice of column 1,2 ior 3 from starttime index +3000 centiseconds
    eeg_np[i] = np.reshape(epoch, (1,3000))
#end = time.time()
#print(end - start)

#convert eeg matrix to pandas dataframe
eeg_df = pd.DataFrame(eeg_np)
#concat both annotations and eeg dataframes together
annote_eeg_df = pd.concat([annote_df, eeg_df], axis=1)

#save to file
annote_eeg_df.to_csv('C:/Users/HP-PC/Documents/fran/Masters/Comp5703 Capstone Project/Physionet-sleepedf/csv files/formatted/SC4012_PzOz.csv')



########################################################################################
#### Combining all files into 1 large file
from os import walk

f = []
#get a list of filenames to loop through
for (dirpath, dirnames, filenames) in walk('C:/Users/HP-PC/Documents/fran/Masters/Comp5703 Capstone Project/Physionet-sleepedf/csv files/formatted'):
    f.extend(filenames)
    break

filepath = 'C:/Users/HP-PC/Documents/fran/Masters/Comp5703 Capstone Project/Physionet-sleepedf/csv files/formatted/'

#write the header row into the combined file
with open('C:/Users/HP-PC/Documents/fran/Masters/Comp5703 Capstone Project/Physionet-sleepedf/csv files/formatted/trainfile.csv','w') as dest:
    full_fn = filepath+f[0]
    with open(full_fn,'r') as src:
        for line in src:
            dest.write(line)
            
#write each file into destination file excluding header
with open('C:/Users/HP-PC/Documents/fran/Masters/Comp5703 Capstone Project/Physionet-sleepedf/csv files/formatted/trainfile.csv','a') as dest:
    for filename in f[1:]:
        full_fn = filepath+filename
        with open(full_fn,'r') as src:
            lines = src.readlines()
            for line in lines[1:]:
                dest.write(line)
            
            
            
            
            

 