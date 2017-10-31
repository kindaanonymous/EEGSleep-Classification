# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 05:03:33 2017

@author: HP-PC
"""

import numpy as np
from os import walk

#combining subject files into test and train files

f = []
#get a list of filenames to loop through
for (dirpath, dirnames, filenames) in walk('../Physionet-sleepedf/csv files/formatted'):
    f.extend(filenames)
    break

filepath = '../Physionet-sleepedf/csv files/formatted/'
filenames.sort()

print("f is", f)
print("filename list", filenames)
#loop through 20 times, creating a train and test file for each sleep subject

for sub_num in np.arange(0,len(filenames)/2):
 
    subject = str(int(sub_num))
    test_file_num1 = int(sub_num*2)
    test_file_num2 = int(sub_num*2+1)
    test_filenames = [filenames[test_file_num1], filenames[test_file_num2]]
    train_filenames = [name for name in filenames if name not in test_filenames]
    
    #TRAINFILES
    #write the header row into the combined file
    with open('../Physionet-sleepedf/csv files/formatted/consolidated files/trainfile'+subject+'.csv','w') as dest:
        full_fn = filepath+train_filenames[0]
        with open(full_fn,'r') as src:
            for line in src:
                dest.write(line)
            
    #write each file into destination file excluding header
    with open('../Physionet-sleepedf/csv files/formatted/consolidated files/trainfile'+subject+'.csv','a') as dest:
        for filename in train_filenames[1:]:
            full_fn = filepath+filename
            with open(full_fn,'r') as src:
                lines = src.readlines()
                for line in lines[1:]:
                    dest.write(line)
            
    #TESTFILES
    #write the header row into the combined file
    with open('../Physionet-sleepedf/csv files/formatted/test files/testfile'+subject+'.csv','w') as dest:
        full_fn = filepath+test_filenames[0]
        with open(full_fn,'r') as src:
            for line in src:
                dest.write(line)
            
    #write each file into destination file excluding header
    with open('../Physionet-sleepedf/csv files/formatted/test files/testfile'+subject+'.csv','a') as dest:
        for filename in test_filenames[1:]:
            full_fn = filepath+filename
            with open(full_fn,'r') as src:
                lines = src.readlines()
                for line in lines[1:]:
                    dest.write(line)
            
            
            

 