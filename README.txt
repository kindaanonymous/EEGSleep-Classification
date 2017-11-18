Implemented model for DeepSleepNet by Francine Khoo

There are a total of 6 files and 2 folders including this README

FILES
1. README.txt - explains how to run the DeepSleepNet model

2. capstone_reformat_eeg.py
This script combines the annotated and EEG files from their respective .csv files.  It also truncates unnecessary wake stages, leaving on the first 30 mins and last 30 mins after the sleep stages.  This requires the annotated files and EEG files for in .csv format.  Run once for each sleep subject's night, after changing the annotation, EEG file and combined file output filenames.  Note line 69, change dataframe slice accordingly for the Fpz-Cz, Pz-Oz or EOG(Horizontal) channel data. This script can be run using the files in the folder "raw EEG and anontation extractions".

3. trainfile_testfile_creation.py
This script combines all 20 sleep subjects formatted EEG and annotations files (40 files in total) into 20 train and test files. Each train file contains 19 sleep subjects data (titled trainfile<subj_num>.csv) and the corresponding test file will contain the 1 remaining sleep subjects data (titled testfile<subj_num>.csv.   The combined files are used by the capstone_cnn_loop.py script.  This script can be run using the files in the folder "formatted EEG and annotations files", which were created by the capstone_reformat_eeg.py script.

4. capstone_cnn_loop.py
This script runs stage 1 of the DeepSleepNet implementation. It takes in the trainfile<subj_num>.csv and testfile<subj_num>.csv based on the subject number in the for loop at the very beginning.  The outputs are the saved model and the classification report and confusion matrix for the final iteration.  The script prints train and test accuracies to screen every 1000 iterations. Note that although it is written as a loop, the loop itself is not used, as the initial graph built by the first loop is not re-initialised each time the loop is run, and causes inaccuracies in the saved model for all the following loop iterations.  The following script capstone_lstm_loop.py is not able to correctly utilise the saved model if it is created via the loop.  Hence for subject 0, run by setting the loop to np.arange(0,1), and for subject 1, by setting to np.arange(1,2), till the final subject 19 as np.arange(19,20). Setting the loop to subject 0, will use subject 0 as test and subjects 1-19 as train. 

5. capstone_lstm_loop.py
This script runs stage 2 of the DeepSleepNet implementation.  It takes in the directory where all the formatted EEG and anotations files are stored (output of capstone_reformat_eeg.py), and uses the selected sleep subject in the loop as the test subject and the remaining 19 subjects data for training.  It also needs to take in the saved model from capstone_cnn_loop.py (line 579), as well as the global list of variables from capstone_cnn_loop.py, so that these variables are not reinitialised and their saved values lost.  This list can be found in the text file cnn_gvar_list_edited.txt. The outputs for the script is the classification report and confusion matrix for the final iteration.  The script prints to screen the training accuracy after each 125 samples (5 batches of 25), and prints to screen the test classification report and confusion matrix after each iteration (1 iteration being the full training sample of all 19 train subjects).  20 iterations are usually run, however the model converges after the 14th iteration and may overfit by the 20th iteration. The screen standard output is saved to identify the best performing iteration.  Same as capstone_cnn_loop.py, the 1st loop creates the graph and does not re-initialise it at the beginning of each loop, hence the loop is not used, but subjects are trained 1 by 1. 

6. cnn_gvar_list_edited.txt
This file contains the list of global variables from the capstone_cnn_loop.py tensorflow graph.  This list of variables must not be initialised to random values in the capstone_lstm_loop.py script, as they contain the saved weights from the capstone_cnn_loop.py graph.  

FOLDERS
1. raw EEG and anotation extractions

This folder contains 20 sleep subjects 
	- 2 nights each of hypnogram annotations
	- 2 nights each of PSG data (contains all polysomnogram signals including EEG)
	- Note sleep subject 13's 2nd night of data is missing, hence there are 78 files in total.
	- the file titles are arranged as follows, eg SC4001EC
		SC = Sleep Casette
		4001 = 4ssn, where ss = sleep subject from 00-19, n = night number 1 or 2
		EC = C identifies the sleep technician scoring the hypnogram
					

2. formatted EEG and annotations files

This folder contains the formatted files needed for running the CNN and LSTM scripts.  The files are the output of running capstone_reformat_eeg.py.  There are a total of 40 files, 1 file for each night of the 20 sleep subjects.  Subject 13's night 2 has been duplicated from night 1.
