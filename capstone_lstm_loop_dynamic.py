# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 10:40:34 2017

@author: HP-PC
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
#from imblearn.over_sampling import SMOTE 
from collections import Counter
from tensorflow.contrib import rnn
from sklearn.preprocessing import OneHotEncoder
from os import walk



for sub_num in np.arange(12,13): 

    print("subject number", sub_num)
    subject = str(sub_num)


    #network parameters
    # Convolutional Layer 1s
    filter_size_1s = 50
    num_filters_1s = 64
    #stride_1s = tf.constant(6, dtype=tf.int16)
    stride_1s = 6
    # Convolutional Layer 2s , 3s , 4s
    filter_size_s = 8
    num_filters_s = 128
    #stride_s = tf.constant(1, dtype=tf.int16)
    stride_s = 1
    # Convolutional Layer 1l
    filter_size_1l = 250
    num_filters_1l = 64
    stride_1l = 12  #officially 100/2=50, but too small for later on.  Try 100/4 instead
    #stride_1l = tf.constant(50, dtype=tf.int16)
    # Convolutional Layer 2l , 3l, 4l
    filter_size_l = 8
    num_filters_l = 128
    stride_l = 1
    #stride_l = tf.constant(1, dtype=tf.int16)
    # Max pool layer 1s
    pool_size_1s = 8
    pool_stride_1s = 8
    #pool_size_1s= tf.constant(8, dtype=tf.int16)
    #pool_stride_1s= tf.constant(8, dtype=tf.int16)
    # Max pool layer 2s
    pool_size_2s = 4
    pool_stride_2s = 4
    #pool_size_2s= tf.constant(4, dtype=tf.int16)
    #pool_stride_2s= tf.constant(4, dtype=tf.int16)
    # Max pool layer 1l
    pool_size_1l = 6
    pool_stride_1l = 6
    #pool_size_1l= tf.constant(4, dtype=tf.int16)
    #pool_stride_1l= tf.constant(4, dtype=tf.int16)

    # Max pool layer 2l
    pool_size_2l = 4
    pool_stride_2l = 4
    #pool_size_2l= tf.constant(2, dtype=tf.int16)
    #pool_stride_2l= tf.constant(2, dtype=tf.int16)
    
    #drop out
    #dropout_prob = 0.5
    dropout_prob= tf.constant(0.5, dtype=tf.float32)
    dropout_prob_ontest= tf.constant(1.0, dtype=tf.float32)
    
    # Fully-connected layer.
    fc_size = 1024         # Number of neurons in fully-connected layer.
    #fc_size= tf.constant(1024, dtype=tf.int32)
    
    # 1d conv will use 1 channel
    num_channels = 1
    #num_channels= tf.constant(1, dtype=tf.int16)
    # classes are Wake, stage1, stage2, stage3, REM
    num_classes = 5   
    # 30s epoch at 100hz 
    stage_length = 3000
    #stage_length= tf.constant(3000, dtype=tf.int16)
        
    #LSTM parameters
    lstm_features = 3000 # EEG stage(img shape: 1*3000)
    sequence_length = 25 # timesteps
    num_hidden = 512 # hidden layer num of features
    num_iterations = 200
    batch_size = 1



    #initialise placeholders
    lr = tf.placeholder(tf.float32)
    # test flag for batch norm
    phase_test= tf.placeholder(tf.bool, name='phase_test')
    iteration= tf.placeholder(tf.int32, name='iteration')

    x= tf.placeholder(tf.float32, shape=[None, stage_length], name='x')
    x_stage = tf.reshape(x, [-1, stage_length, num_channels])   #batch, in_width, channels

    y_true= tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    #CNN only, LSTM Weights are initialised after restoring CNN model
    #Weights and Biases
    W_1s = tf.get_variable('W_1s', [50, 1, 64])  
    B_1s = tf.get_variable('B_1s', [64])
    W_2s = tf.get_variable('W_2s', [8, 64, 128])  
    B_2s = tf.get_variable('B_2s', [128])
    W_3s = tf.get_variable('W_3s', [8, 128, 128]) 
    B_3s = tf.get_variable('B_3s', [128])
    W_4s = tf.get_variable('W_4s', [8, 128, 128]) 
    B_4s = tf.get_variable('B_4s', [128])
    
    W_1l = tf.get_variable('W_1l', [400, 1, 64])
    B_1l = tf.get_variable('B_1l', [64])
    W_2l = tf.get_variable('W_2l', [6, 64, 128])  
    B_2l = tf.get_variable('B_2l', [128])
    W_3l = tf.get_variable('W_3l', [6, 128, 128]) 
    B_3l = tf.get_variable('B_3l', [128])
    W_4l = tf.get_variable('W_4l', [6, 128, 128]) 
    B_4l = tf.get_variable('B_4l', [128])

    BN_beta_1s = tf.get_variable('BN_beta_1s', shape=[num_filters_1s])
    BN_gamma_1s = tf.get_variable('BN_gamma_1s', shape=[num_filters_1s])
    BN_beta_2s = tf.get_variable('BN_beta_2s', shape=[num_filters_s])
    BN_gamma_2s = tf.get_variable('BN_gamma_2s', shape=[num_filters_s])
    BN_beta_3s = tf.get_variable('BN_beta_3s', shape=[num_filters_s])
    BN_gamma_3s = tf.get_variable('BN_gamma_3s', shape=[num_filters_s])
    BN_beta_4s = tf.get_variable('BN_beta_4s', shape=[num_filters_s])
    BN_gamma_4s = tf.get_variable('BN_gamma_4s', shape=[num_filters_s])

    BN_beta_1l = tf.get_variable('BN_beta_1l', shape=[num_filters_1l])
    BN_gamma_1l = tf.get_variable('BN_gamma_1l', shape=[num_filters_1l])
    BN_beta_2l = tf.get_variable('BN_beta_2l', shape=[num_filters_l])
    BN_gamma_2l = tf.get_variable('BN_gamma_2l', shape=[num_filters_l])
    BN_beta_3l = tf.get_variable('BN_beta_3l', shape=[num_filters_l])
    BN_gamma_3l = tf.get_variable('BN_gamma_3l', shape=[num_filters_l])
    BN_beta_4l = tf.get_variable('BN_beta_4l', shape=[num_filters_l])
    BN_gamma_4l = tf.get_variable('BN_gamma_4l',  shape=[num_filters_l])


    #regularisation 
    #we only want to add regularisation to the weights of the 1st 2 layers
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)   #L2 Lambda is 0.001
    weights_list = [W_1s, W_1l]

#####################################################################################################
#READ IN ALL THE FILES FOR TRAINING

    #get a list of filenames to loop through
    f = []
    for (dirpath, dirnames, filenames) in walk('../Physionet-sleepedf/csv files/formatted'):
        f.extend(filenames)
        break

    #sort filenames in alphabetical order
    #obtain the test and training file list
    filenames.sort()
    test_file_num1 = sub_num*2
    test_file_num2 = sub_num*2+1
    test_filenames = [filenames[test_file_num1], filenames[test_file_num2]]
    train_filenames = [name for name in filenames if name not in test_filenames]
    print("subject number", subject)
    print("list of train filenames for subject number", train_filenames)
    print("list of test filenames for subject number", test_filenames)    

    #loop through the training file list and put each subject's-night eeg into x_list and y_list
    x_list = []
    y_list = []
    
    for filename in train_filenames[0:]:
        df = pd.read_csv('../Physionet-sleepedf/csv files/formatted/' + filename)
        np_y = np.array(df.loc[:,"combined sleep stage"], dtype=str)
        np_x = np.array(df.loc[:,'0':'2999']  , dtype=float)  #3000 columns of the eeg

        #converting y indices for one-hot encoding later on
        np.place(np_y, np_y=="Sleep stage W", 0)
        np.place(np_y, np_y=="Sleep stage 1", 1)
        np.place(np_y, np_y=="Sleep stage 2", 2)
        np.place(np_y, np_y=="Sleep stage 3", 3)
        np.place(np_y, np_y=="Sleep stage R", 4)

        np_y = np_y.astype(int)

        np_y_reshaped = np_y.reshape(-1,1)
        enc = OneHotEncoder(sparse=False)
        y_onehot = enc.fit_transform(np_y_reshaped)
        
        x_list.append(np_x)
        y_list.append(y_onehot)
        #print("x_list length", len(x_list))
        #print("shape of x_list addition", x_list[-1].shape)
########################################################################################################

################################################################################
#READ IN DATA FOR TESTING
    #loop through the testing file list and put each subject's-night eeg into x_list and y_list
    x_list_test = []
    y_list_test = []
    for filename in test_filenames[0:]:
        df = pd.read_csv('../Physionet-sleepedf/csv files/formatted/' + filename)
        print("test filename", filename)
        np_y_test = np.array(df.loc[:,"combined sleep stage"], dtype=str)
        np_x_test = np.array(df.loc[:,'0':'2999']  , dtype=float)  #3000 columns of the eeg

        #converting y indices for one-hot encoding later on
        np.place(np_y_test, np_y_test=="Sleep stage W", 0)
        np.place(np_y_test, np_y_test=="Sleep stage 1", 1)
        np.place(np_y_test, np_y_test=="Sleep stage 2", 2)
        np.place(np_y_test, np_y_test=="Sleep stage 3", 3)
        np.place(np_y_test, np_y_test=="Sleep stage R", 4)

        np_y_test = np_y_test.astype(int)
    
        np_y_test_reshaped = np_y_test.reshape(-1,1)
        enc = OneHotEncoder(sparse=False)
        y_onehot_test = enc.fit_transform(np_y_test_reshaped)
        
        x_list_test.append(np_x_test)
        y_list_test.append(y_onehot_test)
        print("x_list_test length", len(x_list_test))
        #create a one hot encoding from the y resampled set
        #index_offset = np.arange(len(np_y_test))* num_classes
        #y_onehot_test = np.zeros((np_y_test.shape[0], num_classes))
        #y_onehot_test.flat[index_offset + np_y_test.ravel()] = 1

    print("x_list_test[0] shape", x_list_test[0].shape)
    print("x_list_test[1] shape", x_list_test[1].shape)
        
################################################################################
#various encapsulators defined

    #random batch only used in CNN
    def random_batch(batch_size, x, y):
        # Number of images (transfer-values) in the training-set.
        num_stages = len(x)
        # Create a random index.
        idx = np.random.choice(num_stages,
                               size=batch_size,
                               replace=False)

        # Use the random index to select random x and y-values.
        # We use the transfer-values instead of images as x-values.
        x_batch = x[idx]
        y_batch = y[idx]
    
        return x_batch, y_batch    

#sequential batch only used for testing, not for training
    def sequential_batch(batch_size, seqbatch_idx, x, y):
        num_stages = len(x)

        if (seqbatch_idx+batch_size> num_stages):
            xtra = num_stages-seqbatch_idx
            x_batch = x[seqbatch_idx:seqbatch_idx+xtra]
            y_batch = y[seqbatch_idx:seqbatch_idx+xtra]
            seqbatch_idx=0
                
        x_batch = x[seqbatch_idx:seqbatch_idx+batch_size]
        y_batch = y[seqbatch_idx:seqbatch_idx+batch_size]
       
        return x_batch, y_batch    


    def new_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    #from https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
    def batch_norm(x, phase_test, beta, gamma):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        #with tf.variable_scope('bn'):
            #beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
            #gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        beta = beta
        gamma = gamma
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        #print("batch_mean", batch_mean)
        with tf.name_scope(None):
            ema = tf.train.ExponentialMovingAverage(decay=0.999)
            ema_apply_op = ema.apply([batch_mean, batch_var])
            #print("ema_apply_op", ema_apply_op)
            
        def mean_var_with_update():
            #ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_test,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)),
                            mean_var_with_update)
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.00001) #episilon is 10^-5
        return normed

    def dropout_layer(input, phase_test, prob):    
        def donothing(input): return input
        output = tf.cond(phase_test, lambda: donothing(input), lambda: tf.nn.dropout(input, prob))
        return output

    #each convolutional layer performs a 1d conv, batch norm and relu activation
    def new_conv_layer(input, weights, bias, stride, phase_test, iteration, beta, gamma):    
        #shape = [filter_size, num_input_channels, num_filters]
        #weights = new_weights(shape)
        #biases = new_biases(num_filters)

        conv = tf.nn.conv1d(value=input, filters=weights, stride=stride, padding='VALID') + bias
        activation = tf.nn.relu(conv)
        bn = batch_norm(activation, phase_test, beta, gamma)    
        #activation = tf.nn.relu(bn)
        return bn

    
    def flatten_layer(input):
        flatten_shape = input.get_shape()
        print("flatten_shape:", flatten_shape)
        num_features = flatten_shape[1:3].num_elements()
        reshaped = tf.reshape(input, shape=[-1, num_features])
        reshaped_shape = reshaped.get_shape()
        print("reshaped_shape:", reshaped_shape)
        return reshaped, num_features


    def fc_layer(input, num_inputs, num_outputs):
        weights = new_weights(shape = [num_inputs, num_outputs])
        #print("weights", weights) 
        biases = new_biases(length = num_outputs)
        #print("bias", biases)
        fullyconnected = tf.matmul(input, weights) + biases
        return fullyconnected


    def shortcut_fc_layer(input, num_inputs, num_outputs, phase_test):
        weights = new_weights(shape = [num_inputs, num_outputs])
        #print("weights", weights) 
        biases = new_biases(length = num_outputs)
        #print("bias", biases)
        mat_mul = tf.matmul(input, weights) + biases
        activation = tf.nn.relu(mat_mul)
        bn = batch_norm(activation, phase_test, None, None)
        return bn


    def CNN_small(input, phase_test, iteration):
    
        #stride is 6 for 1st conv1d and 1 for rest.  Filter size 50 and 8 for conv2 onwards
        #input is (?, 3000, 1), output is (?, 492, 64)  because 3000-50/6
        input_shape = input.get_shape()
        print("input_shape:", input_shape)

        conv1s = new_conv_layer(input, W_1s, B_1s, stride_1s, phase_test, iteration, BN_beta_1s, BN_gamma_1s )
        conv1s_shape = conv1s.get_shape()
        print("conv1s_shape:", conv1s_shape)       
        conv1s = tf.expand_dims(conv1s, 0)   #adds extra dimension to make it 4d for max pool
        #max pool stride and size = 8.  Output size (?, 62, 64)
        conv1s_shape = conv1s.get_shape()
        print("conv1s_shape:", conv1s_shape)

        #note that pool size and stride is 3rd variable!
        max_pool1s = tf.nn.max_pool(conv1s, 
                                    ksize=[1, 1, pool_size_1s,  1],
                                    strides=[1, 1, pool_stride_1s, 1], 
                                    padding='SAME')
    
        max_pool1s_shape = max_pool1s.get_shape()
        print("max_pool1s_shape:", max_pool1s_shape) 
        max_pool1s = tf.squeeze(max_pool1s,0) #takes away added dimension
        max_pool1s_shape = max_pool1s.get_shape()
        print("max_pool1s_shape:", max_pool1s_shape)

        dropout_s = dropout_layer(max_pool1s, phase_test, dropout_prob)      
        dropout_shape = dropout_s.get_shape()
        print("dropout_shape:", dropout_shape)

        #filter size 8, stride size 1 for next 3 conv
        #input size(?, 62, 64) output size (?, 55 , 128)    
        conv2s = new_conv_layer(dropout_s, W_2s, B_2s, stride_s, phase_test, iteration,BN_beta_2s, BN_gamma_2s)     
        conv2s_shape = conv2s.get_shape()
        print("conv2s_shape:", conv2s_shape)
    
        #output size(?, 48 ,128) 
        conv3s = new_conv_layer(conv2s, W_3s, B_3s, stride_s, phase_test, iteration, BN_beta_3s, BN_gamma_3s)      
        conv3s_shape = conv3s.get_shape()
        print("conv3s_shape:", conv3s_shape)

        #output size(?, 41 ,128)
        conv4s = new_conv_layer(conv3s, W_4s, B_4s, stride_s, phase_test, iteration, BN_beta_4s, BN_gamma_4s)       
        conv4s_shape = conv4s.get_shape()
        print("conv4s_shape:", conv4s_shape)   
        conv4s = tf.expand_dims(conv4s, 0)   #adds extra dimension to make it 4d for max pool    
        conv4s_shape = conv4s.get_shape()
        print("conv4s_shape:", conv4s_shape)

        #max pool stride and size = 4.  Output size (?, 11, 128)
        max_pool2s = tf.nn.max_pool(conv4s, 
                                    ksize=[1, 1, pool_size_2s, 1],
                                    strides=[1, 1, pool_stride_2s, 1], 
                                    padding='SAME')

        mp_shape= max_pool2s.get_shape()
        print("max_pool2s shape:", mp_shape)
        max_pool2s = tf.squeeze(max_pool2s,0) #takes away added dimension
        mp_shape= max_pool2s.get_shape()
        print("max_pool2s shape:", mp_shape)
        
        return max_pool2s


    
    def CNN_large(input, phase_test, iteration):
    
        #Filter size 400 for conv1 and 6 for conv2 onwards. Stride is 50 for 1st conv1d and 1 for rest.  
        #Stride size of 100/2=50 too large, change to 100/4=25 instead
        #input is (?, 3000, 1), output is (?, 104, 64)  because 3000-400/25
        conv1l = new_conv_layer(input, W_1l, B_1l, stride_1l, phase_test, iteration, BN_beta_1l, BN_gamma_1l)
        conv1l_shape = conv1l.get_shape()
        print("conv1l_shape:", conv1l_shape)    
   
        #max_pool1l size and stride is 4
        #max_pool1l output is (?, 26, 64)
        conv1l = tf.expand_dims(conv1l, 0)   #adds extra dimension to make it 4d for max pool
        max_pool1l = tf.nn.max_pool(conv1l, 
                                    ksize=[1, 1, pool_size_1l, 1],
                                    strides=[1, 1, pool_stride_1l, 1], 
                                    padding='VALID')
        max_pool1l = tf.squeeze(max_pool1l,0) #takes away added dimension
        max_pool1l_shape = max_pool1l.get_shape()
        print("max_pool1l_shape:", max_pool1l_shape)    

        dropout_l = dropout_layer(max_pool1l, phase_test, dropout_prob)    

        #conv2l output is (?, 21, 128)    
        conv2l = new_conv_layer(dropout_l, W_2l, B_2l, stride_l, phase_test, iteration, BN_beta_2l, BN_gamma_2l)    
        conv2l_shape = conv2l.get_shape()
        print("conv2l_shape:", conv2l_shape)    

        #conv2l output is (?, 16, 128)
        conv3l = new_conv_layer(conv2l, W_3l, B_3l, stride_l, phase_test, iteration, BN_beta_3l, BN_gamma_3l)    
        conv3l_shape = conv3l.get_shape()
        print("conv3l_shape:", conv3l_shape)    
    
        #conv2l output is (?, 11, 128)      
        conv4l = new_conv_layer(conv3l, W_4l, B_4l, stride_l, phase_test, iteration, BN_beta_4l, BN_gamma_4l)
        conv4l_shape = conv4l.get_shape()
        print("conv4l_shape:", conv4l_shape)    
    
        #max_pool2l size and stride is 2
        #conv2l output is (?, 5, 128)      
        conv4l = tf.expand_dims(conv4l, 0)   #adds extra dimension to make it 4d for max pool
        max_pool2l = tf.nn.max_pool(conv4l, 
                                    ksize=[1, 1,  pool_size_2l, 1],
                                    strides=[1, 1, pool_stride_2l, 1], 
                                    padding='VALID')
        max_pool2l = tf.squeeze(max_pool2l,0) #takes away added dimension
        max_pool2l_shape = max_pool2l.get_shape()
        print("max_pool2l_shape:", max_pool2l_shape)    

        return max_pool2l

    #from https://stackoverflow.com/questions/37969065/tensorflow-best-way-to-save-state-in-rnns
    def get_state_variables(batch_size, cell):
        # For each layer, get the initial state and make a variable out of it
        # to enable updating its value.
        state_variables = []
        for state_c, state_h in cell.zero_state(batch_size, tf.float32):
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False),
                    tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)

    def get_state_update_op(state_variables, new_states):
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                               state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)

#    def get_state_reset_op(state_variables, cell, batch_size):
#        # Return an operation to set each variable in a list of LSTMStateTuples to zero
#        zero_states = cell.zero_state(batch_size, tf.float32)
#        return get_state_update_op(state_variables, zero_states)

    def get_state_reset_op(state_variables):
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variable in state_variables:
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(tf.zeros_like(state_variable[0])),
                               state_variable[1].assign(tf.zeros_like(state_variable[1]))])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)


    def bidirect_lstm(input, W, B, phase_test):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
        with tf.name_scope('lstm'):
            input_shape1 = input.get_shape()
            print("LSTM input_shape1:", input_shape1)    #(?, 25, 2048)
    
            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
            # input is now a list of 
            input = tf.unstack(input, sequence_length, 1)
    
            input_shape2 = len(input)
            print("LSTM input length:", input_shape2)    #25
            input_shape3 = input[0].get_shape()
            print("LSTM input shape of each matrix in list:", input_shape3)   #(?, 2048)

            # Define lstm cells with tensorflow
     
            # Forward direction cell
            stacked_rnn_fw = []
            for _ in range(2):
                stacked_rnn_fw.append(rnn.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True))
            mlstm_fw_cell = rnn.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
       
            # Backward direction cell
            stacked_rnn_bw = []
            for _ in range(2):
                stacked_rnn_bw.append(rnn.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True))
            mlstm_bw_cell = rnn.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
    
            #not used as is automatically set to 0 for each new num-step batch
            #init_fw = lstm_fw_cell.zero_state(sequence_length, tf.float32)
            #init_bw = lstm_bw_cell.zero_state(sequence_length, tf.float32)    


            def donothing(i): return i
            lstm_dropout =  tf.cond(phase_test, lambda: donothing(dropout_prob_ontest), lambda: donothing(dropout_prob))   
    
            mlstm_fw_cell = rnn.DropoutWrapper(mlstm_fw_cell, dropout_prob_ontest, lstm_dropout)  #input always 1 and output depends on train/test
            mlstm_fw_cell = rnn.DropoutWrapper(mlstm_bw_cell, dropout_prob_ontest, lstm_dropout)  
    
            #rnn.static_bidirectional_rnn
            #tf.nn.bidirectional_dynamic_rnn
            # Get lstm cell output
            outputs, _, _ = rnn.static_bidirectional_rnn(mlstm_fw_cell, mlstm_bw_cell, 
                                                         input, dtype=tf.float32)
                                                         #initial_state_fw=init_fw, 
                                                         #initial_state_bw=init_bw)
                                                         #dtype=tf.float32)
    

            #static_bidirectional_rnn returns:
            #A tuple (outputs, output_state_fw, output_state_bw) where: outputs 
            #is a length T list of outputs (one for each input), which are 
            #depth-concatenated forward and backward outputs. 
            #output_state_fw is the final state of the forward rnn. 
            #output_state_bw is the final state of the backward rnn.
        
            #output_shape1 = len(outputs)
            #print("output_shape1:", output_shape1)    # 25
            #output_shape1a = outputs[0].get_shape()
            #print("output_shape1a:", output_shape1a)    # (?,1024) 
        
            outputs = tf.stack(outputs)
            #output_shape2 = outputs.get_shape()
            #print("output_shape2:", output_shape2)    #(25, ?, 1024)
            outputs = tf.transpose(outputs, [1, 0, 2])
            #output_shape3 = outputs.get_shape()
            #print("output_shape3:", output_shape3)    #(?, 25, 1024)
            outputs = tf.reshape(outputs, [-1, 1024])
            #output_shape4 = outputs.get_shape()
            #print("output_shape4:", output_shape4)    #(?, 1024)
            outputs = tf.matmul(outputs, W) + B
            #output_shape5 = outputs.get_shape()
            #print("output_shape5:", output_shape5)    #[?,1024].[1024,1024]
    
        return outputs    



    #cnn layer
    max_pool2s = CNN_small(x_stage, phase_test, iteration)   #Output size (?, 11, 128)
    max_pool2l = CNN_large(x_stage, phase_test, iteration)   #Output size (?, 5, 128)

    with tf.name_scope('flatten_cnn'):
        flatten_s, num_features_s = flatten_layer(max_pool2s) #flattenshape:(?, 1408)
        flatten_l, num_features_l = flatten_layer(max_pool2l) #flattenshape:(?, 640)


###################################################################################
#RESTORE MODEL HERE BEFORE CREATING ANY LSTM VARIABLES
#create model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    session = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
#session.run(tf.global_variables_initializer()) 
    model_saver = tf.train.Saver()
    model_saver.restore(session, '../saved_models/capstone_cnn_fullmodel_relu_bn'+subject+'/-100')
    print("Model restored.")
##################################################################################


#LSTM WEIGHTS AND BIASES
    #weights for LSTM should be [hiddensize, num_classes], bias = num classes
    W_lstm1 = tf.Variable(tf.truncated_normal([2*512, 1024], stddev=0.1), name='W_lstm1')  
    B_lstm1 = tf.Variable(tf.constant(0.1, tf.float32, [1024]), name='B_lstm1')
    #W_lstm2 = tf.Variable(tf.truncated_normal([2*512, 1024], stddev=0.1), name='W_lstm1')  
    #B_lstm2 = tf.Variable(tf.constant(0.1, tf.float32, [1024]), name='B_lstm1')


    #continue adding the LSTM part of the model
    #concatenating the 2 cnn layers together
    with tf.name_scope("concat_cnn"):
        cnn_concat = tf.concat([flatten_s, flatten_l], 1)   #output size (?, 2048)

    #after dropout2 CNN concat split into LSTM and FC layers - 
    with tf.name_scope("dropout2"):
        dropout_2 = dropout_layer(cnn_concat, phase_test, dropout_prob)   

#FC layer
    with tf.name_scope("fully_connected_shortcut"):
        #reducing down to 1024 FC. output size (?, 1024)
        #this layer also does batch norm and relu activation
        fullyconnected_cnn = shortcut_fc_layer(dropout_2, 2048, 1024, phase_test)   

#LSTM layer
    dropout_2_shape = dropout_2.get_shape()
    print("initial lstm input: dropout2_shape:", dropout_2_shape)   #(?, 2048)

    concat_dim = dropout_2.get_shape()[-1].value   #get the num features after concatenation 2048
    print("concated dim:", concat_dim)

    #reshape to (batch_size, seq length 25, 2048)
    lstm_input = tf.reshape(dropout_2, shape=[batch_size, sequence_length, concat_dim]) #output size (?, 25, 2048)
    print("lstm input shape", lstm_input.shape)
    
    #with tf.name_scope("lstm"):
        #bidirect_lstm1 = bidirect_lstm(lstm_input, W_lstm1, B_lstm1, phase_test)  #output (?, 5)

#####################################################################################
###LSTM
#####################################################################################

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    # input is now a list of 
    #lstm_input = tf.unstack(lstm_input, sequence_length, 1)
    
    #input_shape2 = len(lstm_input)
    #print("LSTM input length:", input_shape2)    #25
    #input_shape3 = lstm_input[0].get_shape()
    #print("LSTM input shape of each matrix in list:", input_shape3)   #(?, 2048)

    # Define lstm cells with tensorflow
     
    # Forward direction cell
    stacked_rnn_fw = []
    for _ in range(2):
        stacked_rnn_fw.append(rnn.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True))
    mlstm_fw_cell = rnn.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
       
    # Backward direction cell
    stacked_rnn_bw = []
    for _ in range(2):
        stacked_rnn_bw.append(rnn.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True))
    mlstm_bw_cell = rnn.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
    

    # For each layer, get the initial state. states will be a tuple of LSTMStateTuples.
    fw_states = get_state_variables(batch_size, mlstm_fw_cell)
    bw_states = get_state_variables(batch_size, mlstm_bw_cell)    

    #this op gets called at the beginning of each new subject
    fw_reset_state_op = get_state_reset_op(fw_states)
    bw_reset_state_op = get_state_reset_op(bw_states)

    def donothing(i): return i
    lstm_dropout =  tf.cond(phase_test, lambda: donothing(dropout_prob_ontest), lambda: donothing(dropout_prob))   
    
    mlstm_fw_cell = rnn.DropoutWrapper(mlstm_fw_cell, dropout_prob_ontest, lstm_dropout)  #input always 1 and output depends on train/test
    mlstm_fw_cell = rnn.DropoutWrapper(mlstm_bw_cell, dropout_prob_ontest, lstm_dropout)  
        
    # Get lstm cell output
    #outputs, output_state_fw, output_state_bw = rnn.static_bidirectional_rnn(mlstm_fw_cell, mlstm_bw_cell, 
    #                                                                          lstm_input, dtype=tf.float32,
    #                                                                          initial_state_fw=fw_states, 
    #                                                                          initial_state_bw=bw_states)
    
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(mlstm_fw_cell, mlstm_bw_cell, 
                                                                              lstm_input, 
                                                                              initial_state_fw=fw_states, 
                                                                              initial_state_bw=bw_states,
                                                                              sequence_length=[sequence_length],
                                                                              time_major=False)




    fw_update_op = get_state_update_op(fw_states, output_states[0])    
    bw_update_op = get_state_update_op(bw_states, output_states[1])
    
    outputs = tf.concat(outputs, 2)
    
    #output_shape1 = len(outputs)
    #print("output_shape1:", output_shape1)    # 25
        
    #output_shape1a = outputs[0].get_shape()
    #print("output_shape1a:", output_shape1a)    # (?,1024) 
        
    #outputs = tf.stack(outputs)
    output_shape2 = outputs.get_shape()
    print("output_shape2:", output_shape2)    #(25, ?, 1024)
       
    #outputs = tf.transpose(outputs, [1, 0, 2])
    #output_shape3 = outputs.get_shape()
    #print("output_shape3:", output_shape3)    #(?, 25, 1024)
    
    outputs = tf.reshape(outputs, [-1, 1024])
    output_shape4 = outputs.get_shape()
    print("output_shape4:", output_shape4)    #(?, 1024)
    
    outputs = tf.matmul(outputs, W_lstm1) + B_lstm1
    output_shape5 = outputs.get_shape()
    print("output_shape5:", output_shape5)    #[?,1024].[1024,1024]

####################################################################################

    #add FC layer to LSTM via shortcut connection
    with tf.name_scope("Add_LSTM_Shortcut"):
        shortcut_added = tf.add_n([outputs, fullyconnected_cnn])  
    shortcut_added_shape= shortcut_added.get_shape()
    print("shortcut_added_shape:", shortcut_added_shape)    #output size (?, 5)

    #dropout5 (dropout 3 inside multilayer LSTM)   
    with tf.name_scope("dropout5"):
        dropout_5 = dropout_layer(shortcut_added, phase_test, dropout_prob) 

    y_logits = fc_layer(dropout_5, 1024, 5)   

    #predict layer - find th probablities of the highest classes and predict it
    with tf.name_scope("predict"):
        y_pred = tf.nn.softmax(y_logits, name="y_logits")
        #convert from percentage to 1 hot
        y_pred_cls = tf.argmax(y_pred, dimension=1)

    # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
    # problems with log(0) which is NaN
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits,labels=y_true)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, weights_list)        
        cost = tf.reduce_mean(cross_entropy) + reg_term
        tf.summary.scalar("loss", cost)

    #optimize for CNN and LSTM variables separately with different learning rates
    #optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    #separate out the CNN var list from the LSTM var list.
    cnn_var_list_names = ['W_1s:0','B_1s:0','W_2s:0','B_2s:0','W_3s:0','B_3s:0',
                          'W_4s:0','B_4s:0','W_1l:0','B_1l:0','W_2l:0','B_2l:0',
                          'W_3l:0','B_3l:0','W_4l:0','B_4l:0','BN_beta_1s:0',
                          'BN_gamma_1s:0', 'BN_beta_2s:0', 'BN_gamma_2s:0','BN_beta_3s:0', 
                          'BN_gamma_3s:0', 'BN_beta_4s:0', 'BN_gamma_4s:0','BN_beta_1l:0',
                          'BN_gamma_1l:0', 'BN_beta_2l:0', 'BN_gamma_2l:0','BN_beta_3l:0',
                          'BN_gamma_3l:0', 'BN_beta_4l:0', 'BN_gamma_4l:0']

    var_list = tf.trainable_variables()
    #variable_names = [v.name for v in tf.trainable_variables()]
    lstm_var_list = []
    for i in var_list:
        if not i.name in cnn_var_list_names:
            lstm_var_list.append(i)

    cnn_var_list = []
    for i in var_list:
        if not i in lstm_var_list:
            cnn_var_list.append(i)

    #optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)        
    #with tf.name_scope("optimize"):
        #train_op1 = tf.train.AdamOptimizer(0.0001).minimize(cost, var_list=cnn_var_list)
        #train_op2 = tf.train.AdamOptimizer(0.000001).minimize(cost, var_list=lstm_var_list)
        #optimizer= tf.group(train_op1, train_op2)

    #applying different learning rates to the adam optimizers and 
    #applying gradient clipping with threshold of 10

    #CNN optimizer
    adamopt_cnn = tf.train.AdamOptimizer(0.000001)
    gradients_cnn, variables_cnn = zip(*adamopt_cnn.compute_gradients(cost, var_list=cnn_var_list))
    gradients_cnn, _ = tf.clip_by_global_norm(gradients_cnn, 10.0)
    optimizer_cnn = adamopt_cnn.apply_gradients(zip(gradients_cnn, variables_cnn))

    global_norm_cnn = tf.global_norm(gradients_cnn)
    tf.summary.scalar("gradients_cnn", global_norm_cnn)

    #LSTM optimizer
    adamopt_lstm = tf.train.AdamOptimizer(0.0001)
    gradients_lstm, variables_lstm = zip(*adamopt_lstm.compute_gradients(cost, var_list=lstm_var_list))
    gradients_lstm, _ = tf.clip_by_global_norm(gradients_lstm, 10.0)
    optimizer_lstm = adamopt_lstm.apply_gradients(zip(gradients_lstm, variables_lstm))

    global_norm_lstm = tf.global_norm(gradients_lstm)
    tf.summary.scalar("gradients_lstm", global_norm_lstm)

    optimizer= tf.group(optimizer_cnn, optimizer_lstm)

    with tf.name_scope("accuracy"):
        correct_prediction= tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)



##################################################################################
#init only the unrestored variables
#read from a saved file of global variables from restored model 
    with open('cnn_gvar_list_edited.txt') as f:
        CNN_gvar_list = f.readlines()
    CNN_gvar_list = [x.strip() for x in CNN_gvar_list] 

#do not initialise any of the restored varibles, only the new variables.
    gv_list = tf.global_variables()
    list_to_init = []
    for i in gv_list:
        if not i.name in CNN_gvar_list:
            list_to_init.append(i)

    session.run(tf.local_variables_initializer()) 
    session.run(tf.variables_initializer(list_to_init))
    
####################################################################################


    #add tensorboard details
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("../logdir")
    writer.add_graph(session.graph)
    #to start tensorboard
    #tensorboard --logdir ./logdir
    #http://129.78.10.182:6006/

    #add model saver
    model_saver_full = tf.train.Saver()

    

    def rep_learn_optimise(num_iterations):       
        tensorboard_iter_count = 0
        for i in range(num_iterations):
            for subj_night in np.arange(0, len(x_list)):    #x_list contains list of sleep subjects
                x_subject = x_list[subj_night]
                y_subject = y_list[subj_night]
                subject_len = len(x_subject)
                num_sequences = (subject_len//sequence_length) #changed num-steps from 25 to 40. ignoring the final few wake samples for each subject
                print("subject length and num_sequences are", subject_len,  num_sequences)

                            
                #seq_idx=0
                #further split into length of 25 or less samples
                for seq_num in range(num_sequences):                        
                    #if the next 25 will exceed total size of batch
                    #ignoring the final few wake samples for each subject
                    #if (seq_idx+sequence_length > subject_len):
                        #xtra = subject_len - seq_idx
                        #x_seq = x_subject[seq_idx:seq_idx+xtra]
                        #y_seq = y_subject[seq_idx:seq_idx+xtra]
                        #break
                                                
                    #else:
                    x_seq = x_subject[seq_num*sequence_length:(seq_num*sequence_length)+sequence_length]
                    y_seq = y_subject[seq_num*sequence_length:(seq_num*sequence_length)+sequence_length]
                
                    feed_dict_train = {'x:0':x_seq,'y_true:0':y_seq,'phase_test:0':False,'iteration:0':i}
                
                    session.run([optimizer, fw_update_op, bw_update_op], feed_dict=feed_dict_train)
                

                    # Calculate the accuracy on the training-set.
                    #tensorboard_iter_count +=1
                    if seq_num % 5 == 0:
                        #s = session.run(merged_summary, feed_dict_train)
                        #writer.add_summary(s, tensorboard_iter_count)
           
                        acc = session.run(accuracy, feed_dict=feed_dict_train)
                        msg = "LSTM Optimization Iteration: {0:>6}, Subject Night:{1:>6}, Sequence No: {2:>6},  Training Accuracy: {3:>6.1%}"
                        print(msg.format(i + 1, subj_night, seq_num, acc))
 
                        #reset lstm state after ever 5 iterations
                    #if seq_num % 10 == 0:
                     #   session.run([fw_reset_state_op, bw_reset_state_op])	

            
                # save the model after each subject-night
                model_saver_full.save(session, "../saved_models/capstone_lstm_fullmodel"+subject+"/", global_step=100)
                # reset the lstm to 0 after each subject-night
                session.run([fw_reset_state_op, bw_reset_state_op])
   
            #predict after every iteration (each iteration is 19*2 subjects-nights)
            tensorboard_iter_count +=1
            tensorboard_iter_count = stage_predict(tensorboard_iter_count)



    def stage_predict(tensorboard_iter_count):


        #Evaludation on test set
        #seqbatch_idx=0
        y_predicted = []
        y_truth = []
    
        #only loop through the entire test set once
        for subj_night in np.arange(0, len(x_list_test)):    #x_list contains list of sleep subjects
            x_test_subject = x_list_test[subj_night]
            y_test_subject = y_list_test[subj_night]
            subject_len_test = len(x_test_subject)
            print("x_test_subject shape", x_test_subject.shape)
            print("x_test_subject leng", len(x_test_subject))
            
            num_sequences = (subject_len_test//sequence_length) #changed num-steps from 25 to 40. ignoring the final few wake samples for each subject
            print("test subject length and num_sequences are", subject_len_test,  num_sequences)

                            
            #seq_idx=0
            #further split into length of 25 or less samples
            for seq_num in range(num_sequences):                        
                #if the next 25 will exceed total size of batch
                #ignoring the final few wake samples for each subject
                #if (seq_idx+sequence_length > subject_len):
                    #xtra = subject_len - seq_idx
                    #x_seq = x_subject[seq_idx:seq_idx+xtra]
                    #y_seq = y_subject[seq_idx:seq_idx+xtra]
                    #break
                                                
                #else:
                x_seq_test = x_test_subject[seq_num*sequence_length:(seq_num*sequence_length)+sequence_length]
                y_seq_test = y_test_subject[seq_num*sequence_length:(seq_num*sequence_length)+sequence_length]
                
                feed_dict_test = {'x:0':x_seq_test,'y_true:0':y_seq_test,'phase_test:0':True,'iteration:0':seq_num}
                predicted_class,_,_ = session.run([y_pred_cls, fw_update_op, bw_update_op], feed_dict=feed_dict_test)
        
                #print("seq num now is", seq_num)
                y_predicted.extend(predicted_class)
                #print("y_pred leng", len(y_predicted))        

                y_truth.extend(np.argmax(y_seq_test,1))
                #print("y_truth leng", len(y_truth))

                #reset lstm state after every 5 iterations
                #if seq_num % 10 == 0:
                #    session.run([fw_reset_state_op, bw_reset_state_op])	
                #    tensorboard_iter_count +=1
                #    s = session.run(merged_summary, feed_dict_test)
                #    writer.add_summary(s, tensorboard_iter_count)


        test_correct_pred = np.equal(y_predicted, y_truth)
        test_acc = np.count_nonzero(test_correct_pred)/len(test_correct_pred)
    
        print ("Test accuracy is", test_acc)
        target_names = ["Sleep Stage W", "Sleep Stage 1", "Sleep Stage 2", "Sleep Stage 3/4", "Sleep Stage REM"]
        #print ("Classification Report")
        print (classification_report(y_truth, y_predicted, target_names=target_names))
        
        #print ("Confusion_Matrix")
        print (confusion_matrix(y_truth, y_predicted))
         
        with open('./output/lstm_result_stateful'+subject+'.txt', 'w+') as f:
            print >> f, classification_report(y_truth, y_predicted, target_names=target_names)
            print >> f, confusion_matrix(y_truth, y_predicted)
        
        return tensorboard_iter_count
    
               
    rep_learn_optimise(num_iterations)
    #stage_predict()



