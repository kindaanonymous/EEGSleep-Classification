# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:03:44 2017

@author: HP-PC
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import time
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler 
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder


#looping through all subject's training and testing file, and saving the individual model 

for sub_num in np.arange(3,4):
    
    subject = str(sub_num)
    print("subject number", subject)

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


    #initialise placeholders
    lr = tf.placeholder(tf.float32)
    # test flag for batch norm
    phase_test = tf.placeholder(tf.bool)
    iteration = tf.placeholder(tf.int32)

    x = tf.placeholder(tf.float32, shape=[None, stage_length], name='x')
    x_stage = tf.reshape(x, [-1, stage_length, num_channels])   #batch, in_width, channels

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)


    #weights and biases
    # filter tensor of shape [filter_width, in_channels, out_channels]
    W_1s = tf.Variable(tf.truncated_normal([50, 1, 64], stddev=0.1), name='W_1s')  
    B_1s = tf.Variable(tf.constant(0.1, tf.float32, [64]), name='B_1s')
    W_2s = tf.Variable(tf.truncated_normal([8, 64, 128], stddev=0.1), name='W_2s')  
    B_2s = tf.Variable(tf.constant(0.1, tf.float32, [128]), name='B_2s')
    W_3s = tf.Variable(tf.truncated_normal([8, 128, 128], stddev=0.1), name='W_3s') 
    B_3s = tf.Variable(tf.constant(0.1, tf.float32, [128]), name='B_3s')
    W_4s = tf.Variable(tf.truncated_normal([8, 128, 128], stddev=0.1), name='W_4s')  
    B_4s = tf.Variable(tf.constant(0.1, tf.float32, [128]), name='B_4s')

    W_1l = tf.Variable(tf.truncated_normal([400, 1, 64], stddev=0.1), name='W_1l')
    B_1l = tf.Variable(tf.constant(0.1, tf.float32, [64]), name='B_1l')
    W_2l = tf.Variable(tf.truncated_normal([6, 64, 128], stddev=0.1), name='W_2l')  
    B_2l = tf.Variable(tf.constant(0.1, tf.float32, [128]), name='B_2l')
    W_3l = tf.Variable(tf.truncated_normal([6, 128, 128], stddev=0.1), name='W_3l') 
    B_3l = tf.Variable(tf.constant(0.1, tf.float32, [128]), name='B_3l')
    W_4l = tf.Variable(tf.truncated_normal([6, 128, 128], stddev=0.1), name='W_4l')  
    B_4l = tf.Variable(tf.constant(0.1, tf.float32, [128]), name='B_4l')

    BN_beta_1s = tf.Variable(tf.constant(0.0, shape=[num_filters_1s]),name='BN_beta_1s', trainable=True)
    BN_gamma_1s = tf.Variable(tf.constant(1.0, shape=[num_filters_1s]),name='BN_gamma_1s', trainable=True)
    BN_beta_2s = tf.Variable(tf.constant(0.0, shape=[num_filters_s]),name='BN_beta_2s', trainable=True)
    BN_gamma_2s = tf.Variable(tf.constant(1.0, shape=[num_filters_s]),name='BN_gamma_2s', trainable=True)
    BN_beta_3s = tf.Variable(tf.constant(0.0, shape=[num_filters_s]),name='BN_beta_3s', trainable=True)
    BN_gamma_3s = tf.Variable(tf.constant(1.0, shape=[num_filters_s]),name='BN_gamma_3s', trainable=True)
    BN_beta_4s = tf.Variable(tf.constant(0.0, shape=[num_filters_s]),name='BN_beta_4s', trainable=True)
    BN_gamma_4s = tf.Variable(tf.constant(1.0, shape=[num_filters_s]),name='BN_gamma_4s', trainable=True)

    BN_beta_1l = tf.Variable(tf.constant(0.0, shape=[num_filters_1l]),name='BN_beta_1l', trainable=True)
    BN_gamma_1l = tf.Variable(tf.constant(1.0, shape=[num_filters_1l]),name='BN_gamma_1l', trainable=True)
    BN_beta_2l = tf.Variable(tf.constant(0.0, shape=[num_filters_l]),name='BN_beta_2l', trainable=True)
    BN_gamma_2l = tf.Variable(tf.constant(1.0, shape=[num_filters_l]),name='BN_gamma_2l', trainable=True)
    BN_beta_3l = tf.Variable(tf.constant(0.0, shape=[num_filters_l]),name='BN_beta_3l', trainable=True)
    BN_gamma_3l = tf.Variable(tf.constant(1.0, shape=[num_filters_l]),name='BN_gamma_3l', trainable=True)
    BN_beta_4l = tf.Variable(tf.constant(0.0, shape=[num_filters_l]),name='BN_beta_4l', trainable=True)
    BN_gamma_4l = tf.Variable(tf.constant(1.0, shape=[num_filters_l]),name='BN_gamma_4l', trainable=True)

    #regularisation 
    #we only want to add regularisation to the weights of the 1st 2 layers
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)  #L2 Lambda is 0.001
    weights_list_s = [W_1s]
    weights_list_l = [W_1l]
  

    #Read in data
    #==============================================================================

    df = pd.read_csv('../Physionet-sleepedf/csv files/formatted/consolidated files/trainfile'+subject+'.csv')
    
    np_y = np.array(df.loc[:,"combined sleep stage"], dtype=str)
    np_x = np.array(df.loc[:,'0':'2999']  , dtype=float)  #3000 columns of the eeg
  
    #converting y indices for one-hot encoding later on
    np.place(np_y, np_y=="Sleep stage W", 0)
    np.place(np_y, np_y=="Sleep stage 1", 1)
    np.place(np_y, np_y=="Sleep stage 2", 2)
    np.place(np_y, np_y=="Sleep stage 3", 3)
    np.place(np_y, np_y=="Sleep stage R", 4)
 
    np_y = np_y.astype(int)
    
    #np_x contains the features of the eeg (3000 features per sleep stage)
    #np_y contains the targets of the eeg 5 sleep stages from 0 to 4
    #using SMOTE resample np_x and np_y to an even dataset
    print('Original dataset shape {}'.format(Counter(np_y)))
    sm = RandomOverSampler(random_state=42)
    np_x_res, np_y_res = sm.fit_sample(np_x, np_y)
    print('Resampled dataset shape {}'.format(Counter(np_y_res)))
 
    #combine x and y resampled to save as a file for LSTM use
    #res_data = np.column_stack((np_y_res, np_x_res))
    #np.savetxt('../Physionet-sleepedf/csv files/formatted/consolidated files/trainfile_res1.csv', res_data, delimiter=',')
 
 
    ##create a one hot encoding from the y resampled set
    index_offset = np.arange(len(np_y_res))* num_classes
    y_onehot = np.zeros((np_y_res.shape[0], num_classes))
    y_onehot.flat[index_offset + np_y_res.ravel()] = 1
    #==============================================================================


    ################################################################################
    #Read in data for testing
    df = pd.read_csv('../Physionet-sleepedf/csv files/formatted/test files/testfile'+subject+'.csv')
    np_y_test = np.array(df.loc[:,"combined sleep stage"], dtype=str)
    np_x_test = np.array(df.loc[:,'0':'2999']  , dtype=float)  #3000 columns of the eeg

    #converting y indices for one-hot encoding later on
    np.place(np_y_test, np_y_test=="Sleep stage W", 0)
    np.place(np_y_test, np_y_test=="Sleep stage 1", 1)
    np.place(np_y_test, np_y_test=="Sleep stage 2", 2)
    np.place(np_y_test, np_y_test=="Sleep stage 3", 3)
    np.place(np_y_test, np_y_test=="Sleep stage R", 4)

    np_y_test = np_y_test.astype(int)

    #create a one hot encoding from the y resampled set
    index_offset = np.arange(len(np_y_test))* num_classes
    y_onehot_test = np.zeros((np_y_test.shape[0], num_classes))
    y_onehot_test.flat[index_offset + np_y_test.ravel()] = 1
    ################################################################################

    #various encapsulators defined


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
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.00001) #epsilon is 0.001
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
        #print("flatten_shape:", flatten_shape)
        num_features = flatten_shape[1:3].num_elements()
        reshaped = tf.reshape(input, shape=[-1, num_features])
        #reshaped_shape = reshaped.get_shape()
        #print("reshaped_shape:", reshaped_shape)       
        return reshaped, num_features

    def fc_layer(input, num_inputs, num_outputs):
        weights = new_weights(shape = [num_inputs, num_outputs])
        #print("weights", weights) 
        biases = new_biases(length = num_outputs)
        #print("bias", biases)
        fullyconnected = tf.matmul(input, weights) + biases
        #fc_shape = fullyconnected.get_shape()
        #print("fc_shape:", fc_shape)
        return fullyconnected

    def CNN_small(input, phase_test, iteration):
    
        #stride is 6 for 1st conv1d and 1 for rest.  Filter size 50 and 8 for conv2 onwards
        #input is (?, 3000, 1), output is (?, 492, 64)  because 3000-50/6
        #input_shape = input.get_shape()
        #print("input_shape:", input_shape)

        conv1s = new_conv_layer(input, W_1s, B_1s, stride_1s, phase_test, iteration, BN_beta_1s, BN_gamma_1s )
        #conv1s_shape = conv1s.get_shape()
        #print("conv1s_shape:", conv1s_shape)    
    
        conv1s = tf.expand_dims(conv1s, 0)   #adds extra dimension to make it 4d for max pool
        #max pool stride and size = 8.  Output size (?, 62, 64)
        #conv1s_shape = conv1s.get_shape()
        #print("conv1s_shape:", conv1s_shape)

        #note that pool size and stride is 3rd variable!
        max_pool1s = tf.nn.max_pool(conv1s, 
                                    ksize=[1, 1, pool_size_1s,  1],
                                    strides=[1, 1, pool_stride_1s, 1], 
                                    padding='SAME')
    
        #max_pool1s_shape = max_pool1s.get_shape()
        #print("max_pool1s_shape:", max_pool1s_shape)
  
        max_pool1s = tf.squeeze(max_pool1s,0) #takes away added dimension
        #max_pool1s_shape = max_pool1s.get_shape()
        #print("max_pool1s_shape:", max_pool1s_shape)

        dropout_s = dropout_layer(max_pool1s, phase_test, dropout_prob)       
        #dropout_shape = dropout_s.get_shape()
        #print("dropout_shape:", dropout_shape)

        #filter size 8, stride size 1 for next 3 conv
        #input size(?, 62, 64) output size (?, 55 , 128)    
        conv2s = new_conv_layer(dropout_s, W_2s, B_2s, stride_s, phase_test, iteration,BN_beta_2s, BN_gamma_2s)      
        #conv2s_shape = conv2s.get_shape()
        #print("conv2s_shape:", conv2s_shape)
    
        #output size(?, 48 ,128) 
        conv3s = new_conv_layer(conv2s, W_3s, B_3s, stride_s, phase_test, iteration, BN_beta_3s, BN_gamma_3s)      
        #conv3s_shape = conv3s.get_shape()
        #print("conv3s_shape:", conv3s_shape)

        #output size(?, 41 ,128)
        conv4s = new_conv_layer(conv3s, W_4s, B_4s, stride_s, phase_test, iteration, BN_beta_4s, BN_gamma_4s)       
        #conv4s_shape = conv4s.get_shape()
        #print("conv4s_shape:", conv4s_shape)   
        conv4s = tf.expand_dims(conv4s, 0)   #adds extra dimension to make it 4d for max pool    
        #conv4s_shape = conv4s.get_shape()
        #print("conv4s_shape:", conv4s_shape)

        #max pool stride and size = 4.  Output size (?, 11, 128)
        max_pool2s = tf.nn.max_pool(conv4s, 
                                    ksize=[1, 1, pool_size_2s, 1],
                                    strides=[1, 1, pool_stride_2s, 1], 
                                    padding='SAME')

        #mp_shape= max_pool2s.get_shape()
        #print("max_pool2s shape:", mp_shape)
        max_pool2s = tf.squeeze(max_pool2s,0) #takes away added dimension
        #mp_shape= max_pool2s.get_shape()
        #print("max_pool2s shape:", mp_shape)
    
        return max_pool2s

    
    def CNN_large(input, phase_test, iteration):
    
        #Filter size 400 for conv1 and 6 for conv2 onwards. Stride is 50 for 1st conv1d and 1 for rest.  
        #Stride size of 100/2=50 too large, change to 100/4=25 instead
        #input is (?, 3000, 1), output is (?, 104, 64)  because 3000-400/25
        conv1l = new_conv_layer(input, W_1l, B_1l, stride_1l, phase_test, iteration, BN_beta_1l, BN_gamma_1l)
        #conv1l_shape = conv1l.get_shape()
        #print("conv1l_shape:", conv1l_shape)    
   
        #max_pool1l size and stride is 4
        #max_pool1l output is (?, 26, 64)
        conv1l = tf.expand_dims(conv1l, 0)   #adds extra dimension to make it 4d for max pool
        max_pool1l = tf.nn.max_pool(conv1l, 
                                    ksize=[1, 1, pool_size_1l, 1],
                                    strides=[1, 1, pool_stride_1l, 1], 
                                    padding='VALID')
        max_pool1l = tf.squeeze(max_pool1l,0) #takes away added dimension
        #max_pool1l_shape = max_pool1l.get_shape()
        #print("max_pool1l_shape:", max_pool1l_shape)    

        dropout_l = dropout_layer(max_pool1l, phase_test, dropout_prob)    
    
        #conv2l output is (?, 21, 128)    
        conv2l = new_conv_layer(dropout_l, W_2l, B_2l, stride_l, phase_test, iteration, BN_beta_2l, BN_gamma_2l)    
        #conv2l_shape = conv2l.get_shape()
        #print("conv2l_shape:", conv2l_shape)    

        #conv2l output is (?, 16, 128)
        conv3l = new_conv_layer(conv2l, W_3l, B_3l, stride_l, phase_test, iteration, BN_beta_3l, BN_gamma_3l)    
        #conv3l_shape = conv3l.get_shape()
        #print("conv3l_shape:", conv3l_shape)    
    
        #conv2l output is (?, 11, 128)      
        conv4l = new_conv_layer(conv3l, W_4l, B_4l, stride_l, phase_test, iteration, BN_beta_4l, BN_gamma_4l)
        #conv4l_shape = conv4l.get_shape()
        #print("conv4l_shape:", conv4l_shape)    
    
        #max_pool2l size and stride is 2
        #conv2l output is (?, 5, 128)      
        conv4l = tf.expand_dims(conv4l, 0)   #adds extra dimension to make it 4d for max pool
        max_pool2l = tf.nn.max_pool(conv4l, 
                                    ksize=[1, 1,  pool_size_2l, 1],
                                    strides=[1, 1, pool_stride_2l, 1], 
                                    padding='VALID')
        max_pool2l = tf.squeeze(max_pool2l,0) #takes away added dimension
        #max_pool2l_shape = max_pool2l.get_shape()
        #print("max_pool2l_shape:", max_pool2l_shape)    

        return max_pool2l


    #cnn layer
    max_pool2s = CNN_small(x_stage, phase_test, iteration)   #Output size (?, 9, 128)
    max_pool2l = CNN_large(x_stage, phase_test, iteration)
    #flatten layer
    flatten_s, num_features_s = flatten_layer(max_pool2s) #flattenshape:(?, 471, 128)
    flatten_l, num_features_l = flatten_layer(max_pool2l) #flattenshape:(?, 38, 128)
    #fully connected layer
    #fc_s = fc_layer(flatten_s, num_features_s, fc_size)  #num_features_s = 471 * 128
    #fc_l = fc_layer(flatten_l, num_features_l, fc_size)  #num_features_l = 38 * 128
    #fully connected layer 2
    y_logits_s = fc_layer(flatten_s, num_features_s, num_classes)
    y_logits_l = fc_layer(flatten_l, num_features_l, num_classes)

    #y_logits_s = fc_layer(fc_s, fc_size, num_classes)
    #y_logits_l = fc_layer(fc_l, fc_size, num_classes)

    #predict layer - find th probablities of the highest classes and predict it
    y_pred_s = tf.nn.softmax(y_logits_s, name="y_pred_s")
    y_pred_l = tf.nn.softmax(y_logits_l, name="y_pred_l")
    #convert from percentage to 1 hot
    y_pred_cls_s = tf.argmax(y_pred_s, dimension=1)
    y_pred_cls_l = tf.argmax(y_pred_l, dimension=1)

    # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
    # problems with log(0) which is NaN
    cross_entropy_s = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits_s,labels=y_true)
    cross_entropy_l = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits_l,labels=y_true)

    reg_term_s = tf.contrib.layers.apply_regularization(regularizer, weights_list_s)
    reg_term_l = tf.contrib.layers.apply_regularization(regularizer, weights_list_l)

    with tf.name_scope("loss"):
        cost_s = tf.reduce_mean(cross_entropy_s) + reg_term_s
        cost_l = tf.reduce_mean(cross_entropy_l) + reg_term_l
        tf.summary.scalar("cost_s", cost_s)
        tf.summary.scalar("cost_l", cost_l)

    optimizer_s = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_s)
    optimizer_l = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_l)

    correct_prediction_s = tf.equal(y_pred_cls_s, y_true_cls)
    correct_prediction_l = tf.equal(y_pred_cls_l, y_true_cls)

    with tf.name_scope("accuracy"):
        accuracy_s = tf.reduce_mean(tf.cast(correct_prediction_s, tf.float32))
        accuracy_l = tf.reduce_mean(tf.cast(correct_prediction_l, tf.float32))
        tf.summary.scalar("accuracy_s", accuracy_s)
        tf.summary.scalar("accuracy_l", accuracy_l)



    #create model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    session = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
    session.run(tf.global_variables_initializer()) 
    #model_saver = tf.train.Saver({"W_1s":W_1s, "B_1s":B_1s, "BN_beta_1s": BN_beta_1s,"BN_gamma_1s": BN_gamma_1s,
    #                               "W_2s":W_2s, "B_2s":B_2s, "BN_beta_2s": BN_beta_2s,"BN_gamma_2s": BN_gamma_2s,    
    #                               "W_3s":W_3s, "B_3s":B_3s, "BN_beta_3s": BN_beta_3s,"BN_gamma_3s": BN_gamma_3s,
    #                               "W_4s":W_4s, "B_4s":B_4s, "BN_beta_4s": BN_beta_4s,"BN_gamma_4s": BN_gamma_4s,
    #                               "W_1l":W_1l, "B_1l":B_1l, "BN_beta_1l": BN_beta_1l,"BN_gamma_1l": BN_gamma_1l,
    #                               "W_2l":W_2l, "B_2l":B_2l, "BN_beta_2l": BN_beta_2l,"BN_gamma_2l": BN_gamma_2l,
    #                               "W_3l":W_3l, "B_3l":B_3l, "BN_beta_3l": BN_beta_3l,"BN_gamma_3l": BN_gamma_3l,
    #                               "W_4l":W_4l, "B_4l":B_4l, "BN_beta_4l": BN_beta_4l,"BN_gamma_4l": BN_gamma_4l})
    model_saver_full = tf.train.Saver()


    #==============================================================================    
    # #If reloading saved weights  
    #no need to call global_variables_initializer()
    #model_saver_full.restore(session, 'C:/Users/HP-PC/Documents/fran/Masters/Comp5703 Capstone Project/saved_models/saved model full resampled train set 30k iterations/capstone_cnn_fullmodel-2/-100')
    #print("Model restored.")
    #============================================================================
    
    #add tensorboard details
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("../logdir")
    writer.add_graph(session.graph)
    #to start tensorboard, at cmd prompt type line below
    #tensorboard --logdir "C:\Users\HP-PC\Documents\fran\Masters\Comp5703 Capstone Project\logdir"


    train_batch_size = 100
    num_iterations = 20000
    
    def rep_learn_optimise(num_interations):
        # Ensure we update the global variable rather than a local copy.

        for i in range(num_iterations):

            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
        
            x_batch, y_true_batch = random_batch(train_batch_size, np_x_res, y_onehot)
            x_batch_test, y_true_batch_test = random_batch(train_batch_size, np_x_test, y_onehot_test)


            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch,
                               lr: 0.0001,
                               phase_test: False,
                               iteration: i}

            feed_dict_test = {x: x_batch_test,
                              y_true: y_true_batch_test,
                              lr: 0.0001,
                              phase_test: True,
                              iteration: i}


            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            session.run(optimizer_s, feed_dict=feed_dict_train)
            session.run(optimizer_l, feed_dict=feed_dict_train)

            #add summaries for tensorboard tracking
            if i % 5 == 0:
                s = session.run(merged_summary, feed_dict_test)
                writer.add_summary(s, i)

            # Print status every 100 iterations.
            if i % 1000 == 0:
                #model_saver.save(session, "../saved_models/capstone_cnn_trainvar", global_step=100)
                model_saver_full.save(session, "../saved_models/capstone_cnn_fullmodel_relu_bn"+subject+"/", global_step=100)
           
                # Calculate the accuracy on the training-set.
                acc_s = session.run(accuracy_s, feed_dict=feed_dict_train)
                acc_l = session.run(accuracy_l, feed_dict=feed_dict_train)
                acc_s_test = session.run(accuracy_l, feed_dict=feed_dict_test)
                acc_l_test = session.run(accuracy_l, feed_dict=feed_dict_test)

                # Message for printing.
                msg = "CNN S Optimization Iteration: {0:>6}, Training Accuracy_s: {1:>6.1%}"
                print(msg.format(i + 1, acc_s))
                msg = "CNN L Optimization Iteration: {0:>6}, Training Accuracy_l: {1:>6.1%}"
                print(msg.format(i + 1, acc_l))
                msg = "CNN S Optimization Iteration: {0:>6}, Testing Accuracy_s: {1:>6.1%}"
                print(msg.format(i + 1, acc_s_test))
                msg = "CNN L Optimization Iteration: {0:>6}, Testing Accuracy_l: {1:>6.1%}"
                print(msg.format(i + 1, acc_l_test))
            
            
        
        
    def stage_predict_s():

        #Evaludation on test set
        seqbatch_idx=0
        y_predicted = []
        y_truth = []
    
        #only loop through the entire test set once
        num_test_samples = len(y_onehot_test)
        if num_test_samples % train_batch_size == 0:
            iterations = num_test_samples/train_batch_size
        else: iterations = num_test_samples//train_batch_size + 1
    
        for i in range(iterations):       
            x_batch_test, y_true_batch_test = sequential_batch(train_batch_size, seqbatch_idx, np_x_test, y_onehot_test)
            seqbatch_idx = seqbatch_idx + train_batch_size

            feed_dict_train = {x: x_batch_test,
                               y_true: y_true_batch_test,
                               phase_test: True,
                               iteration: i}

            predicted_class = session.run(y_pred_cls_s, feed_dict=feed_dict_train)
        
            y_predicted.extend(predicted_class)
            y_truth.extend(np.argmax(y_true_batch_test,1))
        
        
        target_names = ["Sleep Stage W", "Sleep Stage 1", "Sleep Stage 2", "Sleep Stage 3/4", "Sleep Stage REM"]
        print ("Classification Report CNN small")
        print (classification_report(y_truth, y_predicted, target_names=target_names))
        
        print ("Confusion_Matrix CNN small")
        print (confusion_matrix(y_truth, y_predicted))    
    
        with open('./output/cnn_result_oversamp'+subject+'.txt', 'w+') as f:
            print >> f, classification_report(y_truth, y_predicted, target_names=target_names)
            print >> f, confusion_matrix(y_truth, y_predicted)
                

    def stage_predict_l():

        #Evaludation on test set
        seqbatch_idx=0
        y_predicted = []
        y_truth = []
    
        #only loop through the entire test set once
        num_test_samples = len(y_onehot_test)
        if num_test_samples % train_batch_size == 0:
            iterations = num_test_samples/train_batch_size
        else: iterations = num_test_samples//train_batch_size + 1
    
        for i in range(iterations):       
            x_batch_test, y_true_batch_test = sequential_batch(train_batch_size, seqbatch_idx, np_x_test, y_onehot_test)
            seqbatch_idx = seqbatch_idx + train_batch_size

            feed_dict_train = {x: x_batch_test,
                               y_true: y_true_batch_test,
                               phase_test: True,
                               iteration: i}

            predicted_class = session.run(y_pred_cls_l, feed_dict=feed_dict_train)
        
            y_predicted.extend(predicted_class)
            y_truth.extend(np.argmax(y_true_batch_test,1))
        
        target_names = ["Sleep Stage W", "Sleep Stage 1", "Sleep Stage 2", "Sleep Stage 3/4", "Sleep Stage REM"]
        print ("Classification Report CNN large")
        print (classification_report(y_truth, y_predicted, target_names=target_names))
        
        print ("Confusion_Matrix CNN large")
        print (confusion_matrix(y_truth, y_predicted)) 
    
        with open('./output/cnn_result_oversamp'+subject+'.txt', 'a+') as f:
            print >> f, classification_report(y_truth, y_predicted, target_names=target_names)
            print >> f, confusion_matrix(y_truth, y_predicted)
    



    rep_learn_optimise(num_iterations)
    stage_predict_s()
    stage_predict_l()

