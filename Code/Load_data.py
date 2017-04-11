#coding:utf-8
""" Preprocessing:
*** This file is used for loading data from folder 'Data'
"""

import os
import exceptions
from PIL import Image
import numpy as np
import scipy.io as scio

# Read the Image in 'Data', set the channel is 3 for RGB
# size : 180 * 240 * 3
def load_data():
    ## AftRefined: train & test data
    #  data_train = np.empty((10800, 180, 240, 3), dtype="uint8")
    #  data_test = np.empty((2700, 180, 240, 3), dtype="uint8")
    ## train & test label
    # label_train = np.empty((10800,), dtype="uint8")
    # label_test = np.empty((2700,), dtype="uint8")
    ## NotRefined: train & testdata
    data_train = np.empty((17780, 180, 240, 3), dtype="float16")
    data_test = np.empty((4445, 180, 240, 3), dtype="float16")
    # train & test label
    label_train = np.empty((17780,), dtype="uint8")
    label_test = np.empty((4445,), dtype="uint8")
    # windows : dataPath = "D:/WorkSpace/Python/Convolutional Neural Network/MySelf/Exp_Data/"
    # linuex:
    dataPath = "/home/assassinator/Exp/Convolutional Neural Network/MySelf/Exp_Data/NotRefined"
    trainPath = "/train/"
    testPath = "/test/"
    # imgs = os.listdir("./mnist")
    # processing : train data
    imgs_train = os.listdir(dataPath + trainPath)
    num = len(imgs_train)
    for i in range(num):
        # img = Image.open("./mnist/"+imgs[i])
        img = Image.open(dataPath + trainPath + imgs_train[i])
        arr = np.asarray(img, dtype="float16")
        data_train[i,:,:,:] = arr
        # data_train[i,:,:,:] = img
        label_train[i] = int(imgs_train[i].split('_')[0]) - 1
        # print "( %d ) is successful!" % i
    print "Train Data Preprocessing Completed!"
    # processing : test data
    imgs_test = os.listdir(dataPath + testPath)
    num = len(imgs_test)
    for i in range(num):
        img = Image.open(dataPath + testPath + imgs_test[i])
        arr = np.asarray(img, dtype="float16")
        data_test[i,:,:,:] = arr
        # data_test[i,:,:,:] = img
        label_test[i] = int(imgs_test[i].split('_')[0]) - 1
        # print "( %d ) is successful!" % i
    print "Test Data Preprocessing Completed!"
    data = {"train":data_train, "test":data_test}
    label = {"train":label_train, "test":label_test}
    
    return data,label

'''This Func is used for import the ".mat" file
   .mat file include "Vx" & "Vy" & "Label", each of them consists of 180*240*k .
'''
def load_OP_data():
    # from .mat file import train & test data
    FilePath = "/home/assassinator/Exp/Convolutional Neural Network/MySelf/Exp_Data/OpticalFlow/full_data.mat"
    FullData = scio.loadmat(FilePath)
    row, column = FullData["train"][0][0].shape[:2]
    Num = [FullData["train"][2][0].shape[0], FullData["test"][2][0].shape[0]]

    # construct data numpy
    '''train'''
    # Vx_train & label
    Vx_train = np.empty((Num[0], row, column, 1), dtype="float16")
    # Vy_train & label
    Vy_train = np.empty((Num[0], row, column, 1), dtype="float16")
    # label
    train_label = np.empty((Num[0],), dtype="uint8")
    '''train'''
    # Vx_train & label
    Vx_test = np.empty((Num[1], row, column, 1), dtype="float16")
    # Vy_train & label
    Vy_test = np.empty((Num[1], row, column, 1), dtype="float16")
    # label
    test_label = np.empty((Num[1],), dtype="uint8")

    # import data
    for i in range(Num[0]):
    	pass
    	arr_x = FullData["train"][0][0][:,:,i]
    	arr_y = FullData["train"][1][0][:,:,i]
  
        Vx_train[i,:,:,:] = np.asarray(arr_x.reshape(arr_x.shape[0], arr_x.shape[1], 1), dtype="float16")
        Vy_train[i,:,:,:] = np.asarray(arr_y.reshape(arr_y.shape[0], arr_y.shape[1], 1), dtype="float16")
    	train_label[i] = FullData["train"][2][0][i] - 1
    for i in range(Num[1]):
    	pass
    	arr_x = FullData["test"][0][0][:,:,i]
    	arr_y = FullData["test"][1][0][:,:,i]
    	# Vx_test[i,:,:] = arr_x
    	# Vy_test[i,:,:] = arr_y
        Vx_test[i,:,:,:] = np.asarray(arr_x.reshape(arr_x.shape[0], arr_x.shape[1], 1), dtype="float16")
        Vy_test[i,:,:,:] = np.asarray(arr_y.reshape(arr_y.shape[0], arr_y.shape[1], 1), dtype="float16")
    	test_label[i] = FullData["test"][2][0][i] - 1
    print "OP Data processing completed!"
    op_train = {"Vx":Vx_train, "Vy":Vy_train, "Label":train_label}
    op_test = {"Vx":Vx_test, "Vy":Vy_test, "Label":test_label}

    return op_train, op_test

def All_Epoch():
    '''Return the Epoch nums of training'''
    dataPath = "/home/assassinator/Exp/Convolutional Neural Network/MySelf/3D_CNN/Data"
    trainPath = "/train/"
    # store the Path
    rootPath = dataPath + trainPath
    epoch_all = os.listdir(rootPath)
    # return all epoches & rootPath
    return epoch_all, rootPath

def Batch_LoadData(DataPath, EpochName):
    '''Batch func for loading data from the given path'''
    pass
    # batch Path
    data_batch = DataPath + EpochName
    # read all .mat data
    Data = os.listdir(data_batch)
    num = len(Data)

    # store the data
    store_data_R = np.empty((num, 180, 240, 7), dtype="float16")
    store_data_G = np.empty((num, 180, 240, 7), dtype="float16")
    store_data_B = np.empty((num, 180, 240, 7), dtype="float16")

    store_data_label = np.empty((num,), dtype="uint8")
    
    for i in range(num):
        pass
        # get each .mat 
        # print ("Current is %s" % Data[i])
        store_data_label[i] = int(Data[i].split('_')[0]) - 1
        try:
            each_data = scio.loadmat(data_batch + '/' + Data[i])
            # store_data_label[i] = int(Data[i].split('_')[0]) - 1
            for index in range(7):
                pass
                # load R & G & B channel
                store_data_R[i,:,:,index] = each_data['ImgCube_R'][index,:,:]
                store_data_G[i,:,:,index] = each_data['ImgCube_G'][index,:,:]
                store_data_B[i,:,:,index] = each_data['ImgCube_B'][index,:,:]
        except Exception:
            print ("Data %s is corrupted" % Data[i])           
        
    # output
    data_out = {"R":store_data_R, "G":store_data_G, "B":store_data_B}
    return data_out, store_data_label
    
# AllEpochs, DataPath = All_Epoch()
# dataPath = "/home/assassinator/Exp/Convolutional Neural Network/MySelf/3D_CNN/Data"
# test, test_label = Batch_LoadData(dataPath, '/train/epoch_4')
# print AllEpoches
# print test["R"].shape
# load_OP_data()
# from keras.utils import np_utils, generic_utils
# data, label = load_data()
# Y_train = label["train"]
# Y_train = np_utils.to_categorical(Y_train, 10)

# print Y_train.shape[0], Y_train.shape[1]
