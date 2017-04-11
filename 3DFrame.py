#coding:utf-8
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import merge, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range


'''From given 3D_CNN folder load data'''
# import sys for load data
import sys
# *** linux
sys.path.append('/home/assassinator/Exp/Convolutional Neural Network/Myself/3D_CNN/')
# get Epoches
from load_data import All_Epoch, Batch_LoadData
AllEpochs, DataPath = All_Epoch()
Ep_Nums = len(AllEpochs)

input_shape = (180, 240, 7)

########################################################
#                   开始建立CNN模型
#     Construct Merged Model : merge the R & G & B 
########################################################

'''Created Shared Weight Layer'''
def SharedLayer(input_dims):
	:pass
    Shared = Sequential()
    # input : kernel size(7, 7)
    # Conv1 : conv the R & G & B with shared layer
    #         the result size is : 174, 234, nums is 3 * 4
    # 1. Shared.add(Conv2D(4,(7, 7), padding='valid', input_shape=input_dims))
    # TODO : 3-29
    Shared.add(Conv2D(7, (7, 7), padding='valid', input_shape=input_dims)) 
    Shared.add(Activation('relu'))
    # DownSampling : 
    # the result size is 58, 78, nums is 3 * 4
    Shared.add(MaxPooling2D(pool_size=(3,3)))
    Shared.add(Dropout(0.2))
    
    # Conv2 : conv the merged tensor with kernel(5, 7)
    #         the result size is: 54, 72, nums is 3 * 4 * 4
    # 1. Shared.add(Conv2D(4, (5, 7)))
    # TODO : 3-29
    Shared.add(Conv2D(6, (5, 7)))
    Shared.add(Activation('relu'))
    # DownSampling : 
    # the result size is 18, 24, nums is 3 * 4 * 4
    Shared.add(MaxPooling2D(pool_size=(3,3)))
    Shared.add(Dropout(0.2))

    # Conv3 : con the tensor with kernel(4, 5)
    #         the result size is 15, 21, nums is 3 * 4 * 4 * 3
    # 1. Shared.add(Conv2D(3, (4, 4)))
    # TODO : 3-29
    Shared.add(Conv2D(5, (4, 4)))
    Shared.add(Activation('relu'))
    # DownSampling : 
    # the result size is 5, 7, nums is 3 * 4 * 4 * 3
    Shared.add(MaxPooling2D(pool_size=(3,3)))
    Shared.add(Dropout(0.2))

    # Conv4 : conv the tensor with kernel
    #         the resul size is 1, 1, nums is 3 * 4 * 4 * 3 * 2
    # 1. Shared.add(Conv2D(2, (5, 7)))
    # TODO : 3-29
    Shared.add(Conv2D(3, (5, 7)))
    Shared.add(Activation('relu'))
    Shared.add(Dropout(0.2))

    # Flat
    Shared.add(Flatten())
    Shared.add(Dense(128))
    Shared.add(Activation('relu'))
    Shared.add(Dropout(0.2))

    # return shared layer
    return Shared

'''Create 3D CNN Shared Layer'''
def 3D_Shared_Layer(input_dims, Frequency):
	pass
	# reconstruct the input_shape
	input_dims = input_dims + (Frequency,)
	Shared = Sequential()
    # input : kernel size(7, 7)
    # Conv3D 1 : conv the R & G & B with shared layer
    #            the result size is : 174, 234, dims is 5, nums is 3 * 4
	# TODO : 4-07
	Shared.add(Conv3D(7, (7, 7, 3)), padding='valid', input_shape=input_dims)
	Shared.add(Activation('relu'))
    # DownSampling : 
    # the result size is 58, 78, dims is 5 ,nums is 3 * 4
    Shared.add(MaxPooling2D(pool_size=(3, 3))) # maxpooling 3D
    Shared.add(Dropout(0.2))
    
    # Conv3D 2 : conv the merged tensor with kernel(5, 7)
    #            the result size is: 54, 72, dims is 3, nums is 3 * 4 * 4
	# TODO : 4-07
	Shared.add(Conv3D(6, (5, 7, 3)))
	Shared.add(Activation('relu'))
	# DownSampling :
    # the result size is 18, 24, dims is 3, nums is 3 * 4 * 4
	Shared.add(MaxPooling2D(pooling_size=(3, 3)))
	Shared.add(Dropout(0.2))

	# Conv3D 3 : conv the tensor with kernel size(4, 4, 2)
    #            the result size is 15, 21, dims is 2,  nums is 3 * 4 * 4 * 3
	# TODO : 4-07
    Shared.add(Conv3D(5, (4, 4, 2)))
    Shared.add(Activation('relu'))
    # DownSampling : 
    # the result size is 5, 7, dims is 2,  nums is 3 * 4 * 4 * 3
    Shared.add(MaxPooling2D(pool_size=(3,3)))
    Shared.add(Dropout(0.2))

    # Conv4 : conv the tensor with kernel (5, 7 ,2)
    #         the resul size is 1, 1, dims is 1,  nums is 3 * 4 * 4 * 3 * 2
    # TODO : 4-07
    Shared.add(Conv3D(3, (5, 7, 1)))
    Shared.add(Activation('relu'))
    Shared.add(Dropout(0.2))

    # Flat
    Shared.add(Flatten())
    Shared.add(Dense(128))
    Shared.add(Activation('relu'))
    Shared.add(Dropout(0.2))

	# return constructed Shared layer
	return Shared
'''*********************************************************************************'''
'''Construct the Merged 3D model'''
def Create3DCNN(input_shape):
    pass
    BaseLayer = SharedLayer(input_shape)
    
    # first Input layer
    R_Input = Input(shape=(input_shape))
    G_Input = Input(shape=(input_shape))
    B_Input = Input(shape=(input_shape))
    
    # Conv1 : conv the R & G & B with shared layer
    #         the result size is : 174, 234, nums is 3 * 4
    R_out = BaseLayer(R_Input)
    G_out = BaseLayer(G_Input)
    B_out = BaseLayer(B_Input)
    ## ************************************************************************
    ## Merge
    ## Merged = Merge([R_out, G_out, B_out], mode='concat')
    ## X = merge([R_out, G_out, B_out], mode='concat')
    ## X_out = MaxPooling2D(pool_size=(3,3))
    ## model = Model(X, X_out)
    ## model = Sequential()
    ## model.add(Merged)
    
    ## DownSampling : 
    ## the result size is 58, 78, nums is 3 * 4
    ## model.add(MaxPooling2D(pool_size=(3,3)))
    ## model.add(Dropout(0.2))

    ## Conv2 : conv the merged tensor with kernel(8, 8)
    ##         the result size is: 54, 72, nums is 3 * 4 * 4
    ## model.add(Conv2D(4, (5, 7)))
    ## model.add(Activation('relu'))
    
    ## DownSampling : 
    ## the result size is 18, 24, nums is 3 * 4 * 4
    ## model.add(MaxPooling2D(pool_size=(3,3)))
    ## model.add(Dropout(0.2))
    ## Conv3 : con the tensor with kernel(4, 5)
    ##         the result size is 15, 21, nums is 3 * 4 * 4 * 3
    ## model.add(Conv2D(3, (4, 4)))
    ## model.add(Activation('relu'))

    ## DownSampling : 
    ## the result size is 5, 7, nums is 3 * 4 * 4 * 3
    ## model.add(MaxPooling2D(pool_size=(3,3)))
    ## model.add(Dropout(0.2))

    ## Conv4 : conv the tensor with kernel
    ##         the resul size is 1, 1, nums is 3 * 4 * 4 * 3 * 2
    ## model.add(Conv2D(2, (5, 7)))
    ## model.add(Activation('relu'))
    ## model.add(Dropout(0.2))
    ## Flat
    ## model.add(Flatten())
    ## model.add(Dense(128))
    ## model.add(Activation('relu'))
    ## model.add(Dropout(0.2))
    # *******************************************************************************
    # final : softmax -- 10
    X = merge([R_out, G_out, B_out], mode='concat')
    OutPut = Dense(10, activation='softmax')(X)
    # OutPut = Dense(10, activation='softmax')([R_out, G_out, B_out])
    model = Model([R_Input, G_Input, B_Input], OutPut)

    return model

# Processing
model = Create3DCNN(input_shape)
# Ep_Nums
for i in range(Ep_Nums):
    pass
    print ("(%d): Processing................................................................"% (i+1))
    data_out, store_data_label = Batch_LoadData(DataPath, AllEpochs[i])
    store_data_label = np_utils.to_categorical(store_data_label, 10)
    # the first processing to create the model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.fit([data_out["R"] / 255, data_out["G"] / 255, data_out["B"] / 255], store_data_label, batch_size=100, epochs=10, verbose=1)   
'''Validation on test'''
# test data
testPath = '/home/assassinator/Exp/Convolutional Neural Network/MySelf/3D_CNN/Data'
test, test_label = Batch_LoadData(testPath, '/test/')
test_label = np_utils.to_categorical(test_label, 10)
# final validation
print ("Validation : .........................................................................")
score = model.evaluate([test["R"]/ 255, test["G"] / 255, test["B"] / 255], test_label, batch_size=100, verbose=0)
print ("score : %f"% score[0])
print ("score : %f"% score[1])

