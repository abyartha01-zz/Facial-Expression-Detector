# %% Importing libraries
import numpy as np                                                              #for using the np.load function to load the 
                                                                                #pre-processed data 
import matplotlib.pyplot as plt                                                 #to produce the graphs
import math                                                                     #to use mathematical functions
from keras.layers import Conv2D                                                 #for creating the convolution layers
from keras.layers import MaxPool2D                                              #for creating the pooling layers
from keras.layers import Flatten                                                #for creating the flattening layer
from keras.layers import Dense                                                  #for creating the dense layers
from keras.layers import Input                                                  #for creating the input layer
from keras.layers import concatenate                                            #to concatenate the layers
from keras.models import Model                                                  #finally to create the model
from keras.models import load_model                                             #to load a trained model
from keras.optimizers import SGD                                                #importing SGD optimizer
from keras.layers import AveragePooling2D                                       #for average pooling
from keras.layers import GlobalAveragePooling2D                                 #for global pooling
from keras.layers import Dropout                                                #for dropout layer
from keras.callbacks import LearningRateScheduler                               #to use the learning rate scheduler routine
from keras.callbacks import ModelCheckpoint                                     #to use the model checkpoint routine
 

# %% Importing pre-processed data

X_train = np.load("Pre-processed_data//X_train.npy")                                                #loaded X_train
Y_train = np.load("Pre-processed_data//Y_train.npy")                                                #loaded Y_train 
X_test = np.load("Pre-processed_data//X_test.npy")                                                  #loaded X_test
Y_test = np.load("Pre-processed_data//Y_test.npy")                                                  #loaded Y_train            
                           
# %% Defining a function to return a inception module
#    with required number of filters

def inception_module(layer, filter_1x1, filter_3x3_reduce, filter_3x3, 
                     filter_5x5_reduce, filter_5x5, filter_pool, name=None):      
    
    conv_1x1 = Conv2D(filter_1x1, (1, 1), padding='same',                       #created a 1x1 convolution layer 
                      activation='relu')(layer)                                 #connected with the previous layer
    
    conv_3x3 = Conv2D(filter_3x3_reduce, (1, 1), padding='same',                #created a 1x1 convolution layer
                      activation='relu')(layer)                                 #connected with the previous layer
    conv_3x3 = Conv2D(filter_3x3, (3, 3), padding='same',                       #created a 3x3 convolution layer
                      activation='relu')(conv_3x3)                              #connected to the previous 1x1 convolution
                                                                                #layer                                                                               
    conv_5x5 = Conv2D(filter_5x5_reduce, (1, 1), padding='same',                #created a 1x1 convolution layer    
                      activation='relu')(layer)                                 #connected with the previous layer    
    conv_5x5 = Conv2D(filter_5x5, (5, 5), padding='same',                       #created a 5x5 convolution layer 
                      activation='relu')(conv_5x5)                              #connected to the previous 1x1 convolution    
                                                                                #layer                                                                              
    pool = MaxPool2D((3, 3), padding='same', strides=(1, 1))(layer)             #created a max pool layer connected with
                                                                                #the previous layer
    pool = Conv2D(filter_pool, (1, 1), padding='same',                          #created a 1x1 convolution layer connected
                  activation='relu')(pool)                                      #to the previous pooling layer    
    
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool],                  #concatenated all the layers
                         axis=3, name=name)
    
    return output                                                               #returned the concatenated layers            


# %% Creating the inception network
    
input_layer = Input(shape=(48, 48, 1))                                          #defined the input layer
                                                           
layer = Conv2D(64, (7, 7), strides=(2, 2), padding='same',                      #7x7 convolution layer which is
               activation='relu', name='conv_1_7x7/2')(input_layer)             #connected to the input layer

layer = MaxPool2D((3, 3), strides=(2, 2), padding='same',                       #first maxpool layer connected 
                  name='max_pool_1_3x3/2')(layer)                               #to the 'conv_1_7x7/2' layer 

layer= Conv2D(64, (1, 1), strides=(1, 1), padding='same',                       #1x1 convolution layer which is
              activation='relu', name='conv_2a_1x1/1')(layer)                   #connected to the 'max_pool_1_3x3/2' layer

layer = Conv2D(192, (3, 3), strides=(1, 1), padding='same',                     #3x3 convolutional layer which is
               activation='relu', name='conv_2b_3x3/1')(layer)                  #connected to the 'conv_2a_1x1/1' layer

layer = MaxPool2D((3, 3), strides=(2, 2), padding='same',                       #maxpool layer connected to the
                  name='max_pool_2_3x3/2')(layer)                               #'conv_2b_3x3/1' layer

layer = inception_module(layer, 64, 96, 128, 16, 32, 32, 'inception_3a')        #first inception modeule connected
                                                                                #to the 'max_pool_2_3x3/2' layer
                                                                                
layer = inception_module(layer, 128, 128, 192, 32, 96, 64, 'inception_3b')      #second inception modeule connected
                                                                                #'inception_3a' inception module    
                                                                                
layer = MaxPool2D((3, 3), strides=(2, 2), padding='same',                       #maxpool layer connected to the
                  name='max_pool_3_3x3/2')(layer)                               #'inception_3b' inception module    

layer = inception_module(layer, 192, 96, 208, 16, 48, 64,                       #third inception modeule connected
                         name='inception_4a')                                   #to the 'max_pool_3_3x3/2' layer

#First auxillary output
layer_1 = AveragePooling2D((5, 5), strides=(3, 3), padding='same')(layer)       #average pooling layer connected 
                                                                                #to the 'inception_4a' module            
layer_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(layer_1)       #1x1 convolution layer connected 
                                                                                #to the previous average pooling layer            
layer_1 = Flatten()(layer_1)                                                    #flattening layer connected to the 
                                                                                #previous convolution layer        
layer_1 = Dense(1024, activation='relu')(layer_1)                               #dense layer connected to the 
                                                                                #previous flattening layer
layer_1 = Dropout(0.7)(layer_1)                                                 #dropout to prevent overfitting
                                                                                #connected to the previous dense layer    
layer_1 = Dense(7, activation='softmax', name='auxillary_output_1')(layer_1)    #first auxillary output layer connected
                                                                                #the previous dropout layer   

layer = inception_module(layer, 160, 112, 224, 24, 64, 64, 'inception_4b')      #fourth inception modeule connected
                                                                                #to the 'inception_4a' module
                                                                                
layer = inception_module(layer, 128, 128, 256, 24, 64, 64, 'inception_4c')      #fifth inception modeule connected
                                                                                #'inception_4b' inception module 
                                                                                
layer = inception_module(layer, 112, 14, 288, 32, 64, 64, 'inception_4d')       #sixth inception modeule connected
                                                                                #'inception_4c' inception module 

#Second auxillary output
layer_2 = AveragePooling2D((5, 5), strides=(3, 3), padding='same')(layer)       #average pooling layer connected 
                                                                                #to the 'inception_4d' module  
layer_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(layer_2)       #1x1 convolution layer connected 
                                                                                #to the previous average pooling layer       
layer_2 = Flatten()(layer_2)                                                    #flattening layer connected to the 
                                                                                #previous convolution layer
layer_2 = Dense(1024, activation='relu')(layer_2)                               #dense layer connected to the 
                                                                                #previous flattening layer
layer_2 = Dropout(0.7)(layer_2)                                                 #dropout to prevent overfitting
                                                                                #connected to the previous dense layer
layer_2 = Dense(7, activation='softmax', name='auxillary_output_2')(layer_2)    #second auxillary output layer connected
                                                                                #the previous dropout layer        

layer = inception_module(layer, 256, 160, 320, 32, 128, 128, 'inception_4e')    #seventh inception modeule connected
                                                                                #'inception_4d' inception module 

layer = MaxPool2D((3, 3), strides=(2, 2), padding='same',                       #maxpool layer connected to the
                  name='max_pool_4_3x3/2')(layer)                               #inception_4e' inception module

layer = inception_module(layer, 256, 160, 320, 32, 128, 128, 'inception_5a')    #eigth inception modeule connected
                                                                                #'max_pool_4_3x3/2' layer
                                                                                
layer = inception_module(layer, 384, 192, 384, 48, 128, 128, 'inception_5b')    #eigth inception modeule connected
                                                                                #'inception_5a' inception module    

layer = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(layer)                  #global average pooling layer connected
                                                                                #to the 'inception_5b' inception module   
                                                                                
layer = Dropout(0.4)(layer)                                                     #dropout layer to prevent overfitting
                                                                                #connected to the 'avg_pool_5_3x3/1' layer    

layer = Dense(1024)(layer)                                                      #linear dense layer connected to     
                                                                                #the previous droopout layer
                                                                                
layer = Dense(7, activation='softmax', name='output')(layer)                    #final softmax output layer connected
                                                                                #to the previous dense layer
                                                                                
model = Model(input_layer, [layer, layer_1, layer_2], name='inception_v1')      #finally created the model by joining 
                                                                                #the input layer and the auxillary output
                                                                                #layers and setting the name as 'inception_v1'    

model.summary()                                                                 #printing the model summary


# %% Defining the method to decrease learning rate after every 5 epochs

epochs = 150                                                                    #defined the number of epochs as 150
initial_lrate = 0.001                                                           #defined the learning rate as 0.001

checkpoint = ModelCheckpoint('Facial_expression_detector_InceptionNet.h5',      #defined a model checkpoint routine
                             monitor='val_loss', mode='min', verbose=1,         #which will save the model when the    
                             save_best_only=True)                               #validation loss is minimum    

def decay(epoch, steps=100):                                                    #defined a function which will     
    initial_lrate = 0.001                                                       #decrease the learning rate after
    drop = 0.96                                                                 #every 5 epochs    
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    return lrate

sgd = SGD(lr=initial_lrate)                                                     #defined the optimizer with the 
                                                                                #learning rate    

lr_sc = LearningRateScheduler(decay, verbose=1)                                 #defined the learning rate scheduler
                                                                                #which will call the decay function

# %% Compiling the model

model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy',     #compiled the model with the categorical
                    'categorical_crossentropy'],                                #crossentropy as the loss function for all
              optimizer=sgd, metrics=['accuracy'])                              #three outputs, and the previously defined    
                                                                                #optimiser, and the only metric used is
                                                                                #accuracy

# %% Training the model

history = model.fit(X_train, [Y_train, Y_train, Y_train], batch_size=32,        #trained the model with the augmented data       
          validation_data=(X_test, [Y_test, Y_test, Y_test]),                   #for all the three outputs with a batch
          epochs=epochs, callbacks=[lr_sc, checkpoint])                         #size of 32, and the test set as the validation 
                                                                                #data, epochs set as the previously defined 
                                                                                #number of epochs, and callbacks are the 
                                                                                #learning rate scheduler and the model 
                                                                                #checkpoint routine
                                                                                

# %% Plotting model history

#Summarize history for accuracy 
plt.plot(history.history['output_acc'])                                         #plotted training accuracy
plt.plot(history.history['val_output_acc'])                                     #plotted validation accuracy
plt.title('model accuracy')                                                     #set the title of the graph
plt.ylabel('accuracy')                                                          #set the y-label
plt.xlabel('epoch')                                                             #set the x-label
plt.legend(['train', 'validation'], loc='upper left')                           #set the legend to show on the upper left corner    
plt.show()                                                                      #to display the graph    

#Summarize history for loss
plt.plot(history.history['output_loss'])                                        #plotted training loss
plt.plot(history.history['val_output_loss'])                                    #plotted validation loss
plt.title('model loss')                                                         #set the title of the graph    
plt.ylabel('loss')                                                              #set the x-label
plt.xlabel('epoch')                                                             #set the y-label
plt.legend(['train', 'validation'], loc='upper left')                           #set the legend to show on the upper left corner
plt.show()                                                                      #to display the graph   

# %% Saving the trained model
    
model.save("Facial_expression_detector_InceptionNet.h5")                       #to save the trained model

# %% Loading the trained model

model = load_model("Facial_expression_detector_InceptionNet.h5",                #to load the trained model
                   compile=False)            
