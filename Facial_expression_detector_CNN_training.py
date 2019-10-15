# %% Importing libraries
from keras.models import Sequential                                     #to create a sequential model
from keras.layers import Conv2D                                         #for creating the convolution layers
from keras.layers import MaxPool2D                                      #for creating the pooling layers
from keras.layers import Flatten                                        #for creating the flattening layer
from keras.layers import Dense                                          #for creating the dense layers
from keras.layers import BatchNormalization                             #for creating the batch normalization layer
from keras.layers import Dropout                                        #for creating the droupout layers
from keras.regularizers import l2                                       #for using l2 regularization 
import numpy as np                                                      #for using the np.load function to load the 
                                                                        #pre-processed data 
import matplotlib.pyplot as plt                                         #to produce the graphs
from keras.models import load_model                                     #to load a trained model


# %% Importing pre-processed data

X_train = np.load("Pre-processed_data//X_train.npy")                    #loaded X_train
Y_train = np.load("Pre-processed_data//Y_train.npy")                    #loaded Y_train 
X_test = np.load("Pre-processed_data//X_test.npy")                      #loaded X_test
Y_test = np.load("Pre-processed_data//Y_test.npy")                      #loaded Y_train            

# %% Creating the model

# Defined the initial number of filters and number of output labels
num_features = 64
num_labels = 7

# Model creation started
model = Sequential()                                                    #created a stack which will contain     
                                                                        #the layers
#Created three blocks of convolution, max pooling, 
#batch normalization and dropouts in the same order with 
#increasing number of features in the convolution layers
                                                                    
#First block                                                                    
model.add(Conv2D(num_features, kernel_size=(3, 3),                      #added a convolution layer with 64 filters
                        activation='relu', input_shape=(48, 48, 1),     #and l2 regularizer
                        kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())                                         #added a batch normalization layer for normalizing
                                                                        #the weights
model.add(Conv2D(num_features, kernel_size=(3, 3),                      #added another convolution layer with 64 filters
                        activation='relu', padding='same'))
model.add(BatchNormalization())                                         #added a batch normalization layer for normalizing
                                                                        #the weights
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))                  #max pool layer with size and strides of (2, 2)
model.add(Dropout(0.5))                                                 #added a dropout layer to prevent overfitting

#Second block
model.add(Conv2D(2*num_features, kernel_size=(3, 3),                    #added a convolution layer with 128 filters
                        activation='relu', padding='same'))
model.add(BatchNormalization())                                         #added a batch normalization layer for normalizing
                                                                        #the weights                
model.add(Conv2D(2*num_features, kernel_size=(3, 3),                    #added another convolution layer with 128 filters
                        activation='relu', padding='same'))
model.add(BatchNormalization())                                         #added a batch normalization layer for normalizing
                                                                        #the weights                       
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))                  #max pool layer with size and strides of (2, 2)
model.add(Dropout(0.5))                                                 #added a dropout layer to prevent overfitting

#Third block
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3),                  #added a convolution layer with 256 filters
                        activation='relu', padding='same'))
model.add(BatchNormalization())                                         #added a batch normalization layer for normalizing
                                                                        #the weights        
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3),                  #added another convolution layer with 256 filters
                        activation='relu', padding='same'))
model.add(BatchNormalization())                                         #added a batch normalization layer for normalizing
                                                                        #the weights     
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))                  #max pool layer with size and strides of (2, 2)
model.add(Dropout(0.5))                                                 #added a dropout layer to prevent overfitting

#Flattening of the data
model.add(Flatten())

#Created three fully connected layers with dropout layers in 
#between to prevent overfitting

model.add(Dense(2*2*2*num_features, activation='relu'))                 #first dense layer with 512 units
model.add(Dropout(0.4))                                     
model.add(Dense(2*2*num_features, activation='relu'))                   #first dense layer with 256 units
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))                     #first dense layer with 128 units
model.add(Dropout(0.5))

#Output layer 
model.add(Dense(num_labels, activation='softmax'))                      #output dense layer with required number of 
                                                                        #output labels    
#To print the model summary
model.summary()

# %% Compiling the model
model.compile(loss='categorical_crossentropy',                          #compiled the model with categorical crossentropy
              optimizer='adam',                                         #as the loss function and Adam as the optimizer
              metrics=['accuracy'])                                     #and the only metric used here is accuracy


# %% Training the model
history = model.fit(X_train, Y_train,                                   #training the model with the augmented data
                    batch_size=32,                                      #and batch size of 32,
                    validation_data=(X_test, Y_test),                   #using the test set as validation set,                                                     
                    epochs=100)                                         #and no. of eopchs set to 100
                                                                       
                                                                                                                                 
# %% Plotting model history

#Summarize history for accuracy 
plt.plot(history.history['acc'])                                        #plotted training accuracy
plt.plot(history.history['val_acc'])                                    #plotted validation accuracy
plt.title('model accuracy')                                             #set the title of the graph
plt.ylabel('accuracy')                                                  #set the y-label
plt.xlabel('epoch')                                                     #set the x-label
plt.legend(['train', 'validation'], loc='upper left')                   #set the legend to show on the upper left corner    
plt.show()                                                              #to display the graph    

#Summarize history for loss
plt.plot(history.history['loss'])                                       #plotted training loss
plt.plot(history.history['val_loss'])                                   #plotted validation loss
plt.title('model loss')                                                 #set the title of the graph    
plt.ylabel('loss')                                                      #set the x-label
plt.xlabel('epoch')                                                     #set the y-label
plt.legend(['train', 'validation'], loc='upper left')                   #set the legend to show on the upper left corner
plt.show()                                                              #to display the graph    

# %% Saving the trained model
    
model.save("Facial_expression_detector_CNN.h5")                         #to save the trained model

# %% Loading the trained model

#model = load_model("Facial_expression_detector_CNN.h5",                #to load the trained model 
#                    compile=False)    