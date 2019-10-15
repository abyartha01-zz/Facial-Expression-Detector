# %% Importing libraries
import pandas as pd                                                #for working with the datasets
from keras.utils import to_categorical                             #for applying one-hot encoding on the outputs
from keras.preprocessing.image import ImageDataGenerator           #for data augmentation 
import numpy as np                                                 #for certain operations on matrices and
                                                                   #using the np.save function to save the  
                                                                   #pre-prepreocessed data

# %% Creating necessary variables

# Creating column names 
columns = list(range(1, 2305))                                              #will be used in pre-processing 
columns  = list(map(str, columns))                                          #of the data                      

for i in range(len(columns)):
    columns[i] = "pixel" + columns[i]           
    

# %% Preparing the data    
    
# For ferc2013 dataset    
# Importing dataset
df = pd.read_csv('fer2013.csv')                                             #reading the dataset  

#Creating the training set
X_train = df.iloc[:28709, 1]                                                #extracting the training data (input)
for i in range(28709):
    X_train[i] = X_train[i].split()                                         #created a list of lists containing the
    X_train[i] = list(map(int, X_train[i]))                                 #training data in integer format    
                                                                      
X_train = pd.DataFrame(X_train)
X_train = pd.DataFrame(X_train['Pixels'].values.tolist(),                   #converted the list to a dataframe with
                       columns=columns)                                     #no. of columns same as no. of pixels                  
X_train = X_train.values.reshape(-1, 48, 48, 1)                             #reshaped the data into required shape
X_train = X_train.astype('float32')                                         #converted to float32 as it takes less memory
X_train = X_train / 255.0                                                   #scaled the data in the range 0-1

Y_train = df.iloc[:28709, 0]                                                #extracting the training data (output)
Y_train = to_categorical(Y_train, num_classes=7)                            #applying one-hot encoding  

#Creating the test set
X_test = df.iloc[28709:, 1]                                                 #extracting the test data (input)
for i in range(28709, 35887):
    X_test[i] = X_test[i].split()                                           #created a list of lists containing the  
    X_test[i] = list(map(int, X_test[i]))                                   #test data in integer format        

X_test = pd.DataFrame(X_test)
X_test = pd.DataFrame(X_test['Pixels'].values.tolist(),                     #converted the list to a dataframe with  
                      columns=columns)                                      #no. of columns same as no. of pixels
X_test = X_test.values.reshape(-1, 48, 48, 1)                               #reshaped the data into required shape  
X_test = X_test.astype('float32')                                           #converted to float32 as it takes less memory
X_test = X_test / 255.0                                                     #scaled the data in the range 0-1  

Y_test = df.iloc[28709:, 0]                                                 #extracting the test data (output)  
Y_test = to_categorical(Y_test, num_classes=7)                              #applying one-hot encoding  

# %% Data augmentation
gen = ImageDataGenerator(width_shift_range=0.1,                             #creating more data through augmentation  
                         height_shift_range=0.1,                            #to prevent overfitting 
                         zoom_range=0.1,
                         horizontal_flip=True)

temp = gen.flow(X_train, Y_train)

for i in range(len(temp)):                                                  #appending the augmented data                    
    for j in range(len(temp[i][0])):                                        #to the training data    
        X_train = np.append(arr=X_train, values=temp[i][0][j])
        Y_train = np.append(arr=Y_train, values=temp[i][1][j])
        
X_train = np.reshape(X_train, newshape=(-1, 48, 48, 1))
Y_train = np.reshape(Y_train, newshape=(-1, 7))

# %% Saving data
np.save("X_train", X_train)                                                 #saved X_train     
np.save("X_test", X_test)                                                   #saved X_test
np.save("Y_train", Y_train)                                                 #saved Y_train
np.save("Y_test", Y_test)                                                   #saved Y_test
                                                   