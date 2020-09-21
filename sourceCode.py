#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib
import sklearn
import tensorflow as tf
import collections

from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import tree
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import metrics
from matplotlib import pyplot as plt
from scipy import misc


# In[ ]:


# Functions For Building Deep Neural Networks


# In[ ]:


def newCNNModel(input_shape=(8,8,1), input_nodes=32, output_nodes=10):
    model = Sequential() #use sequential model
    model.add(Conv2D(input_nodes, kernel_size=(3, 3), input_shape=input_shape, activation='relu')) #add convolutional layer  
    model.add(MaxPooling2D(pool_size=(2, 2))) #add pooling layer
    model.add(Flatten()) #add flatten layer
    model.add(Dense(output_nodes, activation='softmax')) #add dense layer
    return model

def compileModel(model, loss='categorical_crossentropy', optimizer='adam'):
    model.compile(
        loss=loss, #use 'categorical_crossentropy' as loss
        optimizer=optimizer, #use 'adam' as optimizer
        metrics=['accuracy'] # use 'accuracy' as list of metrics to be evaluated by the model during training and testing
    )
    return model

def trainModel(model, x, y):
    model_history = model.fit(
    x, 
    y,
    batch_size=10, #set 10 as the number of samples per gradient update
    epochs=100000, #set 50 as number of epochs to train the model
    verbose=1 #set verbosity mode to progress bar
    )
    return model


# In[ ]:


# Data Preperation


# In[ ]:


#Load Optical recognition of handwritten digits dataset
digits = datasets.load_digits()
x = digits.images
y = digits.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=0) # split train and test dataset

num_test = int(y_test.size)


# In[ ]:


#Preprocess data for training deep neural networks

#Reduce the dimension of those 2-dimensional pixels; then convert pixels to floats which are between 0-1
x_train = x_train.reshape(1257, 8, 8, 1).astype('float32')/255 #
x_test = x_test.reshape(540, 8, 8, 1).astype('float32')/255
#transfer the labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# In[7]:


# Make modifications to the training dataset


# In[8]:


#Shuffle pixels

y_train_shuffled = y_train
np.random.shuffle(y_train_shuffled)


# In[9]:


# CNN Training


# In[10]:


cnn_model = compileModel(newCNNModel());
cnn_history = trainModel(cnn_model, x_train, y_train)
cnn_model.save('cnn_model')


# In[ ]:


# Functions 


# In[ ]:


# function for confusion matrix

#transfer the format of the original dataset labels to an usable array
def transferValue(y_test):
    real_y_test = np.zeros(int(num_test))
    q = 0
    while q < num_test:
        p = 0
        while p < 10:
            if y_test[q][p] == 1:
                real_y_test[q] = p
            p += 1
        q += 1
    return real_y_test


# function for ROC


def computePRs(value,prediction,threshold):
    #use the shresholding to compute the final prediction class
    pre_Class = np.zeros(int(num_test))
    a = 0
    while a < num_test:
        if prediction[a] >= threshold:
            pre_Class[a] = 1
        else:
            pre_Class[a] == 0
        a += 1
    
    #calculate PRs
    tp, fp, tn, fn = 0, 0, 0, 0
    for pre, truth in zip(pre_Class, value):
        if pre:                   
            if truth:                              # actually positive 
                tp += 1
            else:                                  # actually negative              
                fp += 1          
        else:                                      # predicted negative 
            if not truth:                          # actually negative 
                tn += 1                          
            else:                                  # actually positive 
                fn += 1 
    
    TPR = tp / (tp + fn)
    FPR = fp / (fp + tn)
    return TPR,FPR


def drawROC(value,prediction):
    value_binary = np.zeros(int(num_test))
    pre_binary = np.zeros(int(num_test))
    
    #use the label "5" as positive label: transfer test set to binary classification
    a = 0
    while a < num_test:
        if value[a] == 5:
            value_binary[a] = 1
        else:
            value_binary[a] = 0
        a += 1
    #use the label "5" as positive label: transfer prediction set to binary classification
    b = 0
    while b < num_test:   
        pre_binary[b] = prediction[b][5]
        b += 1
 
    
    TPR0, FPR0 = computePRs(value_binary,pre_binary,0)
    TPR1, FPR1 = computePRs(value_binary,pre_binary,0.1)
    TPR2, FPR2 = computePRs(value_binary,pre_binary,0.2)
    TPR3, FPR3 = computePRs(value_binary,pre_binary,0.3)
    TPR4, FPR4 = computePRs(value_binary,pre_binary,0.4)
    TPR5, FPR5 = computePRs(value_binary,pre_binary,0.5)
    TPR6, FPR6 = computePRs(value_binary,pre_binary,0.6)
    TPR7, FPR7 = computePRs(value_binary,pre_binary,0.7)
    TPR8, FPR8 = computePRs(value_binary,pre_binary,0.8)
    TPR9, FPR9 = computePRs(value_binary,pre_binary,0.9)
    TPR10, FPR10 = computePRs(value_binary,pre_binary,1)
    
    TPR = [TPR0, TPR1, TPR2, TPR3, TPR4, TPR5, TPR6, TPR7, TPR8, TPR9, TPR10]
    FPR = [FPR0, FPR1, FPR2, FPR3, FPR4, FPR5, FPR6, FPR7, FPR8, FPR9, FPR10]
    
    plt.subplots(figsize=(7,5.5));
    plt.plot(FPR, TPR, color='darkorange',
             lw=2, label='ROC curve');
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('ROC Curve');
    plt.legend(loc="lower right");
    plt.show()


# In[ ]:


# Cross Validation: Dataset Preparation for two  neural networks


# In[ ]:


numSample = int(1257/5) #calculate the number of entries in each sample

#apply k-fold cross validation
#divide and produce each validation dataset
x_valid1 = x_train[0:numSample] 
x_valid2 = x_train[numSample:numSample*2]
x_valid3 = x_train[numSample*2:numSample*3]
x_valid4 = x_train[numSample*3:numSample*4]
x_valid5 = x_train[numSample*4:1258]

x_train1 = x_train[numSample:1258]
x_train2 = np.concatenate((x_valid1, x_train[numSample*2:1258]), axis=0)
x_train3 = np.concatenate((x_valid1, x_valid2, x_train[numSample*3:1258]), axis=0)
x_train4 = np.concatenate((x_valid1, x_valid2, x_valid3, x_train[numSample*4:1258]), axis=0)
x_train5 = np.concatenate((x_valid1, x_valid2, x_valid3, x_valid4), axis=0)

y_valid1 = y_train[0:numSample]
y_valid2 = y_train[numSample:numSample*2]
y_valid3 = y_train[numSample*2:numSample*3]
y_valid4 = y_train[numSample*3:numSample*4]
y_valid5 = y_train[numSample*4:1258]

y_train1 = y_train[numSample:1258]
y_train2 = np.concatenate((y_valid1, y_train[numSample*2:1258]), axis=0)
y_train3 = np.concatenate((y_valid1, y_valid2, y_train[numSample*3:1258]), axis=0)
y_train4 = np.concatenate((y_valid1, y_valid2, y_valid3, y_train[numSample*4:1258]), axis=0)
y_train5 = np.concatenate((y_valid1, y_valid2, y_valid3, y_valid4), axis=0)


# In[ ]:


# Cross Validation: CNN Model


# In[ ]:


cnn_model1 =  keras.models.load_model('cnn_model') #load the model
cnn_history1 = trainModel(cnn_model1, x_train1, y_train1) #use the pre-divided training dataset to train
pre1 = cnn_history1.predict(x_valid1, batch_size=1) #use the pre-divided validatioon dataset to predict
acc1 = sklearn.metrics.accuracy_score(np.argmax(y_valid1, axis=1),np.argmax(pre1, axis=1)) #calculate the accuracy

cnn_model2 =  keras.models.load_model('cnn_model')
cnn_history2 = trainModel(cnn_model2, x_train2, y_train2)
pre2 = cnn_history2.predict(x_valid2, batch_size=1)
acc2 = sklearn.metrics.accuracy_score(np.argmax(y_valid2, axis=1),np.argmax(pre2, axis=1))

cnn_model3 =  keras.models.load_model('cnn_model')
cnn_history3 = trainModel(cnn_model3, x_train3, y_train3)
pre3 = cnn_history3.predict(x_valid3, batch_size=1)
acc3 = sklearn.metrics.accuracy_score(np.argmax(y_valid3, axis=1),np.argmax(pre3, axis=1))

cnn_model4 =  keras.models.load_model('cnn_model')
cnn_history4 = trainModel(cnn_model4, x_train4, y_train4)
pre4 = cnn_history4.predict(x_valid4, batch_size=1)
acc4 = sklearn.metrics.accuracy_score(np.argmax(y_valid4, axis=1),np.argmax(pre4, axis=1))

cnn_model5 =  keras.models.load_model('cnn_model')
cnn_history5 = trainModel(cnn_model5, x_train5, y_train5)
pre5 = cnn_history5 .predict(x_valid5, batch_size=1)
acc5 = sklearn.metrics.accuracy_score(np.argmax(y_valid5, axis=1),np.argmax(pre5, axis=1))

AverageAcc = (acc1+acc2+acc3+acc4+acc5)/5
print("The average accuracy of 5-subsamples cross validation on CNN model is", AverageAcc) #calculate the average of accuracies


# In[ ]:


# Confusion Matrix for CNN


# In[ ]:


cnn_model_confusion =  keras.models.load_model('cnn_model')
y_pre = cnn_model_confusion.predict_classes(x_test)
confusion_Matrix_cnn = np.zeros((10, 10))
real_y_test = transferValue(y_test) 

a = 0

while a < num_test:
    b = 0
    while b < 10:
        if b == y_pre[a]:
            c = 0
            while c < 10:
                if c == real_y_test[a]:
                    confusion_Matrix_cnn[b][c] += 1 # for multiple classes classification, construct a 10*10 matrix
                c += 1
        b += 1
    a += 1
print("The confusion matrix of CNN model is:")
print("The x-axis is for the prediction values, and y-axis is for the real values of samples. Values are 0-9 increasing by order")
print(confusion_Matrix_cnn)    


# In[ ]:


# ROC curve for CNN


# In[ ]:


# prediction
cnn_model_ROC =  keras.models.load_model('cnn_model')
yROC_pre = cnn_model_ROC.predict(x_test, batch_size = 1)
real_y_test = transferValue(y_test)
drawROC(real_y_test,yROC_pre)

