# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:29:16 2019

@author: Jack
"""


import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import datasets
import math
import copy as cp


def sigmoid(z):
    return 1/(1+np.exp(-z))


def Dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
    

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))


def dSoftmax(z):
    d_softmax = np.exp(z)/np.sum(np.exp(z)) - np.exp(z)/np.pow(np.sum(np.exp(z)), 2)
    return d_softmax


def stable_softmax(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)


def Dstable_softmax(z):
    
    p_i = stable_softmax(z)
    
    dP_i = []
    dP_i.append(-p_i[0]*p_i[1]) #when i /= j
    dP_i.append(p_i[0]*(1-p_i[1])) #when i = j
    dP_i = np.array(dP_i)
    
    return dP_i
    
    
def training_errors(y_output, y_actual):
    
    correct_matches = [a for a,b in zip(y_output, y_actual) if a == b]
    correct = len(correct_matches)
    errors = len(y_actual) - correct
    return errors/len(y_actual)
    
   
data = pd.read_csv("dataR2.csv")

'''

train, test = train_test_split(data,test_size=0.1)
columns = list(data.columns)
x_train = train[columns[:-1]]
y_train = train[columns[-1]]
x_test = test[columns[:-1]]
y_test = test[columns[-1]]

'''

def squareLoss(output, y):
    if type(y) != np.ndarray:
        y = np.array(y)
    else:
        pass
    E = []
    for i in range(len(y)):
        E.append(np.square((y[i] - output[i])))
        
    E_sum=sum(E)
    
    return 0.5*E_sum/len(output)

#Function to find the cross entropy, using log base 2
def cross_entropy(output, y):
    loss = y*np.log(output) + (1-y)*np.log(1-output)
    return -loss


#THIS MUST BE WRONG - THROWING AN ERROR OF DIVIDE BY ZERO
#Changed since using the softmax function as output
def Dcross_entropy(output,y):
    #diff = (output - y)/(output*(1 - output))
    diff = output - y
    return diff


def tanh(z):
    return np.tanh(z)

def Dtanh(z):
    return 1 - pow(tanh(z),2)


def ReLu(z):
    return np.maximum(0, z)

def DReLu(z):
    return 1 * (z>0)


#Neural network class
class NN():
    
    def __init__(self, x_data, y, learning_rates: list=[0.4,0.4,0.4], hidden_nodes1: int=9, hidden_nodes2: int=9, output_nodes:int=2):
        
        self.input_data = x_data
        self.y_targets = y
        
        #TRY AND MAKE MINI BATCHES
        
        
        self.example = np.array(x_data.iloc[0])
        #Want to use row-vectors as examples and column-vectors as weights
        self.example =  self.example.reshape(self.example.shape[0], 1)
        self.target = y.iloc[0]
        
        self.hidden_nodes1 = hidden_nodes1
        self.hidden_nodes2 = hidden_nodes2
        self.learning_rate1 = learning_rates[0]
        self.learning_rate2 = learning_rates[1]
        self.learning_rate3 = learning_rates[2]
        
        
        self.weights1 = np.random.rand(len(self.example), self.hidden_nodes1)
        self.weights2 = np.random.rand(self.hidden_nodes1, self.hidden_nodes2)
        self.weights3 = np.random.rand(self.hidden_nodes2,output_nodes)
        
        
        self.hidden_layer1 = np.zeros(shape=(1,self.hidden_nodes1))
        self.hidden_layer2 = np.zeros(shape=(1,self.hidden_nodes2))
        
        self.output = np.zeros(shape=(1,output_nodes))
        self.predictions = np.zeros(len(self.y_targets))
        
        
    def feedforward(self):
        
        self.hidden_layer1 = tanh(np.dot(self.example.T, self.weights1))
        self.hidden_layer1 = self.hidden_layer1.T
        
        self.hidden_layer2 = sigmoid(np.dot(self.hidden_layer1.T, self.weights2))
        self.hidden_layer2 = self.hidden_layer2.T
        
        self.output = sigmoid(np.dot(self.hidden_layer2.T, self.weights3))  #changed from softmax
        self.output = self.output.T
        
        return self.output
        
    
    def backpropagate(self):
        
        delta_loss = Dcross_entropy(self.output, self.target)
        #changed from dstable_softmax
        delta_output = Dtanh(np.dot(self.weights3.T, self.hidden_layer2))
        deltaSig1 = Dsigmoid(np.dot(self.weights1.T ,self.example))
        deltaSig2 = Dsigmoid(np.dot(self.weights2.T, self.hidden_layer1))
        terms1_2 = delta_loss * delta_output
        
        self.d_weight1 = np.dot(self.example, (np.dot(self.weights2, (np.dot(self.weights3, terms1_2) * deltaSig2)) * deltaSig1).T)
        self.d_weight2 = np.dot(self.hidden_layer1, (np.dot(self.weights3, terms1_2) * deltaSig2).T)
        self.d_weight3 = np.dot(self.hidden_layer2, ((delta_loss * delta_output).T))
        
        self.weights1 = self.weights1 - self.learning_rate1 * self.d_weight1
        self.weights2 = self.weights2 - self.learning_rate2 * self.d_weight2
        self.weights3 = self.weights3 - self.learning_rate3 * self.d_weight3


    def train(self, epochs: int=1):

        self.predictions = []
        self.predictions.append(self.output)
        for i in range(1,len(self.y_targets),1):
            self.feedforward()
            self.predictions.append(self.output)
            self.backpropagate()
            self.example = np.array(self.input_data.iloc[i])
            self.example =  self.example.reshape(self.example.shape[0], 1)
            self.target = self.y_targets.iloc[i]
            
    def shuffle(self):
        
        self.input_data, self.y_targets = sklearn.utils.shuffle(self.input_data, self.y_targets)
        self.example = np.array(self.input_data.iloc[0])
        #Want to use row-vectors as examples and column-vectors as weights
        self.example =  self.example.reshape(self.example.shape[0], 1)
        self.target = self.y_targets.iloc[0]
            
            
    #Function to predict outputs
    def predict(self, test_data):
        
        self.guesses = []
        for i in range(len(test_data)):
            
            self.example = np.array(test_data.iloc[i])
            self.example =  self.example.reshape(self.example.shape[0], 1)
            guess = self.feedforward()
            self.guesses.append(guess)
            
                
        return self.guesses  
        
            
cols = list(data.columns)
#cols = cols[:-1]

scaled2data = cp.deepcopy(data)
scaled2 = sklearn.preprocessing.scale(data[cols])



scaled_data = cp.deepcopy(data)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(scaled_data[cols])       
vals = scaler.transform(scaled_data[cols])
scaled_data[cols] = vals


'''

x_trainS = trainS[columns[:-1]]
y_trainS = trainS[columns[-1]]
x_testS = testS[columns[:-1]]
y_testS = testS[columns[-1]]
'''
columns = list(scaled_data.columns)
trainS, testS = train_test_split(scaled_data,test_size=0.2)
cv_test, test = train_test_split(testS, test_size=0.5)
x_trainS = trainS[columns[:-1]]
y_trainS = trainS[columns[-1]]
x_cvtestS = cv_test[columns[:-1]]
y_cvtestS = cv_test[columns[-1]]
x_testS = test[columns[:-1]]
y_testS = test[columns[-1]]


'''
plt.xlabel("Epochs")
plt.ylabel("Square-loss")
plt.show()
'''
def find_errors(output, y):
    df = pd.DataFrame(y)
    df['pred'] = output
    correct = df.loc[np.sign(df['pred']) == np.sign(df['Classification'])].count()
    error = 1 - correct.iloc[0]/len(df['pred'])
    return error


rates = np.arange(0.1,0.5,0.1)

#shuffle and 100 epochs
#explain works
#architecture
#loss vs epochs
#loss vs validation loss
epochs =50
scaled_model = NN(x_trainS, y_trainS, learning_rates=[0.015,0.015,0.015], output_nodes=1)
losses = []
cv_errors = []
train_losses = []
train_loss = []
for i in range(epochs):
#NEED TO SHUFFLE THE DATA
    scaled_model.train()
    pred = scaled_model.predictions
    
    for i in range(len(pred)):
        if pred[i] > 0.90 :
            pred[i] = 0.901388
        elif pred[i] < - 0.99:
            pred[i] = -1.1094
        else:
            pass

    train_loss.append(squareLoss(pred, y_trainS))
print(train_loss)


'''
cv_pred = scaled_model.predict(x_cvtestS)
train_losses.append(train_loss)
cv_loss = squareLoss(cv_pred, y_cvtestS)
losses.append(cv_loss)

#cv_error = find_errors(cv_pred, y_cvtestS)
#cv_errors.append(cv_error)
#scaled_model.shuffle()

x_cvtestS, y_cvtestS = sklearn.utils.shuffle(x_cvtestS, y_cvtestS)
'''

    
'''
plt.plot([i for i in range(epochs)],losses)
plt.xlabel("Epochs")
plt.ylabel("Square-Loss")
plt.show()

plt.plot([i for i in range(epochs)],cv_errors)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
'''
plt.scatter([i for i in range(epochs)],train_loss)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

    



