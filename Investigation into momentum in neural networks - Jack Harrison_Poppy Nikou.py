# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:29:16 2019

This file contains the code used in the COMP61011 final project. By requirement all code had to run within one file, in order to reproduce results in the report.

The project investigated the effect of classical momentum and Nesterov advanced
gradient on the convergence time of a multi-layered neural network.

The code contains a neural network class, which is trained over a chosen number
of epochs on inputted data.


The data used for this code is stored locally, and has been saved using the
filenames:
    "data_banknote_authentication.csv"
    "Breast-cancer-wisconsin.csv"

    
The code outputs all plots for one dataset, and as a result the 
runtime is very long (~45mins). To reduce the time, remove momentum and 
learning rate values to iterate over. 

@authors: Jack Harrison and Poppy Nikou
"""


import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy as cp


#Function to calculate the sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))


#Derivative of the sigmoid function
def Dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
    

#Function to find the Mean Square Loss 
def squareLoss(output, y):
    if type(y) != np.ndarray:
        y = np.array(y)
        
    E = np.square(y - output).sum()
    return E / len(y)


#Function to find the cross entropy
def cross_entropy(output, y):
    loss = y*np.log(output) + (1-y)*np.log(1-output)
    return -loss


#Function to return the difference between output and y
#This is used in place of doing Dcross_entropy * Dsigmoid since cross terms
#cancel just leaving the difference
#D indicates the derivative
def Dcross_and_Dsig(output,y):
    return output - y


#Function to find tanh
def tanh(z):
    return np.tanh(z)


#Derivative of the tanh function
def Dtanh(z):
    return 1 - pow(tanh(z),2)


#Neural network class
class NN():
    
    #Initialized with the training data. Default values are specified
    def __init__(self, x_data, y, learning_rates: list=[0.01,0.01,0.01], hidden_nodes1: int=5,
                 hidden_nodes2: int=5, momentum: float=0, output_nodes:int=1):
        
        #Store the training data
        self.input_data = x_data
        self.y_targets = y
        
        #Set the specific example
        self.example = np.array(x_data.iloc[0])
        self.example =  self.example.reshape(self.example.shape[0], 1)    #Reshape so that row-vectors as examples and column-vectors as weights
        self.target = y.iloc[0]
        

        #Set the hyperparamaters
        self.beta = momentum
        self.hidden_nodes1 = hidden_nodes1
        self.hidden_nodes2 = hidden_nodes2
        self.learning_rate1 = learning_rates[0]
        self.learning_rate2 = learning_rates[1]
        self.learning_rate3 = learning_rates[2]
        
        #Initialise the weight matrices with random numbers in [0,1]
        self.weights1 = np.random.rand(len(self.example), self.hidden_nodes1)
        self.weights2 = np.random.rand(self.hidden_nodes1, self.hidden_nodes2)
        self.weights3 = np.random.rand(self.hidden_nodes2,output_nodes)
        
        #Initialise velocity term with zeros (so no effect with no beta)
        self.z1 = np.zeros(shape=(len(self.example),self.hidden_nodes1))
        self.z2 = np.zeros(shape=(self.hidden_nodes1, self.hidden_nodes2))
        self.z3 = np.zeros(shape=(self.hidden_nodes2,output_nodes))
        
        #Initialise hidden-layer values with 0
        self.hidden_layer1 = np.zeros(shape=(1,self.hidden_nodes1))
        self.hidden_layer2 = np.zeros(shape=(1,self.hidden_nodes2))
        
        self.output = np.zeros(shape=(1,output_nodes))
        
        
    def feedforward(self):
        #Use the chosen activation functions to calculate each layer
        self.hidden_layer1 = tanh(np.dot(self.example.T, self.weights1))
        self.hidden_layer1 = self.hidden_layer1.T
        
        self.hidden_layer2 = tanh(np.dot(self.hidden_layer1.T, self.weights2))
        self.hidden_layer2 = self.hidden_layer2.T
        
        self.output = sigmoid(np.dot(self.hidden_layer2.T, self.weights3))  #changed from softmax
        self.output = self.output.T
        
        return self.output
        
    #Function to backpropagate, bool choice for whether to use Nesterov momentum or not
    def backpropagate(self, Nesterov: bool=False):
        
        if(Nesterov):
            #Calculate the 'look-ahead' weights
            self.ahead_weight1 = self.weights1 - self.beta * self.z1
            self.ahead_weight2 = self.weights2 - self.beta * self.z2
            self.ahead_weight3 = self.weights3 - self.beta * self.z3
            
            #Find the new derivatives with the ahead weights to be used below
            delta_loss = Dcross_and_Dsig(self.output, self.target)
            deltaSig1 = Dtanh(np.dot(self.ahead_weight1.T ,self.example))
            deltaSig2 = Dtanh(np.dot(self.ahead_weight2.T, self.hidden_layer1))
            terms1_2 = delta_loss
            
            #Find the change in weights using ahead weights
            self.d_weight1 = np.dot(self.example, (np.dot(self.ahead_weight2, (np.dot(self.ahead_weight3, terms1_2) * deltaSig2)) * deltaSig1).T)
            self.d_weight2 = np.dot(self.hidden_layer1, (np.dot(self.ahead_weight3, terms1_2) * deltaSig2).T)
            self.d_weight3 = np.dot(self.hidden_layer2, ((terms1_2).T))
        
            #Update the weights using the old z
            self.weights1 = self.weights1 - self.z1 
            self.weights2 = self.weights2 - self.z2
            self.weights3 = self.weights3 - self.z3
    
            #Update the z values
            self.z1 = self.beta * self.z1 + self.learning_rate1 * self.d_weight1
            self.z2 = self.beta * self.z2 + self.learning_rate2 * self.d_weight2
            self.z3 = self.beta * self.z3 + self.learning_rate3 * self.d_weight3
    
        else:
            
            #Find the derivatives used in the chain rule
            delta_loss = Dcross_and_Dsig(self.output, self.target)
            deltaSig1 = Dtanh(np.dot(self.weights1.T ,self.example))
            deltaSig2 = Dtanh(np.dot(self.weights2.T, self.hidden_layer1))
            terms1_2 = delta_loss 
        
            #Find the amounts each weight changes
            self.d_weight1 = np.dot(self.example, (np.dot(self.weights2, (np.dot(self.weights3, terms1_2) * deltaSig2)) * deltaSig1).T)
            self.d_weight2 = np.dot(self.hidden_layer1, (np.dot(self.weights3, terms1_2) * deltaSig2).T)
            self.d_weight3 = np.dot(self.hidden_layer2, ((terms1_2).T))
            
            #Update the weights with the old z
            self.weights1 = self.weights1 -  self.z1 
            self.weights2 = self.weights2 -  self.z2
            self.weights3 = self.weights3 -  self.z3 
            
            #Update the z (note beta=0 implies normal SGD)
            self.z1 = self.learning_rate1 *self.d_weight1 + self.beta * self.z1
            self.z2 = self.learning_rate2 *self.d_weight2 + self.beta * self.z2
            self.z3 = self.learning_rate3 *self.d_weight3 + self.beta * self.z3
        
        return self

    #Function to train over the intialised training set, with a choice for Nesterov
    def train(self, Nesterov):
        
        self.training_outputs = []  #Variable for examining training outputs
        #Iterate over all training examples
        for i in range(0, len(self.y_targets),1):
            
            #Set the example and target
            self.example = np.array(self.input_data.iloc[i])
            self.example =  self.example.reshape(self.example.shape[0], 1)
            self.target = self.y_targets.iloc[i]
            
            #Feedforward and backpropagate
            self.example_output = self.feedforward()
            self.training_outputs.append(self.example_output)
            self.backpropagate(Nesterov)
            
        return self
        
    #Function using sklearn's shuffle
    def shuffle(self):
        
        #Shuffle the training data
        self.input_data, self.y_targets = sklearn.utils.shuffle(self.input_data, self.y_targets)
        #Set the example to the first training example
        self.example = np.array(self.input_data.iloc[0])
        self.example =  self.example.reshape(self.example.shape[0], 1)  #Want to use row-vectors as examples and column-vectors as weights
        self.target = self.y_targets.iloc[0]
            
    #Function to predict outputs
    def predict(self, test_data):
        
        self.predictions = []
        #Iterate over all the testing data
        for i in range(len(test_data)):
            #Initialise the example for passing through the network
            self.example = np.array(test_data.iloc[i])
            self.example =  self.example.reshape(self.example.shape[0], 1)
            result = self.feedforward()
            self.predictions.append(result) #Store the output
            
        return self.predictions 
        
    

filename = "Breast-cancer-wisconsin.csv"
filename2 = "data_banknote_authentication.csv"

#Read in data to DataFrame
data = pd.read_csv(filename2)
#Store the column titles
columns = list(data.columns)
cols = columns[:-1]

#Use MinMaxScaler on the data
scaled_data = cp.deepcopy(data)
scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(scaled_data[cols])
vals = scaler.transform(scaled_data[cols])
scaled_data[cols] = vals

#Split the data randomly 70:30
train, test = train_test_split(scaled_data,test_size=0.3)
x_train = train[columns[:-1]]     #
y_train = train[columns[-1]]

#Further split the testing data 50:50
cv_test, test = train_test_split(test, test_size=0.5)
xcv = cv_test[columns[:-1]]
ycv = cv_test[columns[-1]]
x_test = test[columns[:-1]]
y_test = test[columns[-1]]


#Function to find the mean error
def mean_error(y_output, y_actual):
    
    y_output = [1 if val > 0.5 else 0 for val in y_output]
    correct_matches = [a for a,b in zip(y_output, y_actual) if a == b]
    correct = len(correct_matches)
    errors = len(y_actual) - correct
    return errors/len(y_actual)


# -----------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Code used when trying to find a reasonable learning rate to use:    

lrs = np.arange(0, 1, 0.01)
losses = []
epochs = 100
Nesterov = False
#Loop over all the learning rates
for lr in lrs:
    #Make a new network with the chosen learning rates
    scaled_model = NN(x_train, y_train, learning_rates=[lr,lr,lr], output_nodes=1)
    #Train for the chosen number of epochs
    for e in range(epochs):
        scaled_model.train(Nesterov)
        scaled_model.shuffle()      #Shuffle the data each time
        
    #Find a final testing error
    test_pred = scaled_model.predict(x_test)
    test_loss = mean_error(test_pred, y_test)
    losses.append(test_loss)
    
    
plt.scatter(lrs, losses, marker = 'x')
plt.xlabel("Learning rate")
plt.ylim(0,1)
plt.xlim(0,1)
plt.ylabel("Testing error")
plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Code used to find the effect of momentum:


#Set the number of epochs to train over and tests to average
epochs = 100
number_of_tests = 3


#Containers to store results for each momentum
traindf_list = []
cvdflist = []
losses_dflist = []
final_test_errors = []

#Choose momenta to use and whether or not to use Nesterov
Nesterov = True
momenta = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99]

for m in momenta:
    #Make new DataFrames for each momentum
    loss_curves = pd.DataFrame()
    train_loss_curves = pd.DataFrame()
    cverrorsdf = pd.DataFrame()
    testerrors = []
    #Run the chosen number of tests
    for j in range(number_of_tests):
        #Make a new model each time
        scaled_model = NN(x_train, y_train, learning_rates=[0.005,0.005,0.005], momentum=m, output_nodes=1)
        
        losses = []
        train_losses = []
        test_losses = []
        cv_errors_list = []
        
        #Train over the chosen number of epochs
        for i in range(epochs):
            
            scaled_model.train(Nesterov)
            pred = scaled_model.predict(x_train)
            pred = [i[0] for i in pred]     #Remove one layer (predictions are returned as arrays within arrays)
            
            #Find the training error and append it
            train_loss = mean_error(pred, y_train)
            train_losses.append(train_loss)
            #Repeat for the cross-validation error
            cv_pred = scaled_model.predict(xcv)
            cv_pred = [i[0] for i in cv_pred]
            cv_errors = mean_error(cv_pred, ycv)
            cv_errors_list.append(cv_errors)
            
            scaled_model.shuffle()  #Shuffle the data after each epoch
            
        #Find the testing error after the model is trained
        test_pred = scaled_model.predict(x_test)
        test_loss = squareLoss(test_pred, y_test)
        test_err = mean_error(test_pred, y_test)
        testerrors.append(test_err)
        losses.append(test_loss)
    
        #Add the lists to a test in the DataFrame containers
        loss_curves[f"test {j}"] = losses
        train_loss_curves[f"test {j}"] = train_losses
        cverrorsdf[f"test {j}"] = cv_errors_list

    #Add the final DataFrame (containing all the tests) to a list
    mean_test_error = np.array(testerrors).mean()
    traindf_list.append(train_loss_curves)
    cvdflist.append(cverrorsdf)
    final_test_errors.append(mean_test_error)
    losses_dflist.append(loss_curves)
    

#Find the means and deviations for each example in each DataFrame
for val in traindf_list:
    val["Mean"] = val.mean(axis=1)
    val["std"] = val.std(axis=1)
    
#Repeat for CV data
for val in cvdflist:
    val["Mean"] = val.mean(axis=1)
    val["std"] = val.std(axis=1)
    
    
#Plot the means onto graphs, use z to label the momenta correctly
z = 0
for df in traindf_list:
    #if z in [0,5,9,11]:  #Used for only printing the momentum you want
    plt.errorbar([i+1 for i in range(epochs)], df["Mean"], yerr=df["std"], label=f"{momenta[z]}", errorevery=3)
    plt.xlabel("Epochs")
    plt.ylabel("Training error")
    z+=1
    
plt.legend()
plt.show()


#Repeat for the testing error (CV)
z = 0
for df in cvdflist:

    plt.errorbar([i+1 for i in range(epochs)], df["Mean"],yerr=df["std"], label=f"{momenta[z]}", errorevery=3)
    plt.xlabel("Epochs")
    plt.ylabel("Testing error")
    z+=1
    
plt.title("Set A", fontsize=14)
plt.legend(fontsize=12)
plt.show()
    
