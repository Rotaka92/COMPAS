# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:07:57 2018

@author: TapperR
"""

#matrix math
import numpy as np
#data manipulation
import pandas as pd
#matrix data structure
from patsy import dmatrices
#for error logging
import warnings


import time
start_time = time.time()


import os 
#os.chdir('C:\\Users\\Robin\\Desktop\\compas-analysis')
os.chdir('C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis')




#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#import pickle
#creating our own knn-classifier
#import matplotlib.pyplot as plt
#from matplotlib import style

from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random
from sklearn import preprocessing, cross_validation, neighbors




raw4 = pd.read_table('cox-parsed.csv', sep=',', encoding='utf-8',  na_filter = True)

#factors
raw4['sex'] = raw4['sex'].astype('category')
classnamesSex, indicesSex = np.unique(raw4['sex'], return_inverse=True)
raw4['sexInd'] = indicesSex


raw4['age_cat'] = raw4['age_cat'].astype('category')
classnamesAge, indicesAge = np.unique(raw4['age_cat'], return_inverse=True)
raw4['age_catInd'] = indicesAge


raw4['race'] = raw4['race'].astype('category')
classnamesRace, indicesRace = np.unique(raw4['race'], return_inverse=True)
raw4['raceInd'] = indicesAge



raw5Fac = raw4[['sexInd', 'age_catInd', 'raceInd', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']]
raw5Lab = raw4[['is_recid']]

        
raw5Lab_raw = list(raw5Lab['is_recid'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(np.array(raw5Fac), np.array(raw5Lab_raw), test_size=0.2)


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

## The coefficients
#print('Coefficients: \n', regr.coef_)





#outputs probability between 0 and 1, used to help define our logistic regression curve
def sigmoid(x):
    '''Sigmoid function of x.'''
    return 1/(1+np.exp(-x))


#makes the random numbers predictable
#(pseudo-)random numbers work by starting with a number (the seed), 
#multiplying it by a large number, then taking modulo of that product. 
#The resulting number is then used as the seed to generate the next "random" number. 
#When you set the seed (every time), it does the same thing every time, giving you the same numbers.
#good for reproducing results for debugging


#np.random.seed(0) # set the seed

##Step 1 - Define model parameters (hyperparameters)

## algorithm settings
#the minimum threshold for the difference between the predicted output and the actual output
#this tells our model when to stop learning, when our prediction capability is good enough
tol=1e-8 # convergence tolerance

lam = None # l2-regularization
#how long to train for?
max_iter = 30 # maximum allowed iterations


## data creation settings
#Covariance measures how two variables move together. 
#It measures whether the two move in the same direction (a positive covariance) 
#or in opposite directions (a negative covariance). 
#r = 0.95 # covariance between x and z
#n = 1000 # number of observations (size of dataset to generate) 
sigma = 1 # variance of noise - how spread out is the data?
##
#




### model settings
#beta_x, beta_z, beta_v = -4, .9, 1 # true beta coefficients
#var_x, var_z, var_v = 1, 1, 4 # variances of inputs
##
#### the model specification you want to fit
#formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'
formula = 'y ~ sexInd + age_catInd + raceInd + juv_fel_count + juv_misd_count + juv_other_count + priors_count'
#
#                                 
raw5Fac.columns                              
                                 
## Step 2 - Generate and organize our data

#The multivariate normal, multinormal or Gaussian distribution is a generalization of the one-dimensional normal 
#distribution to higher dimensions. Such a distribution is specified by its mean and covariance matrix.
#so we generate values input values - (x, v, z) using normal distributions

##A probability distribution is a function that provides us the probabilities of all 
##possible outcomes of a stochastic process. 
#
##lets keep x and z closely related (height and weight)
#x, z = np.random.multivariate_normal([0,0], [[var_x,r],[r,var_z]], n).T
##blood presure
#v = np.random.normal(0,var_v,n)**3
#
##create a pandas dataframe (easily parseable object for manipulation)
#A = pd.DataFrame({'x' : x, 'z' : z, 'v' : v})
##compute the log odds for our 3 independent variables
##using the sigmoid function 
#A['log_odds'] = sigmoid(A[['x','z','v']].dot([beta_x,beta_z,beta_v]) + sigma*np.random.normal(0,1,n))
raw5Fac['log_odds'] = sigmoid(raw5Fac.dot(np.transpose([regr.coef_]))[0] + sigma*np.random.normal(0,1,len(raw5Fac)))


##compute the probability sample from binomial distribution
##A binomial random variable is the number of successes x has in n repeated trials of a binomial experiment. 
##The probability distribution of a binomial random variable is called a binomial distribution. 
#A['y'] = [np.random.binomial(1,p) for p in A.log_odds]
raw5Fac['y'] = [np.random.binomial(1,p) for p in raw5Fac.log_odds]


##create a dataframe that encompasses our input data, model formula, and outputs 
#y, X = dmatrices(formula, A, return_type='dataframe')
y, X = dmatrices(formula, raw5Fac, return_type='dataframe')

#print it
X.head(100)                               
  



                               
#like dividing by zero, it is not that necessary
def catch_singularity(f):
    '''Silences LinAlg Errors and throws a warning instead.'''
    
    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Algorithm terminated - singular Hessian!')
            return args[0]
    return silencer


@catch_singularity
def newton_step(curr, X, lam=None):
    '''One naive step of Newton's Method'''
    
    #how to compute inverse? http://www.mathwarehouse.com/algebra/matrix/images/square-matrix/inverse-matrix.gif
    
    ## compute necessary objects
    #create probability matrix, miniminum 2 dimensions, tranpose (flip it)
    p = np.array(sigmoid(X.dot(curr[:,0])), ndmin=2).T
    #create weight matrix from it
    W = np.diag((p*(1-p))[:,0])
    #derive the hessian 
    hessian = X.T.dot(W).dot(X)
    #derive the gradient
    grad = X.T.dot(y-p)
    
    ## regularization step (avoiding overfitting)
    if lam:
        # Return the least-squares solution to a linear matrix equation
        step, *_ = np.linalg.lstsq(hessian + lam*np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)
        
    ## update our weights
    beta = curr + step
    
    return beta


@catch_singularity
def alt_newton_step(curr, X, lam=None):
    '''One naive step of Newton's Method'''
    
    ## compute necessary objects
    p = np.array(sigmoid(X.dot(curr[:,0])), ndmin=2).T
    W = np.diag((p*(1-p))[:,0])
    hessian = X.T.dot(W).dot(X)
    grad = X.T.dot(y-p)
    
    ## regularization
    if lam:
        #Compute the inverse of a matrix.
        step = np.dot(np.linalg.inv(hessian + lam*np.eye(curr.shape[0])), grad)
    else:
        step = np.dot(np.linalg.inv(hessian), grad)
        
    ## update our weights
    beta = curr + step
    
    return beta



def check_coefs_convergence(beta_old, beta_new, tol, iters):
    '''Checks whether the coefficients have converged in the l-infinity norm.
    Returns True if they have converged, False otherwise.'''
    #calculate the change in the coefficients
    coef_change = np.abs(beta_old - beta_new)
    
    #if change hasn't reached the threshold and we have more iterations to go, keep training
    return not (np.any(coef_change>tol) & (iters < max_iter))

                                
                                 
                                 ## initial conditions
#initial coefficients (weight values), 2 copies, we'll update one
beta_old, beta = np.ones((len(X.columns),1)), np.zeros((len(X.columns),1))

#num iterations we've done so far
iter_count = 0
#have we reached convergence?
coefs_converged = False

#if we haven't reached convergence... (training step)
while not coefs_converged:
    
    #set the old coefficients to our current
    beta_old = beta
    #perform a single step of newton's optimization on our data, set our updated beta values
    beta = newton_step(beta, X, lam=lam)
    #increment the number of iterations
    iter_count += 1
    
    #check for convergence between our old and new beta values
    coefs_converged = check_coefs_convergence(beta_old, beta, tol, iter_count)
    
print('Iterations : {}'.format(iter_count))
print('Beta : {}'.format(beta)) 
                                 
                     

#how often do we get the right prediction
raw5Lab['pred'] = y

check = 0      
for i in range(len(raw5Lab)):
    if raw5Lab.is_recid[i] == raw5Lab.pred[i]:
        check += 1
    else:
        pass



quote = check/len(raw5Lab)
print(quote)




'''
#how the typical classification numbers look like
TP = 0
FP = 0
TN = 0
FN = 0


for i in range(len(realVSpred)):

    if realVSpred['pred'][i] == 1 and realVSpred['real'][i] == 1:
        #print('True Positive')
        TP += 1
                
    if realVSpred['pred'][i] == 1 and realVSpred['real'][i] == 0:
        #print('False Positive')
        FP += 1
        
    if realVSpred['pred'][i] == 0 and realVSpred['real'][i] == 0:
        #print('True Negative') 
        TN += 1
        
    if realVSpred['pred'][i] == 0 and realVSpred['real'][i] == 1:
        #print('False Negative')
        FN += 1
        

print('True Positive:', TP, 'False Positive:', FP, 'True Negative:', TN, 'False Negative:', FN)
print('Trues:', TP+TN, 'False:', FP+FN, 'Ratio: True/All = ', (TP+TN)/len(realVSpred))


print("--- %s seconds ---" % (time.time() - start_time))
'''


            
                                 
