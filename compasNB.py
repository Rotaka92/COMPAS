from sklearn import datasets




iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()


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

data = pd.read_csv('spam.csv', encoding = 'latin1')


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




#posterior = prior occurences * likelihood/ evidence = likelihood of something to be positive
#are our features independent and from a similar distribution? check it!



y_pred = gnb.fit(X_train, y_train).predict(X_test)


#how often the predictions are right?
realVSpred = pd.DataFrame({'real': y_test, 'pred': y_pred})



check = 0      
for i in range(len(realVSpred)):
    if realVSpred.real[i] == realVSpred.pred[i]:
        check += 1
    else:
        pass



quote = check/len(realVSpred)
print(quote)


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
data = pd.read_csv('spam.csv', encoding='latin-1')

X = data['v2']
Y = data['v1']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)


#A is probability that email is spam, B is the content of the mail(Email-body)
def train():
    total = 0
    numSpam = 0
    for email in trainData:
        if email.label == SPAM :
            numSpam +=1
        total += 1
        processEmail(email.body , email.label)
    pA = numSpam/float(total)
    pNotA = (total - numSpam)/float(total)

#reading words from a specific email
    def processEmail(body , label):
        for word in body:
            if label == SPAM:
                trainPositive[word] = trainPositive.get(word, 0) + 1
                positiveTotal += 1
            else:
                trainNegative[word] = trainNegative.get(word, 0) + 1
                negativeTotal += 1
#gives the conditional probability p(B_i/A_x)
def conditionalEmail(body, spam) :
    result = 1.0
    for word in body:
        result *= conditionalWord(body , spam)
    return result

#classifies a new email as spam or not spam
    def classify(email):
          isSpam = pA * conditionalEmail(email, True) # P (A | B)
          notSpam = pNotA * conditionalEmail(email, False) # P(¬A | B)
          
          return isSpam > notSpam
      
#Laplace Smoothing for the words not present in the training set
#gives the conditional probability p(B_i | A_x) with smoothing
def conditionalWord(word, spam):
    if spam:
        return (trainPositive.get(word,0)+alpha)/(float)(positiveTotal+alpha*numWords)
    
    
    return (trainNegative.get(word,0)+alpha)/(float)(negativeTotal+alpha*numWords)
'''

