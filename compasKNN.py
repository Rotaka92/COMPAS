# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:52:53 2018

@author: TapperR
"""

import time
start_time = time.time()


import os 
#os.chdir('C:\\Users\\Robin\\Desktop\\compas-analysis')
os.chdir('C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis')




#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#import pickle
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing, cross_validation, neighbors

#creating our own knn-classifier
#import matplotlib.pyplot as plt
#from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
#style.use('fivethirtyeight')
'''

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()





def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result





random.shuffle(full_data)
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]


#populate the dictionaries
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])
    
    
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total)



print("--- %s seconds ---" % (time.time() - start_time))












###   Plotting  ###


dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# same as:
##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1],s=100,color=i)
        
plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color = result)  
plt.show()
'''



######  do the KNN-Classifier for our Compas-Problem  #####

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
#raw5Lab['is_recid'] = [[1,0] if x==1 else [0,1] for x in raw5Lab_raw]
#raw5Lab['is_recid'] = [[0,1] if x==1 else [1,0] for x in raw5Lab_raw]


#list(raw5Lab['is_recid'])




'''
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()


#define our features (X) and labels (y):
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
'''


###create training and testing samples, dataset: breast_cancer
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


###create training and testing samples, dataset: compas
X_train, X_test, y_train, y_test = cross_validation.train_test_split(np.array(raw5Fac), np.array(raw5Lab_raw), test_size=0.2)




#Define the classifier:
clf = neighbors.KNeighborsClassifier()




#Train the classifier:
k = clf.fit(X_train, y_train)



#Test:
accuracy = clf.score(X_test, y_test)
print(accuracy)



#predicting 

example_measures = np.array([[1, 0, 1, 1, 0, 0, 0],[1, 0, 2, 0, 2, 0, 0]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)



y_pred = k.predict(X_test)


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






