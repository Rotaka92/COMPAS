# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:29:20 2018

@author: TapperR
"""

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
from sklearn import preprocessing, cross_validation, neighbors, svm

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





###   difference to kNN   ###
#Define the classifier:
clf = svm.SVC()




#Train the classifier:
k = clf.fit(X_train, y_train)



#Test:
accuracy = clf.score(X_test, y_test)


###   change to kNN   ###
confidence = clf.score(X_test, y_test)
print(confidence)



#predicting 


example_measures = np.array([[1, 0, 1, 1, 0, 0, 0],[1, 0, 2, 0, 2, 0, 0]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)




#show the results

y_pred = k.predict(X_test)

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
####  doing the SVM from scratch  ####
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        
        self.colors = {1:'r',-1:'b'}
        
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
            #ax = fig.add_subplot(1,1,1)
    # train
    def fit(self, data):
        self.data = data
#        data = data_dict
#         { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1,1],
                      [-1,1,1],
                      [-1,-1,1],
                      [1,-1,1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
                    
       
#        for yi in data:
#            for featureset in data[yi]:
#                for feature in featureset:
#                    all_data.append(feature)  
              
                    

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        

#        max_feature_value = max(all_data)
#        min_feature_value = min(all_data)
#        all_data = None


        # support vectors yi(xi.w+b) = 1
        

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]

          
#        step_sizes = [max_feature_value * 0.1,
#                      max_feature_value * 0.01,
#                      # point of expense:
#                      max_feature_value * 0.001,
#                      ]

        
        
        # extremely expensive
        b_range_multiple = 2
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        #latest_optimum = max_feature_value*10
        
        for step in step_sizes:
            #step = step_sizes[0]
            w = np.array([latest_optimum,latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    #b = np.arange(-1*(max_feature_value*b_range_multiple),max_feature_value*b_range_multiple,step*b_multiple)[0]
                    for transformation in transforms:
                        #transformation = transforms[3]
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        for i in self.data:
                           #i = 1
                            for xi in self.data[i]:
                                #xi = data[i][1]
                                #print(xi)
                                yi=i
                                
                                #constraint for entering the opt_dict
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                 
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            #w = opt_choice[0]
            self.b = opt_choice[1]
            #b = opt_choice[1]
            
            #go two steps back because the latest step may have missed the optimum
            latest_optimum = opt_choice[0][0]+step*2
            
        for i in self.data:
            #i = -1
            for xi in self.data[i]:
                #xi = data[i][0]
                yi=i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b)) 
                #print(xi,':',yi*(np.dot(w,xi)+b)) 

    def predict(self,features):
        # sign( x.w+b ). features = predict_us[0]
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        #classification = np.sign(np.dot(np.array(features),w)+b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
            #ax.scatter(features[0], features[1], s=200, marker='*', c=colors[classification])        
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        
        #[[ax.scatter(x[0],x[1],s=100,color=colors[i]) for x in data_dict[i]] for i in data_dict]
        

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        #datarange = (min_feature_value*0.9,max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        #psv1 = hyperplane(hyp_x_min, svm.w, svm.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        #psv2 = hyperplane(hyp_x_max, svm.w, svm.b, 1)
        #fig = plt.figure()
        #ax = fig.add_subplot(1,1,1)
        
        #ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')
        #ax.scatter(hyp_x_min,psv1)
        
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()
        
data_dict = {-1:np.array([[1,7,3],
                          [2,8,4],
                          [3,8,5],]),
             
             1:np.array([[5,1,4],
                         [6,-1,5],
                         [7,3,6],])}

svm = Support_Vector_Machine()

#svm.w

svm.fit(data=data_dict)




predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)



#svm.predict([1,3,1])



svm.visualize()
'''






