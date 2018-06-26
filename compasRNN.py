# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:04:01 2018

@author: TapperR
"""
import time
start_time = time.time()

#### Can we beat it with a Neural Network???? ########
#python compasNN.py





'''
import os 
os.chdir('C:\\Users\\Robin\\Desktop\\compas-analysis')




####TUT2######
from create_sentiment_featuresets import create_feature_sets_and_labels
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np
import pandas as pd



train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Nothing changes
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
	    
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
				epoch_loss += c
				i+=batch_size
				
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

	    
train_neural_network(x)


    
'''
### how does the neural network perform on our data ###
    


import os 
#os.chdir('C:\\Users\\Robin\\Desktop\\compas-analysis')
os.chdir('C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis')




####TUT2######
#from create_sentiment_featuresets import create_feature_sets_and_labels
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#import pickle
import numpy as np
import pandas as pd
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell






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

        
a = list(raw5Lab['is_recid'])
#raw5Lab['is_recid'] = [[1,0] if x==1 else [0,1] for x in a]
raw5Lab['is_recid'] = [[0,1] if x==1 else [1,0] for x in a]

#list(raw5Lab['is_recid'])






test_size = 0.1
testing_size = int(test_size*len(raw5Lab))


train_x = raw5Fac[:-testing_size].values.tolist()   #train_x[0]
train_y = list(raw5Lab['is_recid'])[:-testing_size]                  #train_y[0]
test_x = raw5Fac[-testing_size:].values.tolist()    #test_x[0]
test_y = list(raw5Lab['is_recid'])[-testing_size:]                    #test_y[0]

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 128
hm_epochs = 3
chunk_size = 28
n_chunks = 28
rnn_size = 128




x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

'''

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

#hidden_3_layer = {'f_fum':n_nodes_hl3,
#                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Nothing changes
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

#    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
#    l3 = tf.nn.relu(l3)

    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']

    return output
'''


def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output



#import numpy as np
#
#x = np.ones((1,2,3))
#
#print(x)
#print(np.transpose(x,(1,0,2)))


'''
def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
        
        
	    
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
				epoch_loss += c
				i+=batch_size
				
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

	    
train_neural_network(x)
'''  
    


### how to use the network now for special cases ###


saver = tf.train.Saver()

'''
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
	    
        epoch = 1
        
        while epoch <= hm_epochs:
            if epoch != 1:
                #saver.restore(sess,'C:\\Users\\Robin\\Desktop\\compas-analysis\\model.cpkt')
                saver.restore(sess,'C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis\\model.cpkt')
            epoch_loss = 1
            
            
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
        				                                            y: batch_y})
                epoch_loss += c
                i+=batch_size
				
            #saver.save(sess, 'C:\\Users\\Robin\\Desktop\\compas-analysis\\model.cpkt')
            saver.save(sess, 'C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis\\model.cpkt')
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
            
            if 0.40 < accuracy.eval({x:test_x, y:test_y}) < 0.45:
                break
        
            epoch += 1 
	    
train_neural_network(x)
'''





def train_neural_networkRNN(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
	    
        epoch = 1
        
        while epoch <= hm_epochs: 
            if epoch != 1:
                #saver.restore(sess,'C:\\Users\\Robin\\Desktop\\compas-analysis\\model.cpkt')
                saver.restore(sess,'C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis\\modelRNN.cpkt')
            epoch_loss = 1
            
            
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                
                batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
        				                                            y: batch_y})
                epoch_loss += c
                i+=batch_size
				
            #saver.save(sess, 'C:\\Users\\Robin\\Desktop\\compas-analysis\\model.cpkt')
            saver.save(sess, 'C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis\\modelRNN.cpkt')
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:test_x.reshape(-1, n_chunks, chunk_size), y:test_y}))
            
#            if 0.40 < accuracy.eval({x:test_x, y:test_y}) < 0.45:
#                break
        
            epoch += 1
	    
train_neural_networkRNN(x)

























'''
def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with open('lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,"model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        if result[0] == 0:
            print('Positive:',input_data)
        elif result[0] == 1:
            print('Negative:',input_data)

use_neural_network("He's an idiot and a jerk.")
use_neural_network("This was the best store i've ever seen.")
 '''   







def use_neural_network(input_data, output):
    #input_data = raw5Fac.values.tolist()[:99]
    #input_data = [1, 69, 1, 0, 0, 0, 0, [0, 1]]
    #output = raw5Lab.values[:20]
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    
    prediction = neural_network_model(x)
#    with open('lexicon.pickle','rb') as f:
#        lexicon = pickle.load(f)
      
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,'C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis\\model.cpkt')
#        current_words = word_tokenize(input_data.lower())
        
#        current_words = [lemmatizer.lemmatize(i) for i in current_words]
#        features = np.zeros(len(lexicon))

#        for word in current_words:
            #word = current_words[0]
#            if word.lower() in lexicon:
#                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
#                features[index_value] += 1

#        features = np.array(list(features))
                
        for i in range(len(input_data)):
            features = input_data[i]   
                    
                                        
            # pos: [1,0] , argmax: 0
            # neg: [0,1] , argmax: 1
            result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
            #print(prediction.eval(feed_dict={x:[features]}))
            #print(result)
            if result[0] == 1 and output[i][0][0] == 1:
                #print('True Positive')
                TP += 1
                        
            if result[0] == 1 and output[i][0][0] == 0:
                #print('False Positive')
                FP += 1
                
            if result[0] == 0 and output[i][0][0] == 0:
                #print('True Negative') 
                TN += 1
                
            if result[0] == 0 and output[i][0][0] == 1:
                #print('False Negative')
                FN += 1
                
        
        print('True Positive:', TP, 'False Positive:', FP, 'True Negative:', TN, 'False Negative:', FN)
        print('Trues:', TP+TN, 'False:', FP+FN, 'Ratio: True/All = ', (TP+TN)/len(input_data))
                
            
        
        
        
        
        
def use_neural_networkRNN(input_data, output):
    #input_data = raw5Fac.values.tolist()[:99]
    #input_data = [1, 69, 1, 0, 0, 0, 0, [0, 1]]
    #output = raw5Lab.values[:20]
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    
    prediction = neural_network_model(x)
#    with open('lexicon.pickle','rb') as f:
#        lexicon = pickle.load(f)
      
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,'C:\\Users\\TapperR\\Desktop\\compas\\compas-analysis\\model.cpkt')
#        current_words = word_tokenize(input_data.lower())
        
#        current_words = [lemmatizer.lemmatize(i) for i in current_words]
#        features = np.zeros(len(lexicon))

#        for word in current_words:
            #word = current_words[0]
#            if word.lower() in lexicon:
#                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
#                features[index_value] += 1

#        features = np.array(list(features))
                
        for i in range(len(input_data)):
            features = input_data[i]   
                    
                                        
            # pos: [1,0] , argmax: 0
            # neg: [0,1] , argmax: 1
            result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
            #print(prediction.eval(feed_dict={x:[features]}))
            #print(result)
            if result[0] == 1 and output[i][0][0] == 1:
                #print('True Positive')
                TP += 1
                        
            if result[0] == 1 and output[i][0][0] == 0:
                #print('False Positive')
                FP += 1
                
            if result[0] == 0 and output[i][0][0] == 0:
                #print('True Negative') 
                TN += 1
                
            if result[0] == 0 and output[i][0][0] == 1:
                #print('False Negative')
                FN += 1
                
        
        print('True Positive:', TP, 'False Positive:', FP, 'True Negative:', TN, 'False Negative:', FN)
        print('Trues:', TP+TN, 'False:', FP+FN, 'Ratio: True/All = ', (TP+TN)/len(input_data))        
        
        
        
        
        
        
        
        
        
        

#use_neural_network("He's an idiot and a jerk.")
            

#'sexInd', 'age', 'raceInd', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',     
            
 
start_time = time.time()           
            
use_neural_network(raw5Fac.values.tolist()[200:400], raw5Lab.values[200:400])



print("--- %s seconds ---" % (time.time() - start_time))



### it seems like that the neural network spits out the more positive results the higher the accuracy is, 
# although there are just 45% positives in the dataset

# dont add the model to the branch, because its too big
#git reset -- C:\Users\TapperR\Desktop\compas\compas-analysis\model.cpkt






start_time = time.time()           
            
use_neural_networkRNN(raw5Fac.values.tolist()[200:400], raw5Lab.values[200:400])



print("--- %s seconds ---" % (time.time() - start_time))

































