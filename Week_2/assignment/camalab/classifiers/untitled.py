import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log
import operater

%matplotlib inline

# Read data from train.csv and y_train.csv
train_feature = pd.read_csv('train.csv')
tarin_label = pd.read_csv('y_train.csv')
test_feature = pd.read_csv('test.csv')

train_dataset = pd.merge(train_feature, tarin_label, on = 'ID')

def calculateShannonEntropy(dataset):
    # Todo 1: calculate the entropy given a dataset
   
    numEntries = len(dataset) # There are n rows inside
    labelCounts = {} # Create dictionary for classification

    for featureVector in dataset:
    	currentLabel = featureVector[-1] # Get the last-row data
    	if currentLabel not in labelCounts.keys():
    		labelCounts[currentLabel] = 0
    	labelCounts[currentLabel] += 1

    total_entropy = 0.0
    for key in labelCounts:
    	proportion_k = float(labelCounts[key]) / numEntries
    	total_entropy += -(proportion_k * log(proportion_k, 2))

    return total_entropy

def choose_best_feature_to_split(features, labels):
    # Todo 2: return the best feature based on the maximum number of information gain
    numFeatures = len(features[0]) - 1
    baseEntropy = calculateShannonEntropy(features, labels)
    bestInfoGain = 0
    best_feature = 0

    for i in range(numFeatures):
    	featureList = [number[i] for number in features] # enum for one attribute
    	uniqualValues = set(featureList) # no-relace attribute
    	newEntropy = 0

    	for value in uniqualValues:
    		sub_dataset = split_dataset(features, labels, current_feature, value)
    		proportion_k = len(sub_dataset) / float(len(features))
    		newEntropy += proportion_k * calculateShannonEntropy(sub_dataset) # sum(ShannonEntropy)
    	infoGain = baseEntropy - newEntropy # infoGain

    	# bestInfoGain
    	if (infoGain > bestInfoGain):
    		bestInfoGain = infoGain
    		best_feature = i

    return best_feature

def split_dataset(features, labels, current_feature, value):
    # Todo 3: Split the dataset via current selected feature and it's value
    # For example, when current_feature is TLS(top-left-square), and the value is 'o', 
    # the task is that return the subdataset in which all "TLS" is equal to 'o'
    sub_dataset = []

    for featureVector in features:
    	if current_feature == value:
    		reduceFeatureVector = featureVector[ :axis]
    		reduceFeatureVector.extend(featureVector[axis+1: ])  
    		sub_dataset.append(reduceFeatureVector)

    return sub_dataset

def create_decision_tree(features, labels):
    # Todo 4: Create a decision tree by recursion
    #
    # Tips: Set appropriate boundary conditions; 
    #       think about the values one by one; 
    #       Use the three functions defined before.
    
    
    
    
    return decision_tree


# Get test_dataset from test.csv

def predict(decision_tree, test_dataset):
    # Todo 5
    