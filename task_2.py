import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
import numpy as np
from reader import Reader
import random
import pandas as pd
import math
import operator

class task_2(object):
    def __init__(self, csvFilePath):
        # Universal Doc
        self.csvDoc = csvFilePath
        # Classes
        R = Reader(csvFilePath)
        self.df = R.data
    

    # Defining a function which calculates euclidean distance between two data points
    def euclideanDistance(self,data1, data2, length):
        distance = 0
        for x in range(length):
            distance += np.square(data1[x] - data2[x])
        return np.sqrt(distance)

    def trainTest(self):
        df = self.df
        df = df[['CPI','Weekly_Sales']]
        y = df.iloc[:,1:2].values#Dependent Vars
        X = df.iloc[:,0:1].values #Independent Var
        #Var Training
        X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7)

        return X, y, X_train, X_test, y_train, y_test, df

    # Defining our KNN model
    def knn(self, trainingSet, testInstance, k):
        
        distances = {}
        sort = {}
    
        length = testInstance.shape[1]
        
        #### Start of STEP 3
        # Calculating euclidean distance between each row of training data and test data
        for x in range(len(trainingSet)):
            
            #### Start of STEP 3.1
            dist = self.euclideanDistance(testInstance, trainingSet.iloc[x], length)

            distances[x] = dist[0]
            #### End of STEP 3.1
    
        #### Start of STEP 3.2
        # Sorting them on the basis of distance
        sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
        #### End of STEP 3.2
    
        neighbors = []
        
        #### Start of STEP 3.3
        # Extracting top k neighbors
        for x in range(k):
            neighbors.append(sorted_d[x][0])
        #### End of STEP 3.3
        classVotes = {}
        
        #### Start of STEP 3.4
        # Calculating the most freq class in the neighbors
        for x in range(len(neighbors)):
            response = trainingSet.iloc[neighbors[x]][-1]
    
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        #### End of STEP 3.4

        #### Start of STEP 3.5
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return(sortedVotes[0][0], neighbors)
        #### End of STEP 3.5

    def knnTest(self):
        testSet = [[7.2, 3.6, 5.1, 2.5]]
        test = pd.DataFrame(testSet)
        print('\n\nWith 1 Nearest Neighbour \n\n')
        k = 1
        #### End of STEP 2
        # Running KNN model
        result,neigh = self.knn(self.df, test, k)

        # Predicted class
        print('\nPredicted Class of the datapoint = ', result)

        # Nearest neighbor
        print('\nNearest Neighbour of the datapoints = ',neigh)


        print('\n\nWith 3 Nearest Neighbours\n\n')
        # Setting number of neighbors = 3 
        k = 3 
        # Running KNN model 
        result,neigh = self.knn(self.df, test, k) 

        # Predicted class 
        print('\nPredicted class of the datapoint = ',result)

        # Nearest neighbor
        print('\nNearest Neighbours of the datapoints = ',neigh)

        print('\n\nWith 5 Nearest Neighbours\n\n')
        # Setting number of neighbors = 3 
        k = 5
        # Running KNN model 
        result,neigh = self.knn(self.df, test, k) 

        # Predicted class 
        print('\nPredicted class of the datapoint = ',result)

        # Nearest neighbor
        print('\nNearest Neighbours of the datapoints = ',neigh)
"""
class task_3(self):
    pass
"""
csvPath = 'Oranges vs Grapefruit.csv' 
driver = task_2(csvPath)
driver.knnTest()