import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
import numpy as np
from reader import Reader
import random

class task_1(object):
    def __init__(self, csvFilePath):
        # Universal Doc
        self.csvDoc = csvFilePath
        # Classes
        R = Reader(csvFilePath)
        self.df = R.data
    
    def trainTest(self):
        df = self.df
        df = df[['CPI','Weekly_Sales']]
        y = df.iloc[:,1:2].values#Dependent Vars
        X = df.iloc[:,0:1].values #Independent Var
        #Var Training
        X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7)

        return X, y, X_train, X_test, y_train, y_test, df
    
    # Build a linear model in order to watch out polynomial degree   
    def linearReg(self, X, y):
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        return lin_reg

    def linearGraph(self):
        X, y, X_train, X_test, y_train, y_test, df = self.trainTest()
        linReg = self.linearReg(X,y)
        plt.scatter(X, y, color='red')
        plt.plot(X, linReg.predict(X), color='blue')
        plt.title('Truth or Bluff (Linear Regression)')
        plt.xlabel('CPI')
        plt.ylabel('Weekly_Sales')
        plt.show()
        
    def polyReg(self, X, y):
        poly_reg = PolynomialFeatures(degree=16)
        X_poly = poly_reg.fit_transform(X)
        pol_reg = LinearRegression()
        pol_reg.fit(X_poly, y)

        return poly_reg, pol_reg

    def polyGraph(self):
        X, y, X_train, X_test, y_train, y_test, df = self.trainTest()
        polyReg, polReg = self.polyReg(X,y)
        plt.scatter(X, y, color='red')
        plt.plot(X, polReg.predict(polyReg.fit_transform(X)), color='blue')
        plt.title('Truth or Bluff (Linear Regression)')
        plt.xlabel('CPI')
        plt.ylabel('Weekly_Sales')
        plt.show()

driver = task_1('Walmart.csv')
driver.polyGraph()