# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:13:35 2019

@author: logcode
"""

import numpy as np
import tensorflow as tf
import csv

def toInt(array):
    array = np.mat(array)
    m,n = np.shape(array)
    newArray = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            newArray[i,j] = int(array[i,j])
    return newArray
    
def nomalizing(array):
    m,n = np.shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j] != 0:
                array[i,j] == 1
    return array

def loadtraincsv():
    l = []
    with open("train.csv") as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = np.array(l)
    data = l[:,1:]
    label = l[:,0]
    return nomalizing(toInt(data)),toInt(label)

def loadtestcsv():
    l = []
    with open("test.csv") as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = np.array(l)
    return nomalizing(toInt(data))

def classify(inX,dataSet,labels,k):
    """
    tile函数意义是将矩阵inX重复几次，在这里是在行数上重复训练次
    argsort函数对数据进行排序，按小到大返回索引
    """
    inX = np.mat(inX)
    dataSet = np.mat(dataSet)
    labels = np.mat(labels)
    dataSetSize = np.shape(dataSet)[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqdiffMat = np.array(diffMat)**2
    sqDistances = np.sum(sqdiffMat,axis=1)
    distances = sqDistances**0.5
    sortDistIndex = distances.argsort() 
    print(sortDistIndex)
    classCount = {}
    for i in range(k):
        voteLabel = labels[0,sortDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]
    
def saveFile(result):
    with open("result.csv","wb") as file:
        myWrite = csv.writer(file)
        for i in result:
            temp = []
            temp.append(i)
            myWrite.writerows(temp)

def handwritingClassTest():  
    trainData,trainLabel=loadtraincsv()  
    testData=loadtestcsv()   
    m,n=np.shape(testData)  
    errorCount=0  
    resultList=[]  
    for i in range(m):  
         classifierResult = classify(testData[i], trainData, trainLabel, 5)  
         resultList.append(classifierResult)  
    print("\nthe total number of errors is: %d" % errorCount)  
    print("\nthe total error rate is: %f" % (errorCount/float(m))) 
    saveFile(resultList)  
            

if __name__=='__main__':
    handwritingClassTest()