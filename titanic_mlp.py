# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:07:05 2019

@author: logcode
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import csv

path_train = "titanic_train.csv"
path_test = "titanic_test.csv"

def tofloat(array):
    array = np.mat(array)
    m,n = np.shape(array)
    newArray = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            newArray[i,j] = float(array[i,j])
    return newArray
    
def loaddata(path):
    l = []
    with open(path) as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = np.array(l)
    l = np.delete(l,(0,3,8),1)
    m,n = np.shape(l)
    for i in range(m):
        if l[i,2]=='male':
            l[i,2] = 1
        else:
            l[i,2] = 0
    for i in range(m):
        if l[i,7]=='':
            l[i,7] = 0
        else:
            l[i,7] = 1
    for i in range(m):
        if l[i,8] == 'C':
            l[i,8]=1
        elif l[i,8] == 'Q':
            l[i,8] = 2
        else:
            l[i,8]=3
    temp = tofloat(l)
#    for i in range(m):
#        if l[i,3] == '':
#            l[i,3] = np.mean(l[:,3])
    
            
#    data = [l for]
    
    
    return temp


if __name__=='__main__':
    train = loaddata(path_train)
        

