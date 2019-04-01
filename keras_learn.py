# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:46:57 2019

@author: logcode
"""

from keras.models import Sequential
from keras.layers import Dense,Activation
model = Sequential([Dense(32,input_shape=(784,)),
                    Activation('relu'),
                    Dense(10),
                    Activation('softmax'),
                    ])