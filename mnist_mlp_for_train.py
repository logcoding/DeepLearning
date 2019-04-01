# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:52:22 2019

@author: logcode
"""

import tensorflow as tf
import numpy as np
import csv

train_file_path = "G:\MyGit\DeepLearning\mnist_train.csv" 
test_file_path = "G:\MyGit\DeepLearning\mnist_test1.csv"


def toInt(array):
    """
    将数据转成int型
    """
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
                array[i,j] = 1
    return array

def load_train_data(file):
    """
    提取原始训练数据和标签
    """
    data_label = []
    with open(file,'r') as file:
        lines = csv.reader(file)
        for line in lines:
            data_label.append(line)
    data_label.remove(data_label[0])
    data_label = np.array(data_label)
    data = data_label[:,1:]
    label = data_label[:,0]
    
    return nomalizing(toInt(data)),toInt(label)

def load_test_data(file):
    """
    提取原始测试数据
    """
    data_label = []
    with open(file,'r') as file:
        lines = csv.reader(file)
        for line in lines:
            data_label.append(line)
    data_label = np.array(data_label)
    data = data_label[:,1:]
    label = data_label[:,0]
    return nomalizing(toInt(data)),toInt(label)

def one_hot(labels):
    """
    将label数字换成one_hot编码模式，方便后续损失函数的调用
    """

    m,n = np.shape(labels)
    new_label = np.zeros((n,10),dtype=int) 
    for i in range(n):
        temp = int(labels[0,i])
        new_label[i,temp] = 1
    return new_label
    
    
    
        
if __name__=='__main__':
    
    train_data,train_label = load_train_data(train_file_path)
    train_label = one_hot(train_label)
    test_data,test_label = load_test_data(test_file_path)
    test_label = one_hot(test_label)
    
    tf.reset_default_graph()
    ##隐藏层有1000个节点，迭代次数为100
    hidden,epochs,batch_size,learning_rate = 256,100,100,0.001
    
    X = tf.placeholder(tf.float32,[None,784],name='X')
    Y= tf.placeholder(tf.float32,[None,10],name='Y')
    
    W1 = tf.Variable(tf.random_normal([784,hidden]),name='W1')
    b1 = tf.Variable(tf.random_normal([hidden]),name='b1')
    
    W2 = tf.Variable(tf.random_normal([hidden,10]),name='W2')
    b2 = tf.Variable(tf.random_normal([10]),name='b2')
    
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X,W1),b1))
    
    pred = tf.add(tf.matmul(layer1,W2),b2)
    
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    saver = tf.train.Saver()
    tf.add_to_collection("predict",pred)
    tf.add_to_collection("acc",accuracy)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            cost_now = 0
            total_batch = int(len(train_data)/batch_size)
            for i in range(total_batch):
                batch_x,batch_y = train_data[i*batch_size:(i+1)*batch_size],\
                train_label[i*batch_size:(i+1)*batch_size]
                opt.run(feed_dict={X:batch_x,Y:batch_y})
            if((epoch+1)%20==0):
                saver.save(sess,"save/my_test_model",global_step=epoch)
                

            
        
        
        
        
                
#                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
