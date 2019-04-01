# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 00:28:20 2019

@author: logcode
"""

import tensorflow as tf
from mnist_mlp_for_train import *
import numpy as np
import pylab

test_file_path = "G:\MyGit\DeepLearning\mnist_test1.csv"

test_data,test_label = load_test_data(test_file_path)
test_label = one_hot(test_label)

#def load_model():
#    with tf.Session() as sess:
#        saver = tf.train.import_meta_graph('save/my_test_model-99.meta')
#        saver.restore(sess, tf.train.latest_checkpoint("save/"))
#        print(sess.run('W1:0'))
#        print(sess.run('b1:0'))
#
#
#if __name__=='__main__':
#    load_model()

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph('save/my_test_model-99.meta')
    saver.restore(sess,tf.train.latest_checkpoint('save/'))
    
    ###get weights
    graph = tf.get_default_graph()
#    w1 = graph.get_tensor_by_name("W1:0")
#    b1 = graph.get_tensor_by_name("b1:0")
#    
#    print(".....................")
#    print(sess.run(w1))
#    print(".....................")
#    print(sess.run(b1))
    
    input_x = graph.get_operation_by_name("X").outputs[0]
    

    feed_dict = {"X:0":test_data,"Y:0":test_label}

    pred_y = tf.get_collection("predict")
    pred = sess.run(pred_y,feed_dict)[0]
    
#    print(".......................")
#    print("pred:",pred)
#    print(".......................")
    
    acc = tf.get_collection("acc")
    acc = sess.run(acc,feed_dict)[0]
    print(".......................")
    print("the accuracy is:",acc)
    print(".......................")
    ###########
    """
    对某个图像进行预测并显示
    """
    while(1):
        print("##############################")
        print("选择图片进行验证")
        n = input("输入想测试的图像序号0-9999：")
        if (n=='quit'):
            break
        else:
            n = int(n)
            
            temp_x = np.reshape(test_data[n],[1,784])
            temp_y = np.reshape(test_label[n],[1,10])
            
            im = np.reshape(temp_x,[-1,28])
            pylab.imshow(im)
            pylab.show()
            feed_dict = {"X:0":temp_x,"Y:0":temp_y}
        
            pred_y = tf.get_collection("predict")
            pred = sess.run(pred_y,feed_dict)[0]
            
            predict = sess.run(tf.argmax(pred,1))
            print(".......................")
            print("the predict is:",predict)
            print(".......................")
            

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






    
