# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:18:18 2019

@author: logcode
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

path_train = "G:/MyGit/DeepLearning/train.csv"
path_test = "G:/MyGit/DeepLearning/test.csv"

train = pd.read_csv(path_train)
test = pd.read_csv(path_test)

# Convert to numpy
train = np.asarray(train, np.float32)
test = np.asarray(test, np.float32)

X_train = train[:, 1:]
y_train = np.reshape(train[:, :1], (train.shape[0], ))
del train
del path_train
del path_test

learning_rate = 0.001
batch_size = 128
num_steps = 2000
num_input = 784
num_classes = 10
dropout = 0.25

def conv_net(X,classes,drop):
    x = tf.reshape(X,[-1,28,28,1])
    conv1_1 = tf.layers.conv2d(x,16,5,activation=tf.nn.relu,padding="SAME")
    conv1_2 = tf.layers.conv2d(conv1_1,16,5,activation=tf.nn.relu,padding="SAME")
    conv1 = tf.concat([conv1_1,conv1_2],axis=3)
    conv1_3 = tf.layers.max_pooling2d(conv1,2,2)
    conv1_4 = tf.layers.conv2d(conv1, 32, 3, activation=tf.nn.relu, strides=(2, 2), padding='SAME')
    conv1 = tf.concat([conv1_3, conv1_4], axis=3)
    conv2_1 = tf.layers.conv2d(conv1, 64, 1, activation=tf.nn.relu, padding='SAME')
    conv2_2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, padding='SAME')
    conv2 = tf.concat([conv2_1, conv2_2], axis=3)
    conv2_3 = tf.layers.max_pooling2d(conv2, 2, 2)
    conv2_4 = tf.layers.conv2d(conv1, 128, 3, activation=tf.nn.relu, strides=(2, 2), padding='SAME')
    conv2 = tf.concat([conv2_3, conv2_4], axis=3)
    fc1 = tf.contrib.layers.flatten(conv2)
    fc1 =tf.layers.dense(fc1, 9000)
    fc1 = tf.layers.dropout(fc1, rate=dropout)
    fc2 = tf.layers.dense(fc1, 1024)
    out = tf.layers.dense(fc2, classes)
    return out

def model_fn(features, labels, mode):
  
  logits = conv_net(features, num_classes, dropout)
  
  # Get prediction of model output
  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }
  # for PREDICT mode
  if(mode==tf.estimator.ModeKeys.PREDICT):
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
  # for TRAIN mode
  if(mode==tf.estimator.ModeKeys.TRAIN):
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step = tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss_op, train_op = train_op)
  # for Evaluation
  if(mode==tf.estimator.ModeKeys.EVAL):
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    metrics = {'accuracy': accuracy}
    return tf.estimator.EstimatorSpec(mode, loss=loss_op, eval_metric_ops=metrics)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x=X_train, y=y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)
model = tf.estimator.Estimator(model_fn)
# Train the Model

model.train(input_fn, steps=num_steps)

# Test model 
input_fn_test = tf.estimator.inputs.numpy_input_fn(x=test,num_epochs=1,shuffle=False)
predictions = model.predict(input_fn=input_fn_test)
cls = [p['classes'] for p in predictions]
cls_pred = np.array(cls, dtype='int').squeeze()
# Convert to csv to submit to Kaggle Competitive
submit = pd.DataFrame()
submit['ImageId'] = range(1, test.shape[0]+1)
submit['Label'] = cls_pred
submit.to_csv("submit.csv", index=False)
    
