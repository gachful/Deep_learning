#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Call this function to generate tf_record from mat data'''

from scipy import signal
from scipy import io as scio
import numpy as np
import os
import random as rd
import array
import tensorflow as tf
import matplotlib.pyplot as plt

#%%

data_path=os.getcwd()
tf_dataname=os.path.join(data_path, 'coverted_tf_data/tf_data.tfrecords')
datanum=sum(1 for _ in tf.python_io.tf_record_iterator(tf_dataname))
print(datanum)
#%%

def parse_function(data):
    features = tf.parse_single_example(
            data,
            features={
                # dense data
                'data_ECG': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),                                                
            }
            )
    
    image = tf.decode_raw(features['data_ECG'], tf.float64)
#    image_shape = tf.stack([56, 56,3])
#    image = tf.reshape(image, image_shape)
    label = tf.decode_raw(features['label'], tf.float64)
#    label_shape = tf.stack([56, 56,1])
#    label = tf.reshape(label, label_shape)
    label=tf.cast(label,tf.float32)
    image=tf.cast(image,tf.float32)
    return image,label

data = tf.data.TFRecordDataset(tf_dataname)
datanew=data.map(parse_function)


iterator1 = datanew.make_one_shot_iterator()

next_element1 = iterator1.get_next()
images, labels = [next_element1[0],next_element1[1]]
#%%
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init_op)
#%% run this repeatly


ECG,label=sess.run((images,labels))
  
plt.plot(ECG)
plt.title(str(label))
