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


#%%

data_path=os.getcwd()
tf_dataname=os.path.join(data_path, 'coverted_tf_data/tf_data.tfrecords')
writer= tf.python_io.TFRecordWriter(tf_dataname) 


data_loca =os.path.join(data_path, 'data_folder\REFERENCE.csv')
line_list = open(data_loca).readlines()
#要生成的文件
a=line_list[0]
b=a.split(',')

labels=[line_list[i].split(',')[1] for i in range(len(line_list)) ]
labels=[labels[i][0] for i in range(len(labels)) ]
file_names=[line_list[i].split(',')[0]+'.mat' for i in range(len(line_list)) ]
file_index=[line_list[i].split(',')[0] for i in range(len(line_list)) ]
write_lines=[] 
label_dict={'N':0,'A':1,'O':2,'~':3} 


#%%
def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  
for i in range(len(file_names)) :
    filepath=os.path.join(data_path,'data_folder',file_names[i])
    data = scio.loadmat(filepath) 
    data2=np.float64(np.reshape(data['val'],[-1,1])) 
    label2=np.array(label_dict[labels[i]],dtype='float64')
    #data2.dtype='float64'
    
    example = tf.train.Example(features=tf.train.Features(feature={  
            'data_ECG': _bytes_feature(data2.tostring()),
            'label': _bytes_feature(label2.tostring()) 
        }))  

    writer.write(example.SerializeToString())  #序列化为字符串
writer.close()

