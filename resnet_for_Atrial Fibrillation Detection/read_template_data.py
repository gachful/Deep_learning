# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""mnist_template_matching."""
import tensorflow as tf
import os
#%%
def parse_function(data):
    features = tf.parse_single_example(
            data,
            features={
                # dense data
                'data_ECG': tf.FixedLenFeature([], tf.string),
                'spectro_ECG': tf.FixedLenFeature([], tf.string),                                
                'label': tf.FixedLenFeature([], tf.string),                                                
            }
            )
    
    data1 = tf.decode_raw(features['data_ECG'], tf.float64)
#    image_shape = tf.stack([56, 56,3])
#    image = tf.reshape(image, image_shape)
    data2 = tf.decode_raw(features['spectro_ECG'], tf.float64)
    data2_shape = tf.stack([33,280,1])
    data2 = tf.reshape(data2,data2_shape)
    label = tf.decode_raw(features['label'], tf.float64)
    label_shape = tf.stack([1])
    label = tf.reshape(label, label_shape)
    label=tf.cast(label,tf.float32)
    data1=tf.cast(data1,tf.float32)
    data2=tf.cast(data2,tf.float32)
    return data2,label
#%%
def data_read(filenames1):   
    data = tf.data.TFRecordDataset(filenames1)   
    datanew=data.map(parse_function)
    dataout = datanew.repeat().shuffle(buffer_size=1000)
    dataout=dataout.batch(100)
    dataout=dataout.prefetch(1)   
    return dataout
#%%    
def val_read(filenames2):
    
    data = tf.data.TFRecordDataset(filenames2)
    datanew=data.map(parse_function)
    dataout = datanew.repeat().shuffle(buffer_size=1000)
    dataout=dataout.batch(100)
    dataout=dataout.prefetch(1)
    return dataout
    # Create a queue that produces the filenames to read.

   