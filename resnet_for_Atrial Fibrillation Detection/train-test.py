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
#%%
import sys
import os
data_dir=os.getcwd()
sys.path.append(data_dir)
#%%
import tensorflow as tf
import read_template_data
import net_work
import datetime
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
data_path=os.getcwd()
summary_path=os.path.join(data_dir, 'summery/%s'%nowTime)
os.makedirs(summary_path)
#%%
tf.reset_default_graph() 
buchang=tf.placeholder('float32')
keep_prob=tf.placeholder('float32')
is_training = tf.placeholder('bool', [], name='is_training')
filenames1=tf.placeholder('string')
filenames2=tf.placeholder('string')


training_data=read_template_data.data_read(filenames1)
testing_data=read_template_data.val_read(filenames2)
iterator1 = training_data.make_initializable_iterator()
iterator2 = testing_data.make_initializable_iterator()
next_element1 = iterator1.get_next()
next_element2 = iterator2.get_next()






images, labels = tf.cond(is_training,
        lambda: (next_element1[0],next_element1[1]),
        lambda: (next_element2[0],next_element2[1]))


      
prediction=net_work.network1(images,True)

#%%

label_batch=tf.reshape(labels,[-1])
label_batch = tf.cast(label_batch, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                                       labels=label_batch, name='cross_entropy_per_example')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

#cross_entropy3= tf.reduce_mean(output_f2) 



#cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys -prediction)))      # loss                                               
                                               
train_step = tf.train.AdamOptimizer(buchang).minimize(cross_entropy)


#%%

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#%%

tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)



summary_op = tf.summary.merge_all()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12

#%%
sess.run(init_op)
#%%
coord = tf.train.Coordinator()

summary_writer1 = tf.summary.FileWriter(os.path.join(summary_path,'train'), sess.graph)
summary_writer2 = tf.summary.FileWriter(os.path.join(summary_path,'test'), sess.graph)

#%%
step=0
#%%

fileinput1 = os.path.join(data_dir, 'coverted_tf_data/training_data.tfrecords')
    
fileinput2 = os.path.join(data_dir, 'coverted_tf_data/testing_data.tfrecords')

sess.run(iterator1.initializer, feed_dict={ filenames1:fileinput1,filenames2:fileinput2}) 
sess.run(iterator2.initializer, feed_dict={ filenames1:fileinput1,filenames2:fileinput2}) 

 
    
    
    
#%%

for i in range(15000):
    step=step+1
    buchang_shuru=0.01
    sess.run(train_step, feed_dict={keep_prob: 0.9,buchang:buchang_shuru,is_training:True})
    if step % 50 == 0: 
       print(step)
       print(sess.run(cross_entropy_mean , feed_dict={ filenames1:fileinput1,filenames2:fileinput2,keep_prob: 1,is_training:True} ))
       print(sess.run(cross_entropy_mean , feed_dict={ filenames1:fileinput1,filenames2:fileinput2,keep_prob: 1,is_training:False} ))
       #output= sess.run(h_fc1, feed_dict={xs: inpfinal , ys: label , keep_prob: 1})
       s1=sess.run(summary_op,feed_dict={ keep_prob: 1,is_training:True} )
       s2=sess.run(summary_op,feed_dict={ keep_prob: 1,is_training:False} )
       summary_writer1.add_summary(s1, step)
       summary_writer2.add_summary(s2, step)
    if step % 20000 == 0 and step >500: 
       save_path=tf.train.Saver().save(sess,os.path.join(summary_path, '%s_steps.ckpt'%(step+0000)))  
print ("Training complete！")        
