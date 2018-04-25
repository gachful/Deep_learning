import sys
import os
data_dir=os.getcwd()
sys.path.append(data_dir)
from resnet import *

def network1(x,is_training):    
    
    prediction = inference_small(x,
                             num_classes=4,
                             is_training=is_training,
                             use_bias=(False),
                             num_blocks=14                          
                             )
    
    prediction=prediction
    return prediction
    
def network2(x,is_training):    
    
    prediction = inference(x, 
              is_training=is_training,
              num_classes=4,
              num_blocks=[2, 2, 2, 2],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True)
      
    
    prediction=prediction
    return prediction    
   
def top_k_error(predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = batch_size = float(100)#%%predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        print(batch_size )
        print( num_correct) 
        return (batch_size - num_correct) / float(batch_size) 
    
