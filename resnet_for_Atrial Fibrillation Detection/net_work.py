import sys

#%%
sys.path.append('E:/Yu/projects/mnist_template/resnet4')
from resnet import *

def network1(x,is_training):    
    
    prediction = inference_small(x,
                             num_classes=4,
                             is_training=is_training,
                             use_bias=(False),
                             num_blocks=3                           
                             )
    
    prediction=prediction
    return prediction
    
def network2(x,is_training):    
    
    prediction = inference(x, 
              is_training=is_training,
              num_classes=4,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True)
      
    
    prediction=prediction
    return prediction    
   

