import sys

#%%
sys.path.append('E:/Yu/projects/mnist_template/resnet4')
from resnet import *

def network1(x,is_training):    
    
    prediction = inference_small(x,
                             num_classes=4,
                             is_training=is_training,
                             
                             num_blocks=7                          
                             )
    
    prediction=prediction
    return prediction
    
def network2(x,is_training):    
    
    prediction = inference(x, 
              is_training=is_training,
              num_classes=4,
              num_blocks=[2, 2, 2, 2],  # defaults to 50-layer network             
              bottleneck=True)
      
    
    prediction=prediction
    return prediction    
   

