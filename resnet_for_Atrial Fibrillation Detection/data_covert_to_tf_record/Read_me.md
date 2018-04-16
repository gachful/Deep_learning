# convert data to tf_record format

down load the data from  https://www.physionet.org/challenge/2017/ and extract to data_folder

run tf_data_gen.py to generate tfrecord format data 

training and testing dataset size will be controlled by the split_ratio parameter

tf_data_check.py will check the converted data