#Randomize the data somehow and then grab the first 90%. This will the be the training data
# load numpy array from file
import sys
from numpy import loadtxt
from numpy import load
import numpy as np
# load array

TRAININGSETPERCENT = 0.9

#data_array = loadtxt(sys.argv[1], delimiter=',')
data_array = load(sys.argv[1])

#shuffle
np.random.shuffle(data_array)
rows_in_array = data_array.shape[0]
number_of_training_sets = int(TRAININGSETPERCENT*rows_in_array)
print("rows_in_array: ",rows_in_array)
print("number_of_training_sets: ",number_of_training_sets)
#Seperate the data into X_train and Y_train

X_TRAIN = data_array[:number_of_training_sets,:-1]
X_TEST = data_array[number_of_training_sets:,:-1]
Y_TRAIN = data_array[:number_of_training_sets,-1]
Y_TEST = data_array[number_of_training_sets:,-1]

print("X_TRAIN: ", X_TRAIN.shape)
print("X_TEST: ", X_TEST.shape)
print("Y_TRAIN: ", Y_TRAIN.shape)
print("Y_TEST: ", Y_TEST.shape)

#Grab the rest, this will be the testing data
#Seperate the data into X_test and Y_test
