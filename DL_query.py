
import DLfunctions as dl
import numpy as np
import sys
from numpy import loadtxt
from numpy import load
import pickle

#grab the training examples
#as a subset of the training, classify some of them as testing examples

#
# ### 3.3 - General methodology
#
# As usual you will follow the Deep Learning methodology to build the model:
#     1. Initialize parameters / Define hyperparameters
#     2. Loop for num_iterations:
#         a. Forward propagation
#         b. Compute cost function
#         c. Backward propagation
#         d. Update parameters (using parameters, and grads from backprop)
#     4. Use trained parameters to predict labels


URLSTRINGLENGTH = 3063
n_x = URLSTRINGLENGTH
n_h = 1
n_y = 1
num_iterations = 100
learning_rate = 0.009

#data_array = loadtxt(sys.argv[1], delimiter=',')
#data_array = load(sys.argv[1])

#shuffle
#np.random.shuffle(data_array)
#rows_in_array = data_array.shape[0]
#number_of_training_sets = int(TRAININGSETPERCENT*rows_in_array)
#print("rows_in_array: ",rows_in_array)
#print("number_of_training_sets: ",number_of_training_sets)
#Seperate the data into X_train and Y_train
#
#X_TRAIN = data_array[:number_of_training_sets,:-1].T
#X_TEST = data_array[number_of_training_sets:,:-1].T
#Y_TRAIN = data_array[:number_of_training_sets,-1].T
#Y_TEST = data_array[number_of_training_sets:,-1].T

X_TRAIN = load("X_TRAIN.npy")
X_TEST = load("X_TEST.npy")
Y_TRAIN = load("Y_TRAIN.npy")
Y_TEST = load("Y_TEST.npy")
print("X_TRAIN: ", X_TRAIN.shape)
print("X_TEST: ", X_TEST.shape)
print("Y_TRAIN: ", Y_TRAIN.shape)
print("Y_TEST: ", Y_TEST.shape)

#initialize_parameters

parameters = dl.initialize_parameters(n_x, n_h, n_y)
print(parameters["W1"].shape)
print(parameters["b1"].shape)
#forward propagation
params, grads, costs = dl.optimize(parameters["W1"].T, parameters["b1"], X_TRAIN, Y_TRAIN, num_iterations, learning_rate)

# Retrieve parameters w and b from dictionary "parameters"
w = params["w"]
b = params["b"]

print ("w: ", w)
print ("b: ", b)
#save dictionary values
print("What file would you like to save the hyperparameters to?")
save_dictionary_filename = input()
a_file = open(save_dictionary_filename, "wb")
pickle.dump(params, a_file)
a_file.close()


#Run the Testing set through the predictor and then do a check.

Y_prediction_test = dl.predict(w, b, X_TEST)
Y_prediction_train = dl.predict(w, b, X_TRAIN)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_TRAIN)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_TEST)) * 100))
