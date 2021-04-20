
import DLfunctions as dl
import numpy as np

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


#initialize_parameters

URLSTRINGLENGTH = 3063
n_x = URLSTRINGLENGTH
n_h = 1
n_y = 1
X = training_dataset_X
Y = training_dataset_Y
num_iterations = 100
learning_rate = 0.009

parameters = dl.initialize_parameters(n_x, n_h, n_y)

#forward propagation
params, grads, costs = dl.optimize(parameters["W1"], parameters["b1"], X, Y, num_iterations, learning_rate)
