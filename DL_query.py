
import DLfunctions
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
