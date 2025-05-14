#Importación de librerías
from google.colab import drive
from google.colab import files
import sys
#drive.mount('/content/drive')
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
%run '/content/funciones.py'
from math import e

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)

#Carga de los datasets
train_x = np.load('x_train.npy')
train_y = np.load('y_train.npy')
test_x = np.load('x_std_test.npy')
test_y = np.load('y_std_test.npy')
print(train_x.shape)
print(train_y.shape)

print(train_x)
print('\n')
print(train_y)
print('\n')
print(test_x)
print('\n')
print(test_y)

#Se definen el tamaño de las capas y la cantidad de capas
layers_dims = [784, 6, 5, 10]

#Función que integra todo el modelo
def L_layer_model(X, Y, layers_dims, learning_rate = 0.70, num_iterations = 1400, print_cost=False):#lr was 0.009
    

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
        print(cost)
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
        #if print_cost and i % 20 == 0:
            #print 
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

    #print(train_x)
print(type(train_x))
print(type(train_x[0][0]))

#print(train_y)
print(type(train_y))
print(type(train_y[0][0]))


#Ejecuta el modelo
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1400, print_cost = True)

pred_train = predict(train_x, train_y, parameters)

pred_test = predict(test_x, test_y, parameters)

np.set_printoptions(threshold=np.inf)
print(parameters)