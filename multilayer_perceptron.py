#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:29:33 2020

@author: benoitgeorges
"""
import pandas as pd
import numpy as np
import seaborn as sns;
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocessing_data(df):
    X = df.iloc[:, 2:]
    X = np.array(X)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X = X.T
    # X_train, X_test = X[:,:450], X[:,450:]
    Y = df.iloc[:, 1]
    Y = np.array(Y)
    Y = np.where(Y == 'B', 0, 1)
    
    # Y = Y.reshape((1, len(Y)))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    # Y_train, Y_test = Y[:,:450], Y[:,450:]
    
    enc = OneHotEncoder(sparse=False, categories='auto')
    Y_train = enc.fit_transform(Y_train.reshape(len(y_train), -1))
    Y_test = enc.transform(Y_test.reshape(len(y_test), -1))
    
    return X_train, X_test, Y_train, Y_test


def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def relu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    return (np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True))

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def linear_activation_forward(A_prev, W, b, activation):    
    Z = np.dot(W, A_prev) + b
    
    if activation == "softmax":
        A = softmax(Z)
        print(A.shape)
    elif activation == "sigmoid":
        A = sigmoid(Z)
        
    elif activation == "relu":
        A = relu(Z)
    
    cache = ((A_prev, W, b), Z)
    
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    caches.append(cache)
            
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis= 1, keepdims=True)
    cost = np.squeeze(cost)
    return cost

def sigmoid_backward(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax_backward(dA, Z):
    return dA * Z * (1 - Z)

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, AL, Y, cache, activation):
    linear_cache, Z = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "softmax":
        dZ = AL - Y.T
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    # Y = Y.reshape(AL.shape)

    # dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, None, None, current_cache, activation = "softmax")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], AL, Y, current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

def cross_entropy(AL,Y):
    cost = -np.mean(Y * np.log(AL.T + 1e-8))
    return cost

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    costs = []

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        
        cost = cross_entropy(AL,Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost:
            print ("epoch %i/%i - loss: %f - val_loss: ?" %(i, num_iterations, cost))
        if print_cost:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

if __name__ == '__main__':
    df=pd.read_csv('data.csv', header=None)
    X_train, X_test, Y_train, Y_test = preprocessing_data(df)
    layers_dims = [X_train.shape[0], 15, 7, 2]
    parameters = L_layer_model(X_train, Y_train, layers_dims, learning_rate = 0.0075, num_iterations = 70, print_cost=False)
    Y_hat, _ = L_model_forward(X_test, parameters)
    Y_hat = np.where(Y_hat < 0.5, 0, 1)
    
    Y_hat = Y_hat.T
    
    accuracy = (Y_hat == Y_test).mean()
    print(accuracy)