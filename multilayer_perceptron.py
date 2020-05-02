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
import sys
import pickle

def writeCSV(df):
    df.to_csv('parameters.csv')

def preprocessing_data(df, Lactivation="softmax"):
    Y = np.array(df[1])
    Y = np.where(Y == 'B', 0, 1)
    X = np.array(df.iloc[:, 2:])
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    X_train = X_train.T
    X_test = X_test.T
    if (Lactivation == "sigmoid"):
        Y_train = Y_train.reshape((1, len(Y_train)))
        Y_test = Y_test.reshape((1, len(Y_test)))
    
    if (Lactivation == "softmax"):
        enc = OneHotEncoder(sparse=False, categories='auto')
        Y_train = enc.fit_transform(Y_train.reshape(len(Y_train), -1))
        Y_test = enc.transform(Y_test.reshape(len(Y_test), -1))
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

def sigmoid_backward(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_activation_forward(A_prev, W, b, activation):    
    Z = np.dot(W, A_prev) + b
    
    if activation == "softmax":
        A = softmax(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)
        
    elif activation == "relu":
        A = relu(Z)
    
    cache = ((A_prev, W, b), Z)
    
    return A, cache

def L_model_forward(X, parameters, activation, Lactivation):
    caches = []
    A = X
    L = len(parameters) // 2 
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation)
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], Lactivation)
    caches.append(cache)
            
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis= 1, keepdims=True)
    cost = np.squeeze(cost)
    return cost

def cross_entropy(AL,Y):
    cost = -np.mean(Y * np.log(AL.T + 1e-8))
    return cost

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

def L_model_backward(AL, Y, caches, activation, Lactivation):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    dAL = None
    if Lactivation == "sigmoid":
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, AL, Y, current_cache, Lactivation)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], None, None, current_cache, activation)
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

def L_layer_model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.0075, num_iterations = 3000, activation="relu", Lactivation="softmax", Fcost="crossentropy", print_cost=False):#lr was 0.009
    costs = []
    lst_val_loss = []

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X_train, parameters, activation, Lactivation)
        AL_test, _ = L_model_forward(X_test, parameters, activation, Lactivation)
        
        if Fcost == "crossentropy":
            cost = cross_entropy(AL,Y_train)
            val_loss = cross_entropy(AL_test,Y_test)
        else:
            cost = compute_cost(AL, Y_train)
            val_loss = compute_cost(AL_test,Y_test)

        grads = L_model_backward(AL, Y_train, caches, activation, Lactivation)

        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost  and i % 100 == 0:
           print ("epoch %i/%i - loss: %f - val_loss: %f" %(i, num_iterations, cost, val_loss))
        if print_cost  and i % 100 == 0:
            costs.append(cost)
            lst_val_loss.append(val_loss)
            
    plt.plot(np.squeeze(costs))
    plt.plot(np.squeeze(lst_val_loss))
    plt.legend(['loss', 'val_loss'])
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def accuracy(Y, X, parameters, activation="relu", Lactivation="softmax"):
    Y_hat, _ = L_model_forward(X, parameters, activation, Lactivation)
    if Lactivation == "softmax":
        Y_hat = np.argmax(Y_hat, axis=0)
        Y = np.argmax(Y, axis=1)
        Y_hat = Y_hat.T
    if Lactivation == "sigmoid":
        Y_hat = np.where(Y_hat < 0.5, 0, 1)

    result = (Y_hat == Y).mean()
    print(result)
    return result

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SyntaxError("Insufficient arguments.")
    else:
        df=pd.read_csv('data.csv', header=None)
        X_train, X_test, Y_train, Y_test = preprocessing_data(df, Lactivation="sigmoid")
    if str(sys.argv[1]) == "training":
        layers_dims = [X_train.shape[0], 40, 20, 10, 5, 1]
        parameters = L_layer_model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.007, num_iterations = 35900, activation="relu", Lactivation="sigmoid", Fcost="cost", print_cost=True)
        with open('parameters.pkl', 'wb') as output:
            pickle.dump(parameters, output)
    elif str(sys.argv[1]) == "prediction":
        with open("parameters.pkl", "rb") as fp:
            parameters = pickle.load(fp)
        accuracy(Y_test, X_test, parameters, activation="relu", Lactivation="sigmoid")
        