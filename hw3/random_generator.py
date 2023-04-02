# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:14:35 2023

@author: xuan
"""

import numpy as np   

def univariate_gaussian(m, v):
    """ By Central Limit Theorem, 
        generate 12 number uniform between [0,1], add them up and substract 6
    """
    x = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    return x * (v ** 0.5) + m

def polynomial(x):
    poly_X = np.power(x, range(n)).reshape(1,n)
    return poly_X

def polynomial_model(n, a, w):
    """ calculate the polynomial model, y = w * x + error
    
        Args:
        n (int): number of polynomial basis
        a (float): variance
        w (array): weights (shape = (1, n))
    
        Returns:
        x (float), y (float)
    """
    x = np.random.uniform(-1.0, 1.0)
    error = univariate_gaussian(0, a)
    y = float(np.dot(w, polynomial(x).T) + error)
    return x, y

if __name__ == '__main__':
    m = 0
    v = 1
    random_value = univariate_gaussian(m, v)
    print(random_value)
    
    precision = 1
    n = 4
    var = 1
    a = 1/var
    w = np.array([1.0, 2.0, 3.0, 4.0])
    random_point = polynomial_model(n, a, w)
    print(random_point)