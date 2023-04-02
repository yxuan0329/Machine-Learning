# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:57:57 2023

@author: xuan
"""

import numpy as np
import argparse

def univariate_gaussian(m, v):
    """ By Central Limit Theorem, 
        generate 12 number uniform between [0,1], add them up and substract 6
    """
    x = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    return x * (v ** 0.5) + m

def sequential_generator(m, v):
    n = 0
    prev_mean = m
    prev_var = v + m * n
    mean = prev_mean
    var = prev_var
    threshold = 1e-4
    while True:
        x_new = univariate_gaussian(m, v)
        mean = ((prev_mean * n) + x_new) / (n+1)
        # tmp = prev_var + ((x_new - mean) ** 2 / (n+1))
        # var = n / (n+1) * tmp
        
        var = 1/(n+1) * (x_new - mean) ** 2
        n += 1
        
        print("{} Add data point: {}".format(n, x_new))
        print("Mean = {}\tVariance = {}\n".format(mean, var)) 

        if abs(mean-prev_mean)<= threshold and abs(var-prev_var)<= threshold:
            break
        prev_mean = mean
        prev_var = var
        

def parse_arguments():
    """ Parse all arguments """
    parser = argparse.ArgumentParser(description='Sequential estimator')
    parser.add_argument('mean', help='expectation value or mean', default=0.0, type=float)
    parser.add_argument('var', help='variance', default=1.0, type=float)
    return parser.parse_args()

if __name__ == '__main__': 
    # input mean and variance
    args = parse_arguments()
    mean = args.mean
    var = args.var
    # mean = 3
    # var = 5
    print(f"Data point source function: N({mean}, {var})")
    sequential_generator(mean, var)

    
    