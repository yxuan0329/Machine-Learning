# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 23:26:41 2023

@author: xuan
"""

import numpy as np
import math

def readfile(file_name):
    """ read the input file """
    data = []
    f = open(file_name, 'r')
    for line in f.readlines():
        data.append(line.strip())
    f.close()
    # print(data)
    return data

def count_frequency(data):
    zeros = data.count('0')
    ones = data.count('1')
    return ones, zeros

def calc_likelihood(p, N, m):
    """ binomial """
    likelihood = math.factorial(N) // (math.factorial(m)*math.factorial(N-m)) * pow(p, m) * pow(1-p, N-m)
    return likelihood

def output_result(i, data, likelihood, prior_a, prior_b, posterior_a, posterior_b):
    print("case", i+1, ":", data)
    print("Likelihood:", likelihood)
    print("Beta prior:\t    a =", prior_a, "b =", prior_b)
    print("Beta posterior:\ta =", posterior_a, "b =", posterior_b, "\n")
    
def beta_binomial_conjugation(a, b, dataset):
    prior_a = a
    prior_b = b
    for i, data in enumerate(dataset):
        posterior_a, posterior_b = count_frequency(data)
        prob = float((posterior_a) / (posterior_a + posterior_b))
        likelihood = calc_likelihood(prob, posterior_a + posterior_b, posterior_a)
        posterior_a += prior_a
        posterior_b += prior_b
        output_result(i, data, likelihood, prior_a, prior_b, posterior_a, posterior_b)
        prior_a = posterior_a
        prior_b = posterior_b
        

if __name__ == '__main__':
    file_name = str(input("Please enter the input file name (ex. testfile.txt): "))
    # file_name = './dataset/testfile.txt'
    data = readfile("./dataset/"+file_name)
    
    # input a, b
    a = int(input("Please input prior a: "))
    b = int(input("Please input prior b: "))
    
    beta_binomial_conjugation(a, b, data)