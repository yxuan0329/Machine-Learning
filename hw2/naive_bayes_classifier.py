# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:42:32 2023

@author: xuan
"""

import math
import numpy as np

class Data(object):
    def __init__(self):
        self.classes = 10 # 0-9, 10 numbers
        self.magic_num = -1
        self.num = 60000 # total numbers in image file
        self.row = 28 # rows in each digit
        self.col = 28 # cols in each digit
        self.pixel = int(self.row * self.col)
        self.images = []
        
        self.label_magicnum = -1
        self.label_num = 60000
        self.label = []


def read_dataset(img_file_path, label_file_path):
    """ read the dataset from file path, 
        save the information into the Data() class.
        
        input: image file path, label file path
        output: d = Data()
    """
    d = Data()
    # read image file
    img_file = open(img_file_path, "rb")
    d.magic_num =  int.from_bytes(img_file.read(4), 'big')
    d.num = int.from_bytes(img_file.read(4), 'big')
    d.row = int.from_bytes(img_file.read(4), 'big')
    d.col = int.from_bytes(img_file.read(4), 'big')
    
    d.images = np.zeros((d.num, d.pixel))
    for i in range(d.num):
        for j in range(d.pixel):
            d.images[i][j] = int.from_bytes(img_file.read(1), 'big')
    img_file.close()
    
    # read label file
    label_file = open(label_file_path, "rb")
    d.label_magicnum = int.from_bytes(label_file.read(4), 'big')
    d.label_num = int.from_bytes(label_file.read(4), 'big')
    d.label = np.zeros(d.label_num)
    for i in range(d.label_num):
        d.label[i] = int.from_bytes(label_file.read(1), 'big')
    label_file.close()
    
    return d

def calculate_prior(tr_data, classes):
    """ calculate the prior of the data
    
        input: data
        output: prior
    """
    prior = np.zeros(classes, dtype=float)
    for n in range(tr_data.num):
        number = int (tr_data.label[n])
        # print(tr_data.label[n])
        prior[number] += 1
    prior = prior / tr_data.num # probability
    # print(prior)
    return prior
        

def discrete_likelihood(tr_data, classes, bins=32):
    """ calculate the likelihood of training data 
    
        input: training data
        output: the likelihood of the data
    """
    def calculate_prob():
        likelihood_sum = np.sum(likelihood, axis=2)
        for i in range (len(likelihood)):
            for j in range (len(likelihood[0])):
                likelihood[i,j,:] = likelihood[i,j,:] / likelihood_sum[i,j]
        return likelihood
            
    def pseudo_count(pixel, bins, likelihood):
        """ set likelihood as 0.0001 if it has 0 appearance
        
            input: likelihood
            output: (updated) likelihood
        """
        for n in range (classes): # 0-9
            for p in range(pixel):
                 for b in range(bins):
                    if likelihood[n][p][b] == 0:
                        likelihood[n][p][b] = 0.0001 # or set as min in likelihood
        return likelihood
    
    likelihood = np.zeros((classes, tr_data.pixel, bins), dtype=float)
    for n in range(tr_data.num):
        for p in range(tr_data.pixel):
            b = int(tr_data.images[n][p] // 8) # cut the darkness into 8 bins
            l = int(tr_data.label[n]) # the number l in label[n]
            likelihood[l][p][b] += 1  # +1 to the darkness of the pixel p in number l, n-th training data
    prob = calculate_prob()
    likelihood = pseudo_count(tr_data.pixel, bins, prob)

    return likelihood

def discrete_classifier(f, tr_data, te_data, prior, likelihood, classes):
    """ posterior = prior * likelihood / marginal 
        
        input: training data, testing data, prior, likelihood
        output: prediction error
    """
    prior_likelihood = np.zeros(classes, dtype=float) # prior * likelihood
    
    error = 0
    pixel = te_data.row * te_data.col
    for i in range (te_data.num):
        posterior = [] # store the posterior 0-9 of this number
        for c in range(classes):
            prior_likelihood[c] += np.log(prior[c]) # *= prior
            for p in range(pixel):
                b = int(te_data.images[i][p]//8)
                prior_likelihood[c] += np.log(likelihood[c][p][b]) # *= likelihood
         
        prior_likelihood /= np.sum(prior_likelihood) # divide by marginal
        posterior.append(prior_likelihood)
        prediction = np.argmin(posterior) # my prediction for this number
        show_result(f, te_data, posterior, prediction)
        print("Prediction:", prediction,", Ans:", int(te_data.label[i]), "\n")
        f.write("Prediction: {}, Ans: {}\n\n".format(prediction, str(int(te_data.label[i]))) )
        if not prediction == int(te_data.label[i]): # prediction error
            error += 1
    return error
            

### Continuous 
def continuous_likelihood(tr_data, classes, prior):
    """ calculate the mean and variance of the Gaussian distribution in the dataset
    
        input: training data, prior
        output: mean, variance
    """
    def mean():
        mean = np.zeros((classes, tr_data.pixel), dtype=float)
        for n in range(tr_data.num):
            for p in range(tr_data.pixel):
                c = int(tr_data.label[n]) # the exact number c that train_data[n] represented
                mean[c][p] += tr_data.images[n][p] # add the pixel value to the mapping pixel in the mean of c (mean[c][p])
        for c in range(classes):
            for p in range(tr_data.pixel):
                mean[c][p] = np.divide(mean[c][p], prior[c]*tr_data.num)
        return mean
    
    def variance():
        var = np.zeros((classes, tr_data.pixel), dtype=float)
        for n in range(tr_data.num):
            for p in range(tr_data.pixel):
                c = int(tr_data.label[n])
                var[c][p] += (tr_data.images[n][p] - mean[c][p]) ** 2
        for c in range(classes):
            for p in range(tr_data.pixel):
                var[c][p] = np.divide(var[c][p], prior[c]*tr_data.num)
        return var
    
    mean = mean()
    var = variance()
    return mean, var

def continuous_classifier(f, te_data, prior, classes, mean, var):
    """ calculate posterior according to prior, likelihood, and Gaussian distribution
    
        input: testing data, prior, mean, variance
        output: error
    """
    error = 0
    for n in range (te_data.num):
        posteriors = [] # store the posterior 0-9 of this number
        prior_likelihood = np.zeros(classes, dtype=float)
        prior_likelihood = np.log(prior) # posterior = prior * prior * ... 
        for c in range(classes):
            for p in range(te_data.pixel):
                if (var[c][p] == 0): # prevent divide-by-zero error
                    var[c][p] = 1e3
                prior_likelihood[c] -= np.log(np.sqrt(2.0 * math.pi * var[c][p]))  # posterior *= 1/(2* pi * var)
                prior_likelihood[c] += (-(te_data.images[n][p] - mean[c][p]) ** 2 / (2.0 * var[c][p]) ) # posterior /= -(x - mean)^2 / (2 * var)
        # posterior = np.divide(posterior, sum(posterior)) # divide by marginal
        prior_likelihood /= np.sum(prior_likelihood)
        posteriors.append(prior_likelihood)
        prediction = np.argmin(posteriors) # my prediction of this number 
        show_result(f, te_data, posteriors, prediction)
        print("Prediction:", prediction,", Ans:", int(te_data.label[n]), "\n")
        f.write("Prediction: {}, Ans: {}\n\n".format(prediction, str(int(te_data.label[n]))) )
        if (prediction != te_data.label[n]): # prediction error
            error += 1
    return error

def show_discrete_imagination(f, d, likelihood):
    """ visualize the number by pixel """
    zero = np.sum(likelihood[:,:,0:16], axis=2)
    one = np.sum(likelihood[:,:,16:32], axis=2)
    
    img = np.zeros((d.classes, d.pixel))
    img = (one >= zero)*1

    print("Imagination of numbers in Bayesian classifier:\n")
    f.write("Imagination of numbers in Bayesian classifier:\n")
    for c in range(d.classes):
        print("{}:".format(c))
        f.write("{}:\n".format(c))
        for row in range (28):
            for col in range(28):
                print(img[c][row*28+col], end=' ')
                f.write("{} ".format(img[c][row*28+col]))
            print(" ")
            f.write("\n")

def show_continuous_imagination(f, d, mean):
    """ visualize the number by pixel """
    img = np.zeros((d.classes, d.pixel))
    img = (mean>=128) * 1
    
    print("Imagination of numbers in Bayesian classifier:\n")
    f.write("Imagination of numbers in Bayesian classifier:\n")
    for c in range(d.classes):
        print("{}:".format(c))
        f.write("{}:\n".format(c))
        for row in range (28):
            for col in range(28):
                print(img[c][row*28+col], end=' ')
                f.write("{} ".format(img[c][row*28+col]))
            print(" ")
            f.write("\n")

def show_result(f, d, prob, prediction):
    print("Posterior (in log scale):")
    f.write("Posterior (in log scale):\n")
    for i in range(d.classes):
        print("{}: {}".format(i, prob[0][i]))
        f.write("{}: {}\n".format(i, prob[0][i]))
    
    
if __name__ == '__main__': 
    classes = 10 # 0-9
    print("Read dataset...")
    tr_data = read_dataset("./dataset/train-images.idx3-ubyte", "./dataset/train-labels.idx1-ubyte")
    te_data = read_dataset("./dataset/t10k-images.idx3-ubyte", "./dataset/t10k-labels.idx1-ubyte")
    mode = int(input("Mode: (0:Discrete, 1:Continuous) "))
    
    prior = calculate_prior(tr_data, classes=10)
    if mode == 0:
        f = open("HW2-output-discrete.txt", "w")
        print("Calculating likelihood...")
        likelihood = discrete_likelihood(tr_data, classes=10, bins=32) # bins = 32
        error = discrete_classifier(f, tr_data, te_data, prior, likelihood, classes)
        show_discrete_imagination(f, te_data, likelihood)
        print("\nError rate:", float(error)/te_data.num)
        f.write("\nError rate: {}".format(float(error/te_data.num)))
    
    elif mode == 1:
        f = open("HW2-output-continuous.txt", "w")
        print("Calculating mean and variance...")
        mean, var = continuous_likelihood(tr_data, classes, prior)
        error = continuous_classifier(f, te_data, prior, classes, mean, var)
        show_continuous_imagination(f, te_data, mean)
        print("\nError rate:", float(error)/te_data.num)
        f.write("\nError rate: {}".format(float(error/te_data.num)))
    f.close()