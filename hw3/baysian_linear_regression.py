# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 12:33:22 2023

@author: xuan
"""
import sys
import numpy as np    
import matplotlib.pyplot as plt

def univariate_gaussian(m, v):
    """ By Central Limit Theorem, 
        generate 12 number uniform between [0,1], add them up and substract 6
    """
    x = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    return x * (v ** 0.5) + m

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

def polynomial(x):
    poly_X = np.power(x, range(n)).reshape(1,n)
    return poly_X 

def baysian_linear_regression(b, n, var, a, w):
    cnt = 0
    prior_mean = np.zeros((1, n))
    prior_cov = np.array([sys.maxsize, n])
    x_set = []
    y_set = []
    while True:
        new_x, new_y = polynomial_model(n, var, w)
        x_set.append(new_x)
        y_set.append(new_y)
        
        X = polynomial(new_x)
        XtX = X.T.dot(X)
        if cnt == 0:
            post_cov = a * XtX + b * np.eye(n) # cov = a * XtX + bI
            post_mean = a * np.linalg.inv(post_cov).dot(X.T) * new_y # mean = a * cov^-1 * Xt * y
            plot_groundtruth(x_set, y_set, post_mean, post_cov)
        else:
            post_cov = a * XtX + prior_cov
            post_mean = np.linalg.inv(post_cov).dot(a * X.T.dot(new_y) + prior_cov.dot(prior_mean))
        cnt += 1
        post_cov_inverse = np.linalg.inv(post_cov)
        
        if cnt == 10:
            draw_plot(223, x_set, y_set, post_mean, post_cov_inverse, cnt, "After 10 incomes")
        elif cnt == 50:
            draw_plot(224, x_set, y_set, post_mean, post_cov_inverse, cnt, "After 50 incomes")
        
        marginalize_mean = np.dot(X, post_mean)
        marginalize_cov = ((1/a) + X.dot(post_cov_inverse.dot(X.T) ))
        show(new_x, new_y, post_mean, post_cov, marginalize_mean[0,0], marginalize_cov[0,0], cnt)
        
        # check break
        if np.linalg.norm(prior_mean-post_mean, ord=2) < threshold:
            draw_plot(222, x_set, y_set, post_mean, post_cov_inverse, cnt, "Predict Result")
            break
        
        # update priors
        prior_cov = post_cov
        prior_mean = post_mean
    plt.show()

def calculate_var(x, cov):
    var = np.zeros(100)
    for i in range(100):
        X = polynomial(x[i])
        var[i] = 1/a + X.dot(cov).dot(X.T)
    return var
def show(new_x, new_y, post_mean, post_cov, marginalize_mean, marginalize_cov, count):
    print(f"Add data point({new_x:.5f}, {new_y:.5f})")

    print("Posterior mean:")
    for i in range(n):
        print(f"  {post_mean[i,0]:0.10f}")

    print("Posterior Variance:")
    for r in range(n):
        print("  ", end=" ")
        for c in range(n):
            print(f"{post_cov[r,c]:0.10f}", end="")
            if c != n-1:
                print(", ", end="")
        print(" ")
	
    print(f"{count}th Predictive distribution ~ N({marginalize_mean:.5f}, {marginalize_cov:.5f})")
    print("\n")
    
def plot_groundtruth(x_set, y_set, mean, cov):
    x = np.linspace(-2.0, 2.0, 100)
    func = np.poly1d(np.flip(w))
    y = func(x)
    
    plt.subplot(221)
    plt.title("Ground truth")
    plt.plot(x, y, color = 'black')
    plt.plot(x, y+var, color = 'red')
    plt.plot(x, y-var, color = 'red')
    plt.xlim(-2.0, 2.0)
    plt.ylim([min(y_set)-10, max(y_set)+10])
    
def draw_plot(index, x_set, y_set, mean, cov, cnt, title):
    x = np.linspace(-2.0, 2.0, 100)  
    func = np.poly1d(np.flip(np.reshape(mean, n)))
    y = func(x)
    var = calculate_var(x, cov)
    
    plt.subplot(index)    
    plt.title(title)
    plt.tight_layout()
    plt.scatter(x_set[0:cnt], y_set[0:cnt], s=7.0)
    plt.plot(x, y, color = 'black')
    plt.plot(x, y+var, color = 'red')
    plt.plot(x, y-var, color = 'red')
    plt.xlim(-2.0, 2.0)
    plt.ylim([min(y_set)-10, max(y_set)+10])
    
if __name__ == '__main__': 
    
    precision = float(input("Precision: "))
    n = int(input("n: "))
    var = float(input("variance: "))
    w = input("Weights: ")
    w = np.array([float(x) for x in w.split()])
    """
    precision = 1
    n = 4
    var = 1
    w = np.array([1.0, 2.0, 3.0, 4.0])
    """
    threshold = 1e-4
    a = 1/var
    baysian_linear_regression(precision, n, var, a, w)
    