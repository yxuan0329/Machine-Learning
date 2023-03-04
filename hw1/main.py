# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 21:59:37 2023

@author: xuan
"""
# import argparse
import numpy as np  
import scipy.linalg as la  
import matplotlib.pyplot as plt

class Matrix:
    def __init__(self, mat):
        self.row = len(mat)
        self.col = len(mat[0])
        self.matrix = mat
        
    def init_A(self, n):
        A = []
        for r in range(self.row):
            new_row = []
            for i in range (n):
                new_row.append(self.matrix[r][0] ** (n-i-1) )
            A.append(new_row)
        return A
    
    def init_b(self):
        b = []
        for r in range (self.row):
            new_row = []
            new_row.append(self.matrix[r][1])
            b.append(new_row)
        return b
    
    def scale(self, factor):
        for r in range(self.row):
            for c in range(self.col):
                self.matrix[r][c] *= factor
        return self
    
    def transpose(self):
        Mt = np.zeros((self.col, self.row))
        for r in range(self.row):
            for c in range(self.col):
                Mt[c][r] = self.matrix[r][c]
        return Mt
    
    def LUdecomposition(self):
        P, L, U = la.lu(self.matrix)
        # print(L, U)
        return L, U
        
    
    def inverse(self):
        L, U = self.LUdecomposition()
        row = len(L)
        col = len(L[0])
        # inverse = np.linalg.inv(self.matrix)
        # L_inverse = np.linalg.inv(L)
        # U_inverse = np.linalg.inv(U)
        
        I = identity_matrix(row)
        
        # compute L_inverse
        L_inverse = empty_matrix(row, col)
        for c in range(0, col):
            L_inverse[c][c] = 1.0 / L[c][c]
            
        for r in range(row):
            for c in range(r):
                tmp = 0
                for j in range(r): 
                    tmp += L[r][j] * L_inverse[j][c]
                L_inverse[r][c] = (I[r][c] - tmp)/ L[r][r]
                
                
        # compute U_inverse
        U_inverse = empty_matrix(row, col)
        
        for c in range(col):
            U_inverse[c][c] = 1.0 / U[c][c]
            
        for r in range(row-2, -1, -1):
            for c in range(col-1, r, -1):
                # print(r, c)
                tmp = 0
                for j in range(c, r, -1): 
                    tmp += U[r][j] * U_inverse[j][c]
                U_inverse[r][c] = (I[r][c] - tmp) / U[r][r]
                
        inverse = multiply(U_inverse, L_inverse)
        #print("my:")
        #print(U_inverse, inverse)
        return inverse
    """
    def multiply(self, mat2):
        new_mat = np.matmul(self.matrix(), mat2)
        if len(new_mat) == 1:
            new_mat = new_mat[0]
            # print(new_mat)
        return new_mat
    """   


def identity_matrix(n):
    identity_matrix = empty_matrix(n, n)
    for i in range (n):
        identity_matrix[i][i] = 1;
    # print(identity_matrix)
    return identity_matrix        
       

def empty_matrix(m, n):
    empty_matrix = [[0 for c in range(n)] for r in range(m)]
    return empty_matrix

def add(mat1, mat2):
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            mat1[i][j] += mat2.matrix[i][j]
    return mat1

def multiply(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        print("cannot multiply matrix!")
        return None
    """
    row = len(mat1)
    col = len(mat2[0])
    new_mat = empty_matrix(row, col)
    for r in range(row):
        for c in range(col):
            new_mat[r][c] = 
    """
    new_mat = np.matmul(mat1, mat2)
    if len(new_mat) == 1:
        new_mat = new_mat[0]
    # print(new_mat)
    return new_mat

def LSE(A, b, n, lamda):
    I = Matrix(identity_matrix(n))
    new_A = Matrix(add(multiply(A.transpose(), A.matrix), I.scale(lamda) ) )
    x = multiply (multiply(new_A.inverse(), A.transpose()), b.matrix)
    # print(x)    
    
    error = Matrix(multiply(A.matrix, x) - b.matrix)
    et = error.transpose()
    lse = multiply(et, multiply(A.matrix, x) - b.matrix)
    # print(lse)
    
    return x, lse


def Newton(A, b, n):
    def gradient(x):
        AtA = multiply(A.transpose(), A.matrix)
        gradient = 2 * multiply(AtA, x) - 2 * multiply(A.transpose(), b.matrix)  # 2 * AtA * x - 2 * At * b
        return gradient
        
    def hessian(A):
        hessian = Matrix(2 * multiply(A.transpose(), A.matrix))
        return hessian
    
    error = 0
    x = np.random.rand(n,1) # initial x, random generate
    hessian_inverse = hessian(A).inverse()
    for i in range (10):# while abs(x - x_old) < 0.0001:
        x = x - multiply(hessian_inverse, gradient(x)) 
        # print(i, x)
    error = Matrix(multiply(A.matrix, x) - b.matrix)
    et = error.transpose()
    error = multiply(et, multiply(A.matrix, x) - b.matrix)
    return x, error



# extract x-coord, y-coord from array
def xycoord(data):
    row = len(data)
    x = []
    y = []
    for r in range(row):
        x.append(data[r][0])
        y.append(data[r][1])
    return x, y

# draw plot
def plot(x, y, dim, x_LSE, x_Newton):
    minx = min(x)
    maxx = max(x)
    t = np.arange(minx-2, maxx+2)
    sum_LSE = 0
    sum_Newton = 0
    for i in range(dim):
        sum_LSE += x_LSE[i]*(t**(dim-i-1))
        sum_Newton += x_Newton[i]*(t**(dim-i-1)) # newton
        
    plt.figure(figsize=(8, 4), dpi=100)
    plt.scatter(x, y, color='red', edgecolor='black')
    plt.plot(t, sum_LSE, color='black')
    plt.title("LSE", fontdict={'size': 14})
    
    # newton's method plotting
    plt.figure(figsize=(8, 4), dpi=100)
    plt.scatter(x, y, color='red', edgecolor='black')
    plt.plot(t, sum_Newton, color='black')
    plt.title("Newton's Method", fontdict={'size': 14})
    
    
    plt.show()

def output_result(method, x, error):
    print(method)
    print("Fitting line:", end=' ')
    for i in range(len(x)):
        if(i != 0):
            if(x[i] >= 0):
                print("+", end=' ')
            else:
                print("-", end=' ')
                x[i] *= (-1)
        if(i != len(x)-1):
            print("%.12fx^%d"%(x[i], len(x)-i-1), end=' ')
        else:
            print("%.12f"%(x[i]))
    print("Total error: %.12f"%(error))

def readfile(file_name):
    data = []
    f = open(file_name, 'r')
    for line in f.readlines():
        str_point = line.split(',')
        point = []
        point.append(float(str_point[0]))
        point.append(float(str_point[1]))
        data.append(point)
    f.close()
    
    # print(data)
    return data


if __name__ == '__main__':   
    file_name = "testfile.txt"
    data = readfile(file_name)
    
    # given input
    # n = 3
    # lamda = 10000
    n = int(input("Input the number of polynomial bases (n):"))
    lamda = float(input("Input the lambda, for LSE case:"))
    
    mat = Matrix(data)
    A = mat.init_A(n)
    b = mat.init_b()
    
    # calculate A, b, x => x = (At *A)^-1 * At * b
    A = Matrix(A)
    b = Matrix(b)
    x = empty_matrix(n, 1)
    
    
    x_coord, y_coord = xycoord(data)
    x_lse, error_lse = LSE(A, b, n, lamda)
    x_nwtn, error_nwtn = Newton(A, b, n)
    
    output_result("LSE:", x_lse, error_lse)
    output_result("Newton's Method:", x_nwtn, error_nwtn)
    plot(x_coord, y_coord, n, x_lse, x_nwtn)

