# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 21:59:37 2023

@author: xuan
"""
# import argparse
import numpy as np  
import scipy.linalg as la  

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
        inverse = np.linalg.inv(self.matrix)
        # print(inverse)
        return inverse
        


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
        print("cannot multiply matrix")
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

if __name__ == '__main__':
    """
    dataset = [-5.0,51.76405234596766,
-4.795918367346939,45.42306433039972,
-4.591836734693878,41.274448104888755,
-3.979591836734694,26.636216497466364,
-3.571428571428571,20.256806057008426,
-2.9591836734693877,11.618429243797276,
-2.7551020408163263,10.450525068812203,
-1.7346938775510203,1.8480982318414874,
-1.3265306122448979,-1.0405349639051173,
-0.9183673469387754,-4.614630798757861,
-0.7142857142857144,-1.3871977310902517,
-0.3061224489795915,-1.9916444039966117,
0.1020408163265305,-0.912924608376358,
0.7142857142857144,6.63482003068499,
1.1224489795918373,9.546867459016372,
1.7346938775510203,15.72016146597016,
1.9387755102040813,20.62251683859554,
2.5510204081632653,33.48059725819715,
2.959183673469388,40.76391965675495,
3.979591836734695,66.8997605629381,
4.387755102040817,78.44316465660981,
4.591836734693878,86.99156782355371,
5.0,99.78725971978604]
    x = []
    y = []
    n = len(dataset)
    i = 0
    while i < n:
        x.append(dataset[i])
        y.append(dataset[i+1])
        i += 2
    # print(x, y)
    data = [[-5.0,51.76405234596766],
[-4.795918367346939,45.42306433039972],
[-4.591836734693878,41.274448104888755],
[-3.979591836734694,26.636216497466364],
[-3.571428571428571,20.256806057008426],
[-2.9591836734693877,11.618429243797276],
[-2.7551020408163263,10.450525068812203],
[-1.7346938775510203,1.8480982318414874],
[-1.3265306122448979,-1.0405349639051173],
[-0.9183673469387754,-4.614630798757861],
[-0.7142857142857144,-1.3871977310902517],
[-0.3061224489795915,-1.9916444039966117],
[0.1020408163265305,-0.912924608376358],
[0.7142857142857144,6.63482003068499],
[1.1224489795918373,9.546867459016372],
[1.7346938775510203,15.72016146597016],
[1.9387755102040813,20.62251683859554],
[2.5510204081632653,33.48059725819715],
[2.959183673469388,40.76391965675495],
[3.979591836734695,66.8997605629381],
[4.387755102040817,78.44316465660981],
[4.591836734693878,86.99156782355371],
[5.0,99.78725971978604]]
    """
    # given input
    data = [[3,2], [4,5], [9,8]]
    n = 3
    lamda = 1
    
    mat = Matrix(data)
    A = mat.init_A(n)
    b = mat.init_b()
    
    A = Matrix(A)
    b = Matrix(b)
    x = empty_matrix(n, 1)
    
    I = Matrix(identity_matrix(n))
    new_A = Matrix(add(multiply(A.transpose(), A.matrix), I.scale(lamda) ) )
    x = multiply (multiply(new_A.inverse(), A.transpose()), b.matrix)
    print(x)
