# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 21:59:37 2023

@author: xuan
"""
import numpy as np    
import matplotlib.pyplot as plt

class Matrix(object):
    """ class with single matrix operation """
    def __init__(self, mat):
        self.row = len(mat)
        self.col = len(mat[0])
        self.matrix = mat
    
    def scale(self, factor):
        for r in range(self.row):
            for c in range(self.col):
                self.matrix[r][c] *= factor
        return self
    
    def transpose(self):
        Mt = empty_matrix(self.col, self.row)
        for r in range(self.row):
            for c in range(self.col):
                Mt[c][r] = self.matrix[r][c]
        return Mt
    
    def LUdecomposition(self):
        """ do LU decomposition, return L, U"""
        # P, L, U = la.lu(self.matrix)
        if not self.row == self.col:
            print("The matrix is not a square matrix!")
            return None
        dim = self.row
        L = identity_matrix(dim)    
        U = empty_matrix(dim, dim)
        for r in range(dim):
            # compute U
            for c in range(r, dim):
                tmp = 0
                for i in range(r):
                    tmp += L[r][i] * U[i][c]
                U[r][c] = self.matrix[r][c] - tmp
                
            # compute L
            for c in range(r, dim):
                if (r == c):
                    L[r][c] = 1 # diagonal
                else:
                    tmp = 0
                    for i in range(r):
                        tmp += L[c][i] * U[i][r]
                    L[c][r] = (self.matrix[c][r] - tmp) / U[r][r]
        return L, U
        
    
    def inverse(self):
        """ return its inverse matrix """
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
        return inverse   

def design_matrix(data, n):
    """ returns design matrix A, for calculating Ax = b """ 
    A = []
    for r in range(len(data)):
        new_row = []
        for i in range (n):
            new_row.append(data[r][0] ** (n-i-1) )
        A.append(new_row)
    return Matrix(A)

def design_b(data):
    """ returns matrix b, for calculating Ax = b """ 
    b = []
    for r in range (len(data)):
        new_row = []
        new_row.append(data[r][1])
        b.append(new_row)
    return Matrix(b)

def identity_matrix(n):
    """ returns a n*n identity matrix I """ 
    identity_matrix = empty_matrix(n, n)
    for i in range (n):
        identity_matrix[i][i] = 1;
    return identity_matrix        
       

def empty_matrix(m, n):
    """ returns a m*n empty matrix """ 
    empty_matrix = [[0.0 for c in range(n)] for r in range(m)]
    return empty_matrix

def add(mat1, mat2):
    """ returns the sum of mat1 and mat2 """
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            mat1[i][j] += mat2.matrix[i][j]
    return mat1

def multiply(mat1, mat2):
    """ returns the multiplication of mat1 and mat2 """
    if len(mat1[0]) != len(mat2):
        print("cannot multiply matrix!")
        return None
    
    row = len(mat1)
    col = len(mat2[0])
    new_mat = empty_matrix(row, col)
    for c in range(col):
        for r in range(row):
            for i in range(len(mat1[0])):
                new_mat[r][c] += mat1[r][i] * mat2[i][c] 
    #new_mat = np.matmul(mat1, mat2)
    if len(new_mat) == 1:
        new_mat = new_mat[0]
    return np.array(new_mat)

def square(mat):
    """ returns the square of the matrix, Mt * M"""
    mat_t = Matrix(mat).transpose()
    square = multiply(mat_t, mat)
    return square

def LSE(A, b, n, lamda):
    """ do LSE to find the regression line """
    I = Matrix(identity_matrix(n))
    AtA_I = Matrix(add(multiply(A.transpose(), A.matrix), I.scale(lamda) ) )
    x = multiply (multiply(AtA_I.inverse(), A.transpose()), b.matrix)   
    lse = square(multiply(A.matrix, x) - b.matrix)
    return x, lse


def Newton(A, b, n):
    """ do Newton's Method to find the regression line """
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
    while True:
        delta = multiply(hessian_inverse, gradient(x))
        x = x - delta
        # print(x)
        if (square(delta) < 0.0001):
            break
    error = square(multiply(A.matrix, x) - b.matrix)
    return x, error




def xycoord(data):
    """ extract x-coord, y-coord from array """
    row = len(data)
    x = []
    y = []
    for r in range(row):
        x.append(data[r][0])
        y.append(data[r][1])
    return x, y

def plot(x, y, dim, x_LSE, x_Newton):
    """ draw the plot for LSE and Newton's Method"""
    minx = min(x)
    maxx = max(x)
    t = np.arange(minx-2, maxx+2)
    sum_LSE = 0
    sum_Newton = 0
    for i in range(dim):
        sum_LSE += x_LSE[i]*(t**(dim-i-1))
        sum_Newton += x_Newton[i]*(t**(dim-i-1))
        
    # lse plotting    
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
    """ show the output result of the fitting line and error """
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
            print("%.12fX^%d"%(x[i], len(x)-i-1), end=' ')
        else:
            print("%.12f"%(x[i]))
    print("Total error: %.12f"%(error))

def readfile(file_name):
    """ read the input file """
    data = []
    f = open(file_name, 'r')
    for line in f.readlines():
        str_point = line.split(',')
        point = []
        point.append(float(str_point[0]))
        point.append(float(str_point[1]))
        data.append(point)
    f.close()
    return data


if __name__ == '__main__':   
    file_name = str(input("Please enter the input file name (ex. testfile.txt): "))
    # file_name = 'testfile.txt'
    data = readfile(file_name)
    
    # n = 3, lamda = 10000
    n = int(input("Input the number of polynomial bases (n):"))
    lamda = float(input("Input the lambda, for LSE case:"))
    
    # calculate A, b, x => x = (At *A)^-1 * At * b
    A = design_matrix(data, n)
    b = design_b(data)
    x = empty_matrix(n, 1)
    
    x_lse, error_lse = LSE(A, b, n, lamda)
    output_result("LSE:", x_lse, error_lse)
    
    x_nwtn, error_nwtn = Newton(A, b, n)
    output_result("Newton's Method:", x_nwtn, error_nwtn)
    
    x_coord, y_coord = xycoord(data)
    plot(x_coord, y_coord, n, x_lse, x_nwtn)