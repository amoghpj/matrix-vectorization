'''
Author: Amogh Jalihal
Date: 2018-10-08
Description: Python implementation of the Magnus-Neudecker Eliminator-Duplicator matrix vectorization method for symmetric matrices
Note: This is a reimplementation of the R package matrixcalc which has utilities for the same. Matrixcalc is authored by Frederick Novomestky, and the package lives at 
https://cran.r-project.org/web/packages/matrixcalc/index.html

Usage:
To generate the eliminator matrix L for an nxn symmetric matrix, call L(n). Likewise for the duplicator matrix D.

'''

import numpy as np
import sys

def L(n):
    '''
    Arguments:
    n: integer >=2
    Returns:
    Numpy array of dimensions n(n+1)/2 x n**2
    '''
    if type(n) != int:
        print('Incorrect type')
        sys.exit()
    if n <=2 :
        print('n is less than 2')

    k, I = u_vectors(n)

    E =  E_matrices(n)
    p = n*(n+1)/2
    nsquare = n**2
    
    L = np.zeros((p,nsquare))
    for j in range(0,n):
        for i in range(j,n):
            L = L + np.matmul(I[np.ix_(np.arange(0,len(I)),[int(k[i][j])])],
                              E[i][j].reshape((-1,1),order='F').transpose())

    rerturn(L)


def D(n):
    '''
    Arguments:
    n: integer >=2
    Returns:
    Numpy array of dimensions n**2 x n(n+1)/2
    '''
    if type(n) != int:
        print('Incorrect type')
        sys.exit()
    if n <=2 :
        print('n is less than 2')
    
    p = n*(n+1)/2
    nsquare = n**2
    Dt = np.zeros((p,nsquare))
    k, I = u_vectors(n)
    T = T_matrices(n)
    for j in range(0,n):
        for i in range(j,n):
            Dt = Dt + np.matmul(I[np.ix_(np.arange(0,len(I)),[int(k[i][j])])],
                              T[i][j].reshape((-1,1),order='F').transpose())
    D = Dt.transpose()
    return D

def T_matrices(n):
    E = E_matrices(n)
    T = list()
    for i in range(0,n):
        T.append(list())
        for j in range(0,n):
            if i==j:
                T[-1].append(E[i][j])
            else:
                T[-1].append(E[i][j] + E[j][i])
    return T
                
def u_vectors(n):
    p = n*(n+1)/2
    I = np.eye(p)
    k = np.zeros((n,n))
    
    for j in range(1,n+1):
        for i in range(j,n+1):
            k[i-1][j-1] = int((j-1)*n + i -0.5*(j)*(j-1)) -1
    return(k, I)


def E_matrices(n):
    I = np.eye(n)
    #print(I)
    E = list()
    for i in range(0,n):
        E.append(list())
        for j in range(0,n):
            E[-1].append(np.outer(I[i],I[j]))
    return E

