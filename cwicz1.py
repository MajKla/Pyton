# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:25:24 2021

@author: kmajk
"""

import numpy as np
#zad.4.1.1
# arr = np.array([1,2,3,4,5])
# print(arr)

#A= np.array([[1,2,3],[7,8,9]])
#print(A)
# A=np.array([[1,2,3],[7,8,9]])
# print(A)
#A= np.array([[1,2, \
 #              3],
 #             [7,8,9]])
# print(A)

#zad.4.1.2
# v=np.arange(1,7)
# print(v,"\n")#[1 2 3 4 5 6] 
# v=np.arange(-2,7)
# print(v,"\n")#[-2 -1  0  1  2  3  4  5  6] 
# v=np.arange(1,10,3)
# print(v,"\n")#[1 4 7] 
# v=np.arange(1,10.1,3)
# print(v,"\n")#[ 1.  4.  7. 10.] 
# v=np.arange(1,11,3)
# print(v,"\n")#[ 1  4  7 10] 
# v=np.arange(1,2,0.1)
# print(v,"\n")#[1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9]

# v=np.linspace(1,3,4)
# print(v)#[1.         1.66666667 2.33333333 3.        ]-dzieli odcinek od 1-3 na 4 czesci
# v=np.linspace(1,10,4)
# print(v)#[ 1.  4.  7. 10.]

# X= np.ones((2,3))#[[1. 1. 1.]
# #  #                [1. 1. 1.]] 
# # Y= np.zeros((2,3,4))#[[[0. 0. 0. 0.]
# # #                      [0. 0. 0. 0.]
# # #                      [0. 0. 0. 0.]]

# # #                     [[0. 0. 0. 0.]
# # #                      [0. 0. 0. 0.]
# # #                      [0. 0. 0. 0.]]] 
# Z=np.eye(2,2)#[[1. 0.]
# # #             [0. 1.]] 
# Q= np.random.rand(2,5)# [[0.2703581  0.4252598  0.13326085 0.9938472  0.97452604]
# # #                        [0.51242445 0.07090245 0.85388187 0.96494366 0.11106337]]

# print (X,"\n\n",Y,"\n\n",Z,"\n\n",Q)

#zad4.1.3
# U= np.block([[Q],[X,Z]])
# print (U)#[[0.96072395 0.25582989 0.62182342 0.70819855 0.97194474]
 #[0.48114079 0.52996773 0.35675755 0.76105575 0.70547601]
# [1.         1.         1.         1.         0.        ]
 #[1.         1.         1.         0.         1.        ]]
 
 #zad.4.1.4
V= np.block([[
np.block([
np.block([[np.linspace(1,3,3)],
    [np.zeros((2,3))]]),
np.ones((3,1))])
],
[np.array([100,3,1/2,0.333])]])
print(V)#[[  1.      2.      3.      1.   ]
  #        [  0.      0.      0.      1.   ]
  #        [  0.      0.      0.      1.   ]
  #        [100.      3.      0.5     0.333]]

#4.2
print( V[0,2] )
print( V[3,0] )
print( V[3,3] )
print( V[-1,-1] )
print( V[-4,-3] )
print( V[3,:] )
print( V[:,2] )
print( V[3,0:3] )
print( V[np.ix_([0,2,3],[0,-1])] )
print( V[3] )
#[[  1.      2.      3.      1.   ]
# [  0.      0.      0.      1.   ]
# [  0.      0.      0.      1.   ]
# [100.      3.      0.5     0.333]]
#3.0
#100.0
#0.333
#0.333
#2.0
#[100.      3.      0.5     0.333]
#[3.  0.  0.  0.5]
#[100.    3.    0.5]
#[[  1.      1.   ]
# [  0.      1.   ]
# [100.      0.333]]
#[100.      3.      0.5     0.333]