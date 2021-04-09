import numpy as np

# zad.4.1.1
# arr = np.array([1,2,3,4,5])
# print(arr)

# A= np.array([[1,2,3],[7,8,9]])
# print(A)
# A=np.array([[1,2,3],[7,8,9]])
# print(A)
# A= np.array([[1,2, \
#              3],
#             [7,8,9]])
# print(A)

# zad.4.1.2
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

# zad4.1.3
# U= np.block([[Q],[X,Z]])
# print (U)#[[0.96072395 0.25582989 0.62182342 0.70819855 0.97194474]
# [0.48114079 0.52996773 0.35675755 0.76105575 0.70547601]
# [1.         1.         1.         1.         0.        ]
# [1.         1.         1.         0.         1.        ]]

# zad.4.1.4
V = np.block([[
    np.block([
        np.block([[np.linspace(1, 3, 3)],
                  [np.zeros((2, 3))]]),
        np.ones((3, 1))])
],
    [np.array([100, 3, 1 / 2, 0.333])]])
# print(V)  # [[  1.      2.      3.      1.   ]
# #        [  0.      0.      0.      1.   ]
# #        [  0.      0.      0.      1.   ]
# #        [100.      3.      0.5     0.333]]
#
# # 4.2
# print(V[0, 2])
# print(V[3, 0])
# print(V[3, 3])
# print(V[-1, -1])
# print(V[-4, -3])
# print(V[3, :])
# print(V[:, 2])
# print(V[3, 0:3])
# print(V[np.ix_([0, 2, 3], [0, -1])])
# print(V[3])
# # [[  1.      2.      3.      1.   ]
# # [  0.      0.      0.      1.   ]
# # [  0.      0.      0.      1.   ]
# # [100.      3.      0.5     0.333]]
# # 3.0
# # 100.0
# # 0.333
# # 0.333
# # 2.0
# # [100.      3.      0.5     0.333]
# # [3.  0.  0.  0.5]
# # [100.    3.    0.5]
# # [[  1.      1.   ]
# # [  0.      1.   ]
# # [100.      0.333]]
# # [100.      3.      0.5     0.333]
# Q = np.delete(V, 2, 0)
# print(Q)
#[[  1.      2.      3.      1.   ]
 # [  0.      0.      0.      1.   ]
 # [100.      3.      0.5     0.333]]
# Q = np.delete(V, 2, 1)
# print(Q)
# # [[  1.      2.      1.   ]
# #  [  0.      0.      1.   ]
# #  [  0.      0.      1.   ]
# #  [100.      3.      0.333]]
v = np.arange(1,7)
# # print( np.delete(v, 3, 0) )
# # # [1 2 3 5 6]
# np.size(v)
# np.shape(v)
# np.size(V)
# np.shape(V)
#4.5.1. Operacje na macierzach
A = np.array([[1, 0, 0],
[2, 3, -1],
[0, 7, 2]] )
B = np.array([[1, 2, 3],
[-1, 5, 2],
[2, 2, 2]] )
# print( A+B )
# print( A-B )
# print( A+2 )
# print( 2*A )
# wynik
# [[2 2 3]
#  [1 8 1]
#  [2 9 4]]
# [[ 0 -2 -3]
#  [ 3 -2 -3]
#  [-2  5  0]]
# [[3 2 2]
#  [4 5 1]
#  [2 9 4]]
# [[ 2  0  0]
#  [ 4  6 -2]
#  [ 0 14  4]]
#4.5.2 Mnożenie macierzowe
MM1 = A@B
# print(MM1)
# MM2 = B@A
# print(MM2)
#wynik
# [[ 1  2  3]
#  [-3 17 10]
#  [-3 39 18]]
# [[ 5 27  4]
#  [ 9 29 -1]
#  [ 6 20  2]]
#Mnożenie tablicowe
# MT1 = A*B
# print(MT1)
# MT2 = B*A
# print(MT2)
# [[ 1  0  0]
#  [-2 15 -2]
#  [ 0 14  4]]
# [[ 1  0  0]
#  [-2 15 -2]
#  [ 0 14  4]]
#4.5.4 Dzielenie tablicowe
# DT1 = A/B
# print(DT1)
# [[ 1.   0.   0. ]
#  [-2.   0.6 -0.5]
#  [ 0.   3.5  1. ]]
# C = np.linalg.solve(A,MM1)
# print(C) # porownaj z macierza B
# x = np.ones((3,1))
# b = A@x
# y = np.linalg.solve(A,b)
# print(y)
# #
# [[ 1.  2.  3.]
#  [-1.  5.  2.]
#  [ 2.  2.  2.]]
# [[1.]
#  [1.]
#  [1.]]
#zad.4.5.6 Potęgowanie
# PM = np.linalg.matrix_power(A,2) # por. A@A
# PT = A**2 # por. A*A
#4.5.7 Transpozycja

# A.T # transpozycja
# A.transpose()
# A.conj().T # hermitowskie sprzezenie macierzy (dla m. zespolonych)
# A.conj().transpose()
#Operacje porównań i funkcje logiczne
# A == B
# A != B
# 2 < A
# A > B
# A < B
# A >= B
# A <= B
# np.logical_not(A)
# np.logical_and(A, B)
# np.logical_or(A, B)
# np.logical_xor(A, B)
# print( np.all(A) )
# print( np.any(A) )
# print( v > 4 )
# print( np.logical_or(v>4, v<2))
# print( np.nonzero(v>4) )
# print( v[np.nonzero(v>4) ] )

# False
# True
# [False False False False  True  True]
# [ True False False False  True  True]
# (array([4, 5], dtype=int64),)
# [5 6]
#Inne
 # print(np.max(A))
# print(np.min(A))
# print(np.max(A,0))
# print(np.max(A,1))
# print( A.flatten() )
# print( A.flatten('F') )

#Matplotlib
#wykresy funkcji
import matplotlib.pyplot as plt
# x = [1,2,3]
# y = [4,6,5]
# plt.plot(x,y)
# plt.show()
# x = np.arange(0.0, 2.0, 0.01)
# y = np.sin(2.0*np.pi*x)
# plt.plot(x,y,'r:',linewidth=6)
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Nasz pierwszy wykres')
# plt.grid(True)
# plt.show()
# x = np.arange(0.0, 2.0, 0.01)
# y1 = np.sin(2.0*np.pi*x)
# y2 = np.cos(2.0*np.pi*x)
# plt.plot(x,y1,'r:',x,y2,'g')
# plt.legend(('dane y1','dane y2'))
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Wykres')
# plt.grid(True)
# plt.show()
#
# x = np.arange(0.0, 2.0, 0.01)
# y1 = np.sin(2.0*np.pi*x)
# y2 = np.cos(2.0*np.pi*x)
# y = y1*y2
# l1, = plt.plot(x,y,'b')
# l2,l3 = plt.plot(x,y1,'r:',x,y2,'g')
# plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Wykres')
# plt.grid(True)
# plt.show()
#
#
#cwiczenia
#cw3
N=np.block([[np.arange(1,6)],
            [np.arange(5,0,-1)]])
print(N)
zm=np.block([[np.ones((2,3),int)*2], [np.arange(-90,-60,10)]])
print(zm)
B=np.block([np.zeros((3,2),int)])
print(B)
C=np.block([np.ones((5,1),int)*10])
print(C)
zm2=np.block([[N],
[B,zm]])
print(zm2)
A=np.block([zm2,C])
print(A)
#c4
B=A[1,:]+A[3,:]
print(B)
#c5
C = [np.max(A[:,0]),np.max(A[:,1]),np.max(A[:,2]),np.max(A[:,3]),np.max(A[:,4]),np.max(A[:,5])]
print(C)

#c7
np.put(D, np.where(D == 4), 0, mode='clip')
print(D)