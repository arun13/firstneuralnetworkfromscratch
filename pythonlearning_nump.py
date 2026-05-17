import numpy as np


# Matrix multiplication is a fundamental operation in linear algebra that involves multiplying two matrices together to produce a new matrix.

# without using numpy, we can implement matrix multiplication using nested loops.
def matrixMultWithoutNumpy(a,b):
    c = np.zeros((a.shape[0],b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                c[i,j] += a[i,k] * b[k,j]
    return c
# dpt is used for  dot product of two matrices, it is a built-in function in numpy that performs matrix multiplication. 
# It can be used to compute the dot product of two arrays,
#  which is a fundamental operation in linear algebra 
# and is widely used in various applications such as machine learning, physics, and engineering.
def matrixMultWithNumpy(a,b):
    return np.dot(a,b)

# matmul is another built-in function in numpy that performs matrix multiplication.
#  It is similar to np.dot but is more flexible and can handle higher-dimensional arrays.
#  The np.matmul function can be used to compute the matrix product of two arrays,
#  and it supports broadcasting, which allows for operations on arrays of different shapes.
def matrixMultWithNumpy2(a,b):
    return np.matmul(a,b)   


# @np.vectorize is a decorator in NumPy that allows you to apply a function element-wise to arrays.
# It is used to create a vectorized version of a function, which can be applied to arrays of any shape and size.
# When you use @np.vectorize, the decorated function is automatically applied to each element of the input arrays, and the output is returned as an array of the same shape.
# This can be particularly useful when you want to apply a function that is not natively supported  by NumPy to arrays, or when you want to improve the performance of a function by vectorizing it.    

def vectorizedFunction(x): 
    return x**2 + 2*x + 1
vectorizedFunction = np.vectorize(vectorizedFunction)  

# differentiate between np.dot and np.matmul
# np.dot is used for dot product of two matrices, it is a built-in function in  numpy that performs matrix multiplication. It can be used to compute the dot product of two arrays, which is a fundamental operation in linear algebra and is widely used in various applications such as machine learning, physics, and engineering.
# np.matmul is another built-in function in numpy that performs matrix multiplication. It is similar to np.dot but is more flexible and can handle higher-dimensional arrays. The np.matmul function can be used to compute the matrix product of two arrays, and it supports broadcasting, which allows for operations on arrays of different shapes.  
# A @ B is equivalent to np.matmul(A, B) and is used for matrix multiplication in Python 3.5 and later versions. It is a more concise and readable way to perform matrix multiplication compared to using np.dot or np.matmul directly.

# Let see how to do transpose of a matrix using numpy
# The transpose of a matrix is a new matrix that is obtained by swapping the rows and columns
def transposeMatrix(a):
    return np.transpose(a)

def transposeMatrix2(a):
    return a.T

# Vectortor vs matrix is a fundamental concept in linear algebra. 
# A vector is a one-dimensional array of numbers, while a matrix is a two-dimensional array of numbers.
# A vector can be represented as a column vector (n x 1) or a row vector (1 x n),
#  while a matrix can have any number of rows and columns.
# In terms of operations, vectors can be added, subtracted, and multiplied by scalars,
#  while matrices can be added, subtracted, multiplied by scalars, and multiplied by other matrices.       
