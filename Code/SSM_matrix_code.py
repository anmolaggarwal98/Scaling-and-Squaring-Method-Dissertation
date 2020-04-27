import numpy as np
from math import factorial
from numpy.linalg import norm, matrix_power
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt


def taylor_exp(A, tol = 1e-8, p = 2):
    '''Compute the approximation of exp(A) using scaling and squaring method
       based on composing Taylor polynomials. We have used the theory from
       'Nineteen Dubious Ways to Compute the Exponential of a Matrix
       ,Twenty-Five Years Later' by Moler and Loan

       Parameters
       ==========
       A: np.ndarray
          This is a square matrix
       tol: float
            user defined accuracy
       p: int or 'fro', np.inf
          the matrix norm we are using

       output
       ============
       Matrix: np.ndarray
               approximation of exp(A)
       Error: float
              the relative error between our approximation and actual exp'''

    if A.shape[0] != A.shape[1]:
        print('Make sure matrix is a square matrix')
        return

    start = time.perf_counter() #timing the code
    n = len(A)
    print(f'Dim of A n: {n}\n')
    print(f'The Matrix A:\n{A}\n')

    norm_A = norm(A,ord = p) #p-norm

    s = int(np.ceil(np.log2(norm_A)+1))  # by solving ||A||/2^s < 1
    print(f'The value of squaring size s: {s}\n')

    B = A/2**s

    k = 0
    M0 = np.eye(n)  #identity matrix

    while norm(matrix_power(B,k),ord = p)/factorial(k) > tol:
        k+=1
        M0 = np.add(M0,matrix_power(B,k)/factorial(k))

    end = time.perf_counter()
    print('DONE')
    print(f'We truncated our Taylor Series to m = {k-1} terms where m(tol)\n') #by the code we add calculate dimension of T_m not degree

    M = matrix_power(M0,2**s)
    exact_M = expm(A)

    print(f'Actual e^A using expm function:\n{exact_M}\n')
    print(f'Estimation :\n{M}\n')

    error = norm(M-exact_M,ord=p)/(norm(exact_M,ord = p))

    print(f'Relative Forward Error: {error}')
    print(f'Wrong by: {round(np.log10(error))} digits') #if we have negative val then its accurate to that many digits
    print(f'Time required for computation: {end-start} seconds')
    print('===============================================================')

    return error,M,s,k  #here my m = k

if __name__ == '__main__':
    tol = 0.5*np.spacing(1)  #machine precision
    A = np.array([[-49,24],[-64,31]])
    norm_A = norm(A,ord = 2) #2 norm
    taylor_exp(A,tol = tol, p=2)
