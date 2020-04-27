import numpy as np
from scipy.special import factorial as factorial

def random_points(R =10, no_of_pts = 200):
    '''gives user-defined number of complex points when a radius R is given'''
    #Theta = np.linspace(-np.pi,np.pi,no_of_pts)
    Theta = np.random.uniform(-np.pi,np.pi,no_of_pts)
    Z = [R*np.exp(1j*theta) for theta in Theta]  #R*np.cos(theta)+1j*R*np.sin(theta)
    return np.array(Z)

def approx_taylor_exp(x,m):
        '''Calculates the truncated taylor series of degree m and returns the value
           at the point x'''
        f_m = list(map(lambda j: 1/factorial(j),range(0,m+1)))
        f_m.reverse()
        p = np.poly1d(f_m)
        return p(x)

def rel_error(list1,exact,type='abs'):
    '''Calcutes the log10 of the relative forward error'''
    eps = 0.5*np.spacing(1)
    if type=='rel':
        error = list(map(lambda vec: np.log10((abs(vec[0] - vec[1])/(abs(vec[1])))+eps), zip(list1,exact)))
    elif type=='abs':
        error = list(map(lambda vec: np.log10(abs(vec[0] - vec[1])+eps), zip(list1,exact)))
    return error

def get_m(s,max_dof,b):
    '''Used only for DoF. For every s I give it, it gives me a value of truncation parameter
       m s.t. maximum DOF remains constant'''

    m = max_dof - b*s - 1
    if m<=0:
        raise ValueError('the value of m must be positive so please check your code')
    return int(m)
