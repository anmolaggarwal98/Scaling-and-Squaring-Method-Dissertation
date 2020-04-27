import numpy as np
import pandas as pd
from scipy.special import factorial as factorial
from math import pow as power

def _optimalval(s,R):
    '''Calculating the truncation parameter m where we see line flattening'''
    e_mach = 0.5*np.spacing(1)
    max_m = int(R/2**s)   #value of m where we have the largesr error
    delta = np.abs(np.exp(-R/2**s)-((-R/2**s)**max_m)/factorial(max_m))*e_mach   #as shown in the above equation
    m = 0
    error = R**(m+1)/(power(2,s*(m+1))*factorial(m+1))
    while error > delta:
        m+=1
        error = R**(m+1)/(power(2,s*(m+1))*factorial(m+1))
    return m

def error_list(R = 20, S = np.arange(0,5)):
    e_mach = 0.5*np.spacing(1)
    truncation_para = []  #list to store m
    Degrees = []   #list to store degree of poly (2^s*m)
    DoF = []       #list to store DoF ((m+1)+3s)
    Flattening_error = []  #list to store error where flattening begins
    print(f'Value of truncation parameter m where we see line flattening is:\n')
    for s in S:
        max_m = int(R/2**s)   #value of m where we have the largesr error
        m = _optimalval(s,R)
        truncation_para.append(m)

        '''Here we use above equation to calculate the rel error where the flatten takes place'''
        flatting_val = np.log10(2**s*np.abs(((-R/2**s)**max_m)/factorial(max_m))*np.exp(R/2**s)*e_mach)
        Flattening_error.append(flatting_val)

        '''Calculating the degree and DoF for using the m and s value computed'''
        Degrees.append(2**s*m)
        DoF.append((m+1)+3*s)

    df = pd.DataFrame([[m,deg,dof, flattening_val] for m,deg,dof,flattening_val in zip(truncation_para, Degrees, DoF,Flattening_error)],\
                        columns = [r'Truncation Parameter ($m$)',r'Degrees ($2^sm$)', fr'DoF $((m+1)+3s)$', r'$\log_{10}(\text{Flattening})$'],\
                        index = [fr'$s = {s}$' for s in S])
    if '__main__'==__name__:
        print(df)
    return truncation_para,Degrees, DoF, Flattening_error

if '__main__'==__name__:
    error_list()
