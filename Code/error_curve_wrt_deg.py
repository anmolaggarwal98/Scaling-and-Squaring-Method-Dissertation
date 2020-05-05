import numpy as np
from scipy.special import factorial as factorial
import matplotlib.pyplot as plt
from math import pow as power
import pandas as pd
import os

from calculate_plain_taylor import *
from computingOptimalVal_m_s import error_list


def error_curve_deg(R = 20, max_m = 20, max_s=4,type = 'rel', no_of_pts = 1000, saveimage = False):
    S = np.arange(0,max_s+1)   #s in {0,....,4}
    if type == 'abs':
        print('ABSOLUTE ERROR')
    else:
        print('RELATIVE ERROR')
    #a way of choosing more points for small s and less for large s so that degrees remain constant
    M = [np.arange(1,max_m*(2**s)+1) for s in S][::-1]


    Z = random_points(R,no_of_pts)  #generate z values with Radius R
    # Z = [-20]
    Exact = np.exp(Z)  #exact using in-built function
    error_matrix = np.zeros((len(S),int(max_m*(2**S[-1]))),dtype = np.float64)  #creating an empty matrix to store errors

    truncation_para,Degrees, DoF, Flattening_error = error_list(R,S)

    print(f'\nMaximum {type} Error\n')
    for i,s in enumerate(S):
        for j,m in enumerate(M[i]):
            T_m_s = (approx_taylor_exp(Z/(2**s),m))**(power(2,s))   #composite poly
            error = rel_error(T_m_s, Exact,type)  #calculating the log10 rel error
            max_idx = np.argmax(error)
            max_error = max(error)  #choosing the maximum one
            idx = np.argmax(error)
            #print(f's = {s}, m={m}: {Z[idx]}')
            error_matrix[i,j] = max_error  #storing it in the matrix
        print(f'At s={s}\tz={Z[max_idx]}\tTheoretical log10 error={Flattening_error[i]}\tlog10(max_error)={max_error}')
    print()
    s = 1
    #for j,m in enumerate(M[1]):
        #print(np.log10(R/(2**s))*(m+1) - np.log10(float(factorial(m+1))))
    '''Plotting on same diagram because made sure all degrees are the same'''
    plt.figure(figsize=(10,8),facecolor='w')
    ax = plt.subplot(1, 1, 1)

    for i,s in enumerate(S):
        #print(M[i])
        #print(error_matrix[i])
        ax.plot([m*2**s for m in M[i]], error_matrix[i][:M[i][-1]], '.-',label = f's = {s}',\
                markersize = 12, lw = 1)
        d = int(R/2**s)


    plt.legend(prop={'size': 17})
    plt.title(f'Maximum Error (R: {R})',fontsize = 22)
    plt.xlabel(r'Degrees $2^sm$',fontsize = 18)
    plt.ylabel(r'Maximum $\log_{10}$ Relative error',fontsize = 18)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='k')


    ax.tick_params(labelsize=12)

    plt.tight_layout()

    if saveimage:
        # creates a folder plots to save  graphs in the folder
        path = r'C:\Users\xxxxx\OneDrive - Nexus365\Oxford Masters\Modules\Dissertation\Code\Candidate_no_1040706\plots\line_graphs'
        if not os.path.exists(path):
            os.mkdir(path)
            print('Folder `plots/line_graphs` is created\n')
        plt.savefig(path+f'\{type}_error_degrees.pdf', format='pdf', dpi=1200)

    plt.show()

if '__main__'==__name__:
    error_curve_deg(R = 20,
                  max_m = 20,
                  max_s=4,
                  type = 'abs',
                  no_of_pts = 1000,
                  saveimage = False)
