import numpy as np
from scipy.special import factorial as factorial
import matplotlib.pyplot as plt
from math import pow as power
import os

from calculate_plain_taylor import *
from computingOptimalVal_m_s import error_list

def error_curve_DoF(R = 20, max_m = 55, max_s=4,type = 'rel', no_of_pts = 1000, saveimage = False):
    '''Plotting the same thing but against DoF'''
    b = 3
    # max_m = 55
    # max_s = 4
    S = np.arange(0,max_s+1)   #s in {0,....,4}
    max_dof = (max_m+1)+b*max_s+10
    M_dof = [np.arange(1,get_m(s,max_dof,b)+1) for s in S]

    Z = random_points(R,no_of_pts)  #generate z values with Radius R
    Exact = np.exp(Z)  #exact using in-built function

    truncation_para,Degrees, DoF, Flattening_error = error_list(R,S)
    #print(M_dof)
    error_mat_dof = np.zeros((len(S),int(max_dof)))
    if type == 'abs':
        print('ABSOLUTE ERROR')
    else:
        print('RELATIVE ERROR')

    for i,s in enumerate(S):
        for j,m in enumerate(M_dof[i]):
            T_m_s = (approx_taylor_exp(Z/(2**s),m))**(2**s)   #composite poly
            error = rel_error(T_m_s, Exact,type)  #calculating the log10 rel error
            max_error = max(error)  #choosing the maximum one
            max_idx = np.argmax(error)
            error_mat_dof[i,j] = max_error  #storing it in the matrix
        print(f'At s={s}\tz={Z[max_idx]}\tTheoretical log10 error={Flattening_error[i]}\tlog10(max_error)={max_error}')
    plt.figure(figsize=(10,8))
    for i,s in enumerate(S):
        #print(error_matrix[i])
        plt.plot([((m+1)+b*s) for m in M_dof[i]], error_mat_dof[i][:M_dof[i][-1]], '.-',label = f's = {s}',\
                markersize = 12, lw = 1)

    plt.legend(prop={'size': 20})
    plt.title(f'Maximum Absolute Error (R: {R})',fontsize = 22)
    plt.xlabel(f'Degrees of Freedom (m+1) + {b}s',fontsize = 18)
    plt.ylabel(r'Maximum $\log_{10}$ Absolute error',fontsize = 18)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='k')
    #plt.tight_layout
    #plt.axhline(error_matrix[0,-1],ls = '--')
    #plt.savefig(f'plots/line_graphs/absolute_error_DoF.pdf', format='pdf', dpi=1200)
    #plt.xlim((0,80))
    if saveimage:
        # creates a folder plots to save  graphs in the folder
        if not os.path.exists('plots/line_graphs'):
            os.mkdir('plots/line_graphs')
            print('Folder `plots/line_graphs` is created\n')
        plt.savefig(f'plots\line_graphs/{type}_error_DoF.pdf', format='pdf', dpi=1200)
    plt.show()

if '__main__'==__name__:
    error_curve_DoF(R = 20,
                    max_m = 55,
                    max_s=4,
                    type = 'rel',
                    no_of_pts = 1000,
                    saveimage = False)
