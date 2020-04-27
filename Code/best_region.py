import numpy as np
from math import factorial
from numpy.linalg import norm, matrix_power
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt
import os
from calculate_plain_taylor import approx_taylor_exp

def experiment_taylor(S=[3], M=[18] , n = 400, saveimage = True):
    '''Plots the contour plot and also the region computed in my report
       R which is theoretical region of best approximation

       Parameters:
       =============
       S: list
          Contains a list of all scaling parameters
       M: list
          Contains a list of all truncation parameters
       n: int
          How many points to take in the mesh
       saveimage: boolean
          whether to save image in a folder or not'''

    if len(S) != len(M):
        raise ValueError('The dimension of S and M set should be the same to have one to one mapping')

    def segment(m,s):
        '''Computes the Theoretical region given the machine precision and s and m values'''
        eps = 0.5*np.spacing(1)  #1x10^-16
        r = ((eps**(2**(-s)))*(2**(s*(m+1))*factorial(m+1)))**(1/(m+1))  #r_{s,m} in pg 23
        re_z_high = np.log(10**(-16))
        return r, re_z_high

    no_of_rows = len(M)//2 + max(len(M)%2,0)
    no_of_cols = min(len(list(zip(M,S))),2)
    fig, axes = plt.subplots(no_of_rows,no_of_cols,gridspec_kw={'hspace': 0.2, 'wspace': 0.1},figsize=(17,7*(no_of_rows)))

    if len(list(zip(M,S))) != 1:
        axes = axes.flatten()
    fig.suptitle(r'Absolute Error for $[T_m(z/2^s)]^{{2^s}}$',fontsize = 22)

    for idx,(m, s) in enumerate(zip(M,S)):

        r,re_z_high = segment(m,s)
        r,re_z_high = int(round(r)), int(round(re_z_high))  #rounding them so they look nicer on graph

        print(f'For m:{m} and s:{s}\t r_{{m,s}}={round(r,2)}')

        # this is just aesthetic - trying the graph look nice
        gap = r*0.1
        a = int(-r-gap)
        b = int(r+gap)

        xgrid = np.linspace(a,b,n+1)
        X,Y = np.meshgrid(xgrid,xgrid)
        Z = X+1j*Y

        T_m = approx_taylor_exp(Z/2**s,m)**(2**s)
        exact_M = np.exp(Z)
        eps = np.spacing(1)  #machine precision

        rel_Error = np.log10((abs(exact_M - T_m)/abs(exact_M)) + eps)
        abs_Error = np.log10(abs(exact_M - T_m) + eps)

        # plt.figure(figsize = (8.5,7))

        # ax = plt.subplot(1, 1, 1)
        if len(list(zip(M,S))) !=1:
            ax = axes[idx]
        else:
            ax = axes
        ax.contour(X,Y,abs_Error, levels = np.arange(-16,1,1),colors='black', linewidths=0.5);
        CS2 = ax.contourf(X,Y,abs_Error, levels = np.arange(-16,1,1));
        # plt.colorbar(CS2, orientation = 'horizontal',pad = 0.07).ax.tick_params(labelsize=13);

        ax.axvline(x= -r ,ls='--',color = 'r',lw=3)
        ax.axvline(x= re_z_high,ls='--',color = 'y',lw=3)

        # creating the segment
        seg_x = np.array(list(filter(lambda val: abs(val) < r and np.real(val) < re_z_high, Z.reshape(-1))))
        real,imag = seg_x.real, seg_x.imag

        ax.fill(real,imag,color = 'r', alpha = 0.4,hatch='x')

        circ = plt.Circle((0, 0), radius=r, color='k',fill=False,linestyle = '--',linewidth =  3)
        ax.add_patch(circ)

        ax.set_title(f's: {s}, m: {m}',fontsize = 15)

        ax.set_xlim(a,b)
        ax.set_ylim(a,b)
        ax.set_xticks(np.arange(a, b+1, 10))
        ax.set_yticks(np.arange(a, b+1, 10))

        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', colors='k')
        ax.set_yticks([])
        ax.set_xticks([-r,re_z_high])

        # plt.tight_layout()
    fig.colorbar(CS2, ax=axes, orientation='horizontal',pad=0.05).ax.tick_params(labelsize=13)

    if saveimage:
        # creates a folder plots to save  graphs in the folder
        path = r'C:\Users\Anmol\OneDrive - Nexus365\Oxford Masters\Modules\Dissertation\Code\Candidate_no_1040706\plots\segmentplots'
        if not os.path.exists(path):
            os.mkdir(path)
            print('Folder `plots/segmentplots` is created\n')
        # plt.savefig(f'plots/segmentplots/m_{m}_s_{s}.png', format='png', dpi=1200)
        plt.savefig(path+'\grouped_contour_plot.pdf', format='pdf', dpi=1200)

    plt.show()


if '__main__'==__name__:
    experiment_taylor(S=[3,5,3,4] ,M=[20,20,30,30] , n = 400, saveimage=False)
