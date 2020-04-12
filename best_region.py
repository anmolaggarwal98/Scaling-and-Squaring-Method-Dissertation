import numpy as np
from math import factorial
from numpy.linalg import norm, matrix_power
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt
import os
from calculate_plain_taylor import approx_taylor_exp

def experiment_taylor(s=3, m=18 , n = 400, saveimage = True):
    '''Plots the contour plot and also the region computed in my report
       R which is theoretical region of best approximation'''

    def segment(m,s):
        eps = 0.5*np.spacing(1)
        r = ((eps**(2**(-s)))*(2**(s*(m+1))*factorial(m+1)))**(1/(m+1))
        print(f'For m:{m} and s:{s}\t r_{{m,s}}={round(r,2)}')
        re_z_high = np.log(10**(-16))
        return r, re_z_high

    r,re_z_high = segment(m,s)
    r,re_z_high = int(round(r)), int(round(re_z_high))

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

    plt.figure(figsize = (8.5,7))

    ax = plt.subplot(1, 1, 1)
    plt.contour(X,Y,abs_Error, levels = np.arange(-16,1,1),colors='black', linewidths=0.5);
    CS2 = plt.contourf(X,Y,abs_Error, levels = np.arange(-16,1,1));
    plt.colorbar(CS2, orientation = 'horizontal',pad = 0.07).ax.tick_params(labelsize=13);

    plt.axvline(x= -r ,ls='--',color = 'r',lw=3)
    plt.axvline(x= re_z_high,ls='--',color = 'y',lw=3)

    # creating the segment
    seg_x = np.array(list(filter(lambda val: abs(val) < r and np.real(val) < re_z_high, Z.reshape(-1))))
    real,imag = seg_x.real, seg_x.imag

    plt.fill(real,imag,color = 'r', alpha = 0.4,hatch='x')

    circ = plt.Circle((0, 0), radius=r, color='k',fill=False,linestyle = '--',linewidth =  3)
    ax.add_patch(circ)

    plt.title(f'Absolute Error for $T_m$   (s: {s}, m: {m})',fontsize = 22)

    plt.xlim(a,b)
    plt.ylim(a,b)
    plt.xticks(np.arange(a, b+1, 10))
    plt.yticks(np.arange(a, b+1, 10))
    #plt.tick_params(labelsize=14,);
    plt.tick_params(labelsize=14)
    ax.tick_params(axis='x', colors='k')
    ax.set_yticks([])
    ax.set_xticks([-r,re_z_high])
    #plt.axis('off')
    plt.tight_layout()

    if saveimage:
        # creates a folder plots to save  graphs in the folder
        if not os.path.exists('plots/segmentplots'):
            os.mkdir('plots/segmentplots')
            print('Folder `plots/segmentplots` is created\n')
        plt.savefig(f'plots/segmentplots/m_{m}_s_{s}.png', format='png', dpi=1200)

    plt.show()


if '__main__'==__name__:
    experiment_taylor(s=3 ,m=30 , n = 400, saveimage=False)
