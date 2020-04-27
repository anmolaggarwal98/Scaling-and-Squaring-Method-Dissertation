import numpy as np
from math import factorial
from numpy.linalg import norm, matrix_power
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt
import os
from calculate_plain_taylor import approx_taylor_exp

def experiment_taylor(s=3, m=20 , n = 500, a = -70, b = 70, saveimage = True):
    '''This contour plots visualize the error in absolute and relative terms
        for different values of z in complex plane when approximating exp(z) using scaling
        and squaring method'''

    xgrid = np.linspace(a,b,n+1)  #creating a grid
    X,Y = np.meshgrid(xgrid,xgrid) # -70 to 70 with n pts
    Z = X+1j*Y

    T_m = np.power(approx_taylor_exp(Z/(2**s),m),(2**s)) #SSM for each point in the grid
    exact_M = np.exp(Z)  #calculates the exact

#   computing the relative and absolute Error
    eps = 0.5*np.spacing(1)  #machine precision
    rel_Error = np.log10((abs((exact_M - T_m)/exact_M)) + eps)
    abs_Error = np.log10(abs(exact_M - T_m) + eps)

    '''Computing where relative error is maximised and minimised in the GRID'''
    max_rel_error_ind = np.where(rel_Error == rel_Error.max())
    min_rel_error_ind = np.argmin(rel_Error)  #flattens the matrix and then gives an index
    rel_i,rel_j = min_rel_error_ind//(n+1), min_rel_error_ind%(n+1)

    '''Computing where absolute error is maximised and minimised in the GRID'''
    max_abs_error_ind = np.where(abs_Error == abs_Error.max())
    min_abs_error_ind = np.argmin(abs_Error)  #flattens the matrix and then gives an index
    abs_i,abs_j = min_abs_error_ind//(n+1), min_abs_error_ind%(n+1)

    ax = plt.figure(figsize = (17,7))

    # plotting the relative error contour
    ax = plt.subplot(1, 2, 1)
    plt.contour(X,Y,rel_Error, levels = np.arange(-16,18,2),colors='black', linewidths=0.5);
    CS = plt.contourf(X,Y,rel_Error, levels = np.arange(-16,18,2));
    rel_real, rel_imag = Z[rel_i,rel_j].real, Z[rel_i,rel_j].imag
    rel_real_max, rel_imag_max = Z[max_rel_error_ind].real, Z[max_rel_error_ind].imag
    #plt.plot(rel_real,rel_imag,'rx',markersize = 10,label=f'min $z$');

    plt.plot(rel_real_max,rel_imag_max,'rD',markersize = 7,label='max $z$'); plt.legend();
    plt.colorbar(CS,orientation = 'horizontal',pad = 0.07).ax.tick_params(labelsize=13);
    plt.title(fr'Relative Error for $[T_m(z/2^s)]^{{2^s}}$   (m: {m}, s: {s})',fontsize = 22)
    if s ==0:
        plt.title(f'Relative Error for $T_m(z)$   (m: {m}, s: {s})',fontsize = 22)

    padding = 2.4
    plt.tick_params(labelsize=14);
    plt.xticks(np.arange(a, b+1, 10))
    plt.yticks(np.arange(a, b+1, 10))
    plt.xlim(a-padding,b+padding)
    plt.ylim(a-padding,b+padding)
    plt.legend(prop={'size': 15})
    plt.tight_layout()

    # plotting the relative error contour

#     ax = plt.figure(figsize = (8.5,7))
    #ax = plt.subplot(1, 2, 2,facecolor = (230/255, 237/255, 12/255))
    ax = plt.subplot(1, 2, 2)
    plt.contour(X,Y,abs_Error, levels = np.arange(-16,1,1),colors='k',linewidths=0.5);
    CS2 = plt.contourf(X,Y,abs_Error, levels = np.arange(-16,1,1));

    abs_real, abs_imag = Z[abs_i,abs_j].real, Z[abs_i,abs_j].imag
    abs_real_max, abs_imag_max = Z[max_abs_error_ind].real, Z[max_abs_error_ind].imag

    #plt.plot(abs_real,abs_imag,'rx',markersize = 10,label=f'min $z$');
    plt.plot(abs_real_max,abs_imag_max,'rD',markersize = 7,label=f'max $z$'); plt.legend();

    plt.colorbar(CS2, orientation = 'horizontal',pad = 0.07).ax.tick_params(labelsize=13);

    plt.title(f'Absolute Error for $[T_m(z/2^s)]^{{2^s}}$   (m: {m}, s: {s})',fontsize = 22)
    if s ==0:
        plt.title(f'Absolute Error for $T_m(z)$   (m: {m}, s: {s})',fontsize = 22)

    plt.xlim(a-padding,b+padding)
    plt.ylim(a-padding,b+padding)
    plt.xticks(np.arange(a, b+1, 10))
    plt.yticks(np.arange(a, b+1, 10))
    plt.tick_params(labelsize=14);
    plt.legend(prop={'size': 15})
    plt.tight_layout()

    if saveimage:
        # creates a folder plots to save  graphs in the folder
        path = r'C:\Users\Anmol\OneDrive - Nexus365\Oxford Masters\Modules\Dissertation\Code\Candidate_no_1040706\plots\contourplots'
        if not os.path.exists(path):
            os.mkdir(path)
            print('Folder `plots/contourplots` is created\n')
        plt.savefig(path+f'/error_m_{m}_s_{s}.pdf', format='pdf', dpi=1200)

    plt.show()
    print(f's={s} and m={m}\n')
    print('Relative Error Analysis\n=======================')
    print(f"Largest error z_jk:{[z for z in Z[max_rel_error_ind]]}")
    print(f'Maximum Error Order: {np.max(rel_Error)}\n')
    print(f"Smallest error z_jk: {Z[rel_i,rel_j]}")
    print(f'Minimum Error Order: {np.min(rel_Error)}\n')

    print('Absolute Error Analysis\n=======================')
    print(f"Largest error z_jk:{[z for z in Z[max_abs_error_ind]]}")
    print(f'Maximum Error Order: {np.max(abs_Error)}\n')
    print(f"Smallest error z_jk: {Z[abs_i,abs_j]}")
    print(f'Minimum Error Order: {np.min(abs_Error)}\n')
    print('=====================================================================================================================')


if __name__ == '__main__':
    # for composing polynomial with s = 3 and m = 20
    experiment_taylor(s = 3,
                     m = 20,
                     n = 500,
                     a = -70,
                     b = 70,
                     saveimage = False)

    # for plain Taylor with s=0 and m=160
    experiment_taylor(s = 0,
                     m = 160,
                     n = 500,
                     a = -70,
                     b = 70,
                     saveimage = False)
