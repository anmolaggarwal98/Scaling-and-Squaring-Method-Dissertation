import numpy as np
from math import factorial
import time
import matplotlib.pyplot as plt
from math import pow as power
import os

def theo_error_curve(m=80,R = 20,s = 0,saveimage=False):
    '''It computes the actual error and theoretical error in relative sense
        for a fixed m,s and R'''
    print(f'm:{m}, s:{s} and R:{R}')
    M = m
    f_m = [((-R)**k/(2**(k*s)*factorial(k))) for k in range(0,M+1)]

    Error = np.zeros(M+1)

    # theoretical error 
    for idx,m in enumerate(np.arange(M+1)):
        error = (power(R,m+1)*np.exp(R/(2**s)))/(power(2,s*m)*factorial(m+1))
        Error[idx] = error

    plt.figure(figsize=(12,9))
    ax = plt.subplot(1, 1, 1)

    ax.plot(np.arange(M+1),np.log10(np.abs((np.cumsum(f_m))**(2**s)-np.exp(-R))/np.exp(-R)),'-k',lw=4,label = fr'Computed error');
    ax.plot(np.arange(M+1),np.log10(Error),'--r',lw=4,label = fr'Theoretical error');
    #plt.plot(range(0,M+1),np.log10(np.absolute((1/np.cumsum(f_m))**(power(2,s))-np.exp(-R))/np.exp(-R)),'-k',lw=3,label = fr'relative error$');

    #plt.axhline(y=np.log10(np.exp(-R)),xmin=0,ls='--',color = 'b',lw=2, label=fr'Order of $\exp({-R})$')
    #plt.axhline(y=np.log10(np.spacing(1)*0.5),xmin=0,ls='--',color = 'b',lw=2, label = r'$\epsilon_{mach}\approx 1\times 10^{-16}$')   #machine precision

    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='k')
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 10))
    plt.xlabel('Truncation parameter $m$ for $T_m(-R/2^s)^{2^s}$',fontsize = 24)
    plt.ylabel(r"$\log_{10}$ Relative Error",fontsize = 24)
    #plt.title(fr'Relative Error for Composite Taylor for $s={s}$ at $z={-R}$',fontsize=18)
    plt.legend(loc=1,prop={'size': 25})
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    if saveimage:
        # creates a folder plots to save  graphs in the folder
        path = r'C:\Users\Anmol\OneDrive - Nexus365\Oxford Masters\Modules\Dissertation\Code\Candidate_no_1040706\plots\theoretical_plots'
        if not os.path.exists(path):
            os.mkdir(path)
            print('Folder `plots/theoretical_plots` is created\n')
        plt.savefig(path+f'/theo_s{s}.pdf', format='pdf', dpi=1200)
    plt.show()

if '__main__'==__name__:
    theo_error_curve(m=80,R=20,s=1,saveimage = False)
