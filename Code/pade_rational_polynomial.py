import numpy as np
from scipy.special import factorial as factorial
import matplotlib.pyplot as plt


def experiment_pade(s=3  ,k =10, m=10, n = 500, a = -70, b = 70,annotate=True,saveimage=True):

    eps = np.spacing(1)  #machine precision

    def pade_approx(k,m,x):
        def p(k,m,x):
            terms = np.zeros(k+1,dtype=np.ndarray)
            for j in range(k+1):
                coeff = (factorial(k)*factorial(k+m-j))/(factorial(k+m)*factorial(k-j)*factorial(j))
                terms[j] = coeff*(x**j)
            return np.sum(terms)
        return p(k,m,x)/p(m,k,-x)

    def approx_taylor_exp(x,m):
        f_m = list(map(lambda k: 1/factorial(k),range(0,m+1)))
        f_m.reverse()
        p = np.poly1d(f_m)
        return p(x)

    xgrid = np.linspace(a,b,n+1)
    X,Y = np.meshgrid(xgrid,xgrid)
    Z = X+1j*Y

    R_m = pade_approx(k,m,Z/2**s)**(2**s)
    exact_M = np.exp(Z)

    rel_Error = np.log10((abs(exact_M - R_m)/abs(exact_M)) + eps)
    abs_Error = np.log10(abs(exact_M - R_m) + eps)

    max_rel_error_ind = np.where(rel_Error == rel_Error.max())
    min_rel_error_ind = np.argmin(rel_Error)  #flattens the matrix and then gives an index
    rel_i,rel_j = min_rel_error_ind//(n+1), min_rel_error_ind%(n+1)

    max_abs_error_ind = np.where(abs_Error == abs_Error.max())
    min_abs_error_ind = np.argmin(abs_Error)  #flattens the matrix and then gives an index
    abs_i,abs_j = min_abs_error_ind//(n+1), min_abs_error_ind%(n+1)

    fig, axes = plt.subplots(2,2,sharey=True,gridspec_kw={'hspace': 0.3, 'wspace': 0.05},figsize=(17,14))
    ax0,ax1,ax2,ax3 = axes.flatten()
    if annotate:
        if s>0:
            title = fr'The contour plot shows the error of approximating $\exp(x)$ with a Composite Taylor $[T_{{m}}(z/2^s)]^{{2^s}}$\
                    and Composite Pad$\'e$ polynomial $[R_{{k,m}}(z/2^s)]^{{2^s}}$ with the polynomial degree of $2^sm$ and $2^s(m+k)$ respectively\
                    over the domain [{a},{b}].'
        if s==0:
            title = fr'The contour plot shows the error of approximating $\exp(x)$ with a plain Taylor $T_{{m}}(z)$\
                        and plain Pad$\'e$ polynomial $R_{{k,m}}(z)$ with the polynomial degree of $m$ and $(m+k)$ respectively\
                        over the domain [{a},{b}].'

        fig.suptitle(title,fontsize=15)

    else:
        fig.suptitle(r'Composite Pade $[R_{{k,m}}(z/2^s)]^{{2^s}}$ vs Composite Taylor $[T_{{m}}(z/2^s)]^{{2^s}}$ Approximation of $\exp(z)$ in both absolute and relative sense. We have\
tried to keep the degree of both polynomial the same for a fair comparison.',fontsize=15)
    rel_real, rel_imag = Z[rel_i,rel_j].real, Z[rel_i,rel_j].imag
    rel_real_max, rel_imag_max = Z[max_rel_error_ind].real, Z[max_rel_error_ind].imag

    ax1.plot(rel_real_max,rel_imag_max,'ro',markersize = 8,label='max $z$'); ax1.legend(prop={'size': 15});
    ax1.contour(X,Y,rel_Error, levels = np.arange(-16,18,1),colors='black', linewidths=0);
    CS = ax1.contourf(X,Y,rel_Error, levels = np.arange(-16,18,1));
    fig.colorbar(CS,ax=(ax1,ax3),orientation = 'horizontal',pad = 0.05);
    if s==0:
        ax1.set_title(f'Relative Error for $R_m(z)$   (s: {s}, k: {k},m: {m})',fontsize=15)
    else:
        ax1.set_title(fr'Relative Error for $[R_{{k,m}}(z/2^s)]^{{2^s}} (s: {s}, k: {k},m: {m})$',fontsize=15)
    padding = 2.4
    ax1.tick_params(labelsize=14);
    ax1.set_xticks(np.arange(a, b+1, 10))
    ax1.set_yticks(np.arange(a, b+1, 10))
    ax1.set_xlim(a-padding,b+padding)
    ax1.set_ylim(a-padding,b+padding)
    ax1.legend(prop={'size': 13})

    abs_real, abs_imag = Z[abs_i,abs_j].real, Z[abs_i,abs_j].imag
    abs_real_max, abs_imag_max = Z[max_abs_error_ind].real, Z[max_abs_error_ind].imag

    ax0.plot(abs_real_max,abs_imag_max,'ro',markersize = 8,label='max $z$'); ax0.legend(prop={'size': 15});
    ax0.contour(X,Y,abs_Error, levels = np.arange(-16,2,1),colors='black', linewidths=0);
    CS2 = ax0.contourf(X,Y,abs_Error, levels = np.arange(-16,2,1));
    fig.colorbar(CS2, ax=(ax0,ax2),orientation = 'horizontal',pad = 0.05);
    if s==0:
        ax0.set_title(f'Absolute Error for $R_m(z)$   (s: {s}, k: {k},m: {m})',fontsize=15)
    else:
        ax0.set_title(fr'Absolute Error for $[R_{{k,m}}(z/2^s)]^{{2^s}} (s: {s}, k: {k},m: {m})$',fontsize=15)
    ax0.tick_params(labelsize=14);
    ax0.set_xticks(np.arange(a, b+1, 10))
    ax0.set_yticks(np.arange(a, b+1, 10))
    ax0.set_xlim(a-padding,b+padding)
    ax0.set_ylim(a-padding,b+padding)
    ax0.legend(prop={'size': 13})
    #############################################################################################################

    m = 2*m
    T_m = np.power(approx_taylor_exp(Z/(2**s),m),(2**s))

    rel_Error = np.log10((abs((exact_M - T_m)/exact_M)) + eps)
    abs_Error = np.log10(abs(exact_M - T_m) + eps)

    max_rel_error_ind = np.where(rel_Error == rel_Error.max())
    min_rel_error_ind = np.argmin(rel_Error)  #flattens the matrix and then gives an index
    rel_i,rel_j = min_rel_error_ind//(n+1), min_rel_error_ind%(n+1)

    max_abs_error_ind = np.where(abs_Error == abs_Error.max())
    min_abs_error_ind = np.argmin(abs_Error)  #flattens the matrix and then gives an index
    abs_i,abs_j = min_abs_error_ind//(n+1), min_abs_error_ind%(n+1)

    ax3.contour(X,Y,rel_Error, levels = np.arange(-16,18,2),colors='black', linewidths=0);
    CS = ax3.contourf(X,Y,rel_Error, levels = np.arange(-16,18,2));

    rel_real, rel_imag = Z[rel_i,rel_j].real, Z[rel_i,rel_j].imag
    rel_real_max, rel_imag_max = Z[max_rel_error_ind].real, Z[max_rel_error_ind].imag

    '''plt.plot(rel_real,rel_imag,'rx',markersize = 10,label=f'min $z$');'''

    ax3.plot(rel_real_max,rel_imag_max,'rD',markersize = 7,label='max $z$'); ax3.legend(prop = {'size':13});

    if s==0:
        ax3.set_title(fr'Relative Error for $T_m(z)$  (m: {m}, s: {s})',fontsize = 15)
    else:
        ax3.set_title(fr'Relative Error for $[T_m(z/2^s)]^{{2^s}}$   (m: {m}, s: {s})',fontsize = 15)

#     padding = 2.4
    ax3.tick_params(labelsize=14);
    ax3.set_xticks(np.arange(a, b+1, 10))
    ax3.set_yticks(np.arange(a, b+1, 10))
    ax3.set_xlim(a-padding,b+padding)
    ax3.set_ylim(a-padding,b+padding)
    ax3.legend(prop={'size': 13})

    ax2.contour(X,Y,abs_Error, levels = np.arange(-16,1,1),colors='k',linewidths=0);
    CS2 = ax2.contourf(X,Y,abs_Error, levels = np.arange(-16,1,1));
    abs_real, abs_imag = Z[abs_i,abs_j].real, Z[abs_i,abs_j].imag
    abs_real_max, abs_imag_max = Z[max_abs_error_ind].real, Z[max_abs_error_ind].imag
    if s==0:
        ax2.set_title(fr'Absolute Error for $T_m(z)$  (m: {m}, s: {s})',fontsize = 15)
    else:
        ax2.set_title(fr'Absolute Error for $[T_m(z/2^s)]^{{2^s}}$   (m: {m}, s: {s})',fontsize = 15)
    ax2.plot(abs_real_max,abs_imag_max,'rD',markersize = 7,label=f'max $z$'); ax2.legend(prop = {'size':13});
    ax2.tick_params(labelsize=14);
    ax2.set_xticks(np.arange(a, b+1, 10))
    ax2.set_yticks(np.arange(a, b+1, 10))
    ax2.set_xlim(a-padding,b+padding)
    ax2.set_ylim(a-padding,b+padding)
    ax2.legend(prop={'size': 13})
#     plt.colorbar(CS2, orientation = 'horizontal',pad = 0.07).ax.tick_params(labelsize=13);
    #########################################################################################################
    if annotate:
        if s>0:
            analyses = fr'Note: both Composite Taylor and Pade have the same polynomial degree of {2**s*m} but it looks like Pade does suprisingly well for\
            a fixed s at {s} in both absolute and relative case. In terms of the maximum errors, both Taylor and Pade have them at the exact same spots\
            (Re(z)={b} and Re(z)={a} for absolute and relative respectively.)'
        if s==0:
            analyses = fr'Note: both plain Taylor and Pade have the same polynomial degree of {m} but it looks like Pade does suprisingly well for\
            in both absolute and relative case. In terms of the maximum errors, both Taylor and Pade have them at the exact same spots\
            (Re(z)={b} and Re(z)={a} for absolute and relative respectively). In relative error, Pade seems to do really well to approximate\
            exponential function when Re(z)<0, which we always had trouble in Taylor approximation.'
        fig.text(0.5,0.12, analyses, horizontalalignment='center', verticalalignment='bottom',fontsize=15)

    if saveimage:
        path = r'C:\Users\Anmol\OneDrive - Nexus365\Oxford Masters\Modules\Dissertation\Code\Candidate_no_1040706\plots\pade_comparison'
        plt.savefig(path + f'\group_comp_s_{s}_m_{m/2}_k_{k}.pdf', format='pdf', dpi=500)
    plt.show()

    print('\nRelative Error Analysis\n=======================')
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


if __name__=='__main__':
    experiment_pade(s=2,k =20, m=20, n = 500, a = -70, b = 70,annotate=False,saveimage=False)
