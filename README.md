## Approximating Exponential Function by Composite Taylor Polynomials

A classical problem in approximation theory is to approximate a (real or
complex) function f by a polynomial p. Combining beautiful theory
and reliable algorithms, univariate polynomial approximation has reached a
mature stage, as implemented in the Chebfun software package. For example
for analytic functions on [-1, 1], polynomial approximation converges geometrically. Approximation theory is used virtually everywhere in scientific
computing.


A relatively uncharted question is to approximate f by a composite poly-
nomial, of the form q = pk(pk-1...(p2(p1))...). Composing polynomials is a
highly efficient way of generating high-degree polynomials, so they can po-
tentially be much more powerful than plain polynomials, with respect to the
degrees of freedom. In fact they are the crucial tool for most algorithms
for computing matrix functions (see Higham 2008), and one can understand
deep learning as a composition of large number of piecewise polynomials.
This project aims to investigate the power and limitations of composite
polynomials as a tool for approximating __exponential__ functions.

We focus particularly on __Scaling and Squaring__ Method based on __Composite Taylor Polynomials__ to approximate matrix exponential. We carry out some 
original research especially in the later sections of the dissertation. All the code related to the report is provided in this repository, however, the 
actual report is not attached yet. Once it is assessed, we will upload it here (if permission is given). 

An interactive jupyter notebooks can be seen here (in HTML): 
* ContourPlots: https://htmlpreview.github.io/?https://github.com/anmolaggarwal98/Scaling-and-Squaring-Method-Dissertation/blob/master/Code/Contour_plots.html
* Error Analysis Lines: https://htmlpreview.github.io/?https://github.com/anmolaggarwal98/Scaling-and-Squaring-Method-Dissertation/blob/master/Code/Error_Analysis_curves.html
* Matrix Exponential Approximation with SSM: https://htmlpreview.github.io/?https://github.com/anmolaggarwal98/Scaling-and-Squaring-Method-Dissertation/blob/master/Code/Matrix_error_analysis.html
* Oral Presentation https://htmlpreview.github.io/?https://github.com/anmolaggarwal98/Scaling-and-Squaring-Method-Dissertation/blob/master/Code/Presentation.html

It is possible that these link do not run since I have to make everything anonymous. Hence, the jupyter notebook files are also available in folder named `Code`. There respective graphs are available in the folder `plots` which might also not open because of the same anonymous reason. 

Please feel free to download the code and run it on your local machine so you can tweak different parameters. 

We hope you enjoy it. 

