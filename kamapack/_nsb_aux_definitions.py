#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Nemenmann-Shafee-Bialek Auxilary Definitions
    Copyright (C) November 2021 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
from scipy.special import loggamma, polygamma
from scipy import optimize

##############
#  NOTATION  #
##############

def D_polyGmm(order, x, y):
    '''
    Difference between same `order` polygamma functions, computed in `x` and `y`.
    '''
    return polygamma(order,x) - polygamma(order,y)  

def LogGmm( x ): 
    ''' alias '''
    return loggamma( x ).real    

def measureMu( a, compACT ) :
    '''
    Measure Mu term in the posterior estimators computed as the exponent of an exponential.
    '''
        
    # loading parameters from compACT        
    N, nn, ff, K = compACT.N, compACT.nn, compACT.ff, compACT.K
    
    # mu computation    
    LogMu = LogGmm( K*a ) - K * LogGmm( a )                   # Dirichelet prior normalization contribution
    LogMu += ff.dot( LogGmm(nn+a) ) - LogGmm( N + K*a )       # posterior contribution

    return mp.exp( LogMu )

def integral_with_mu( mu, func, x ) :
    ''' alias '''   
    return np.trapz(np.multiply(mu, func), x=x)

########################
#  get_from_implicit  #
########################

def get_from_implicit( implicit_relation, y, lower, upper, *args,
                      maxiter=100, xtol=1.e-20 ):
    '''
    Find the root of the implicit relation for x in (0, infty):  
    >    `implicit relation` ( x, *args ) - `y` = 0
    It uses the Brent's algorithm for the root finder in the interval (lower, upper)
    '''   

    # NOTE : the implicit_realtion must have opposite signs in 0 and up_bound
    output = optimize.brentq( implicit_relation, lower, upper,
                             args=( y , *args ), xtol=xtol, maxiter=maxiter )
    return output
                                                  
def implicit_S_vs_Alpha( alpha, S, K ):
    '''
    implicit relation to be inverted.
    '''
    return D_polyGmm( 0, K * alpha + 1, alpha + 1 ) - S   

def implicit_H_vs_Beta( beta, x, K ):
    '''
    implicit relation to be inverted.
    '''
    return D_polyGmm( 0, K * beta , beta ) - x

    