#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Nemenmann-Shafee-Bialek Auxilary Definitions
    Copyright (C) March 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
from scipy.special import loggamma, polygamma
from scipy import optimize

##############
#  NOTATION  #
##############

def diGmm(x) :    
    '''Digamma function (polygamma of order 0).'''
    return polygamma(0, x)

def triGmm(x) :    
    '''Trigamma function (polygamma of order 1).'''
    return polygamma(1, x)

def D_diGmm(x, y):
    '''Difference between digamma functions in `x` and `y`.'''
    return diGmm(x) - diGmm(y)  

def D_triGmm(x, y):
    '''Difference between trigamma functions in `x` and `y`.'''
    return triGmm(x) - triGmm(y)  

def LogGmm( x ): 
    ''' alias '''
    return loggamma( x ).real    

################
#  POSTERIORS  #
################

def measureMu( alpha, compACT ) :
    '''Measure Mu term in the posterior estimators'''
        

    # mu computation  :  
    # Dirichelet prior normalization contribution
      

    return mp.exp( LogMu(alpha, compACT) )

def LogMu( alpha, compACTexp ) :
    '''logarithm computation of Measure Mu term.'''
    N, K = compACTexp.N,  compACTexp.K
    nn, ff = compACTexp.nn, compACTexp.ff

    # log(mu) computation  :  
    # Dirichelet prior normalization contribution
    output = LogGmm( K*alpha ) - K * LogGmm( alpha )                  
    # posterior contribution 
    output += ff.dot( LogGmm(nn+alpha) ) - LogGmm( N + K*alpha ) 
    return output

def optimal_dirichlet_param( compACTexp, upper=1e-4, lower=1e2 ) :
    '''Return Dirchlet parameter which optimizes entropy posterior (~).''' 

    def implicit_relation( x, y, compACTexp ):
        N, K = compACTexp.N,  compACTexp.K
        nn, ff = compACTexp.nn, compACTexp.ff

        tmp = K * diGmm(N+K*x) - K * diGmm(K*x) + K * diGmm(x) - ff.dot(diGmm(nn+x))
        return tmp - y

    output = get_from_implicit(implicit_relation, 0, upper, lower, compACTexp )
    return output

def optimal_ed_param( compACTdiv, upper=1e-4, lower=1e2 ) :
    '''Return Dirchlet parameter which optimizes divergence posterior alpha=beta (~).''' 

    def implicit_relation( x, y, compACTdiv ):
        N_1, N_2, K  = compACTdiv.N_1, compACTdiv.N_2, compACTdiv.K
        nn_1, nn_2, ff = compACTdiv.nn_1, compACTdiv.nn_2, compACTdiv.ff

        tmp = K * diGmm(N_1+K*x) - K * diGmm(K*x) + K * diGmm(x) - ff.dot(diGmm(nn_1+x))
        tmp += K * diGmm(N_2+K*x) - K * diGmm(K*x) + K * diGmm(x) - ff.dot(diGmm(nn_2+x))

        return tmp - y

    output = get_from_implicit(implicit_relation, 0, upper, lower, compACTdiv )
    return output


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
                                                  
def implicit_entropy_vs_alpha( alpha, entropy, K ):
    '''
    implicit relation to be inverted.
    '''
    return D_diGmm( K * alpha + 1, alpha + 1 ) - entropy

def implicit_crossentropy_vs_beta( beta, crossentropy, K ):
    '''
    implicit relation to be inverted.
    '''
    return D_diGmm( K * beta , beta ) - crossentropy


############
#  others  #
############

def number_of_coincidences( compACTexp ) :
    '''Number of coincidences in the experiment.'''
    output = compACTexp.N - np.sum(compACTexp.ff[compACTexp.nn == 1])
    return output

    