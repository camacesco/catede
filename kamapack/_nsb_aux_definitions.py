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


###################
#  POWER 2 TERMS  #
###################

def estimate_S2_at_alpha( a, compACTexp ) :
    
    # loading parameters from Experiment.compACT exp       
    N, nn, ff, K = compACTexp.N, compACTexp.nn, compACTexp.ff, compACTexp.K
    
    single_sum, double_sum = Power2_Term1( a, compACTexp )
    
    # single sum term
    Ss = (nn+a+1) * (nn+a) * single_sum
    
    # double sum term 
    Ds = ( (nn+a)[:,None] * (nn+a) ) * double_sum
        
    output = ff.dot( Ss - Ds.diagonal() + Ds.dot(ff) )
    output = mp.fdiv( output, mp.fmul(N+K*a+1, N+K*a) ) 
    
    return output
    
def estimate_DKL2_at_alpha_beta( a, b, compACTdiv ) :
    
    # loading parameters from Divergence Compact        
    N_A, N_B, K = compACTdiv.N_A, compACTdiv.N_B, compACTdiv.K
    nn_A, nn_B, ff = compACTdiv.nn_A, compACTdiv.nn_B, compACTdiv.ff
        
    single_sum1, double_sum1 = Power2_Term1( a, compACTdiv.compact_A )
    single_sum2, double_sum2 = Power2_Term2( a, b, compACTdiv )
    single_sum3, double_sum3 = Power2_Term3( b, compACTdiv.compact_B )
    
    Ss = (nn_A+a+1) * (nn_A+a) * ( single_sum1 - 2 * single_sum2 + single_sum3 )
    Ds = ( (nn_A+a)[:,None] * (nn_A+a) ) * ( double_sum1 - 2 * double_sum2 + double_sum3 )

    output = ff.dot( Ss - Ds.diagonal() + Ds.dot(ff) )
    output = mp.fdiv( output, mp.fmul(N_A+K*a+1, N_A+K*a) ) 
    
    return output    
    
#############
    
def Power2_Term1( a, compACTexp ) :
    '''
    squared term : q_i q_j ln(q_i) ln(q_j)
    '''
    # loading parameters from Experiment.compACT exp       
    N, nn, ff, K = compACTexp.N, compACTexp.nn, compACTexp.ff, compACTexp.K
    
    # double sum term 
    double_sum = D_polyGmm(0, nn+a+1, N+K*a+2)[:,None] * D_polyGmm(0, nn+a+1, N+K*a+2) - polygamma(1, N+K*a+2)
    # single sum term
    single_sum = np.power(D_polyGmm(0, nn+a+2, N+K*a+2), 2) + D_polyGmm(1, nn+a+2, N+K*a+2)
    
    return single_sum, double_sum


def Power2_Term2( a, b, compACTdiv ) :
    '''
    squared term : q_i q_j ln(q_i) ln(t_j)
    '''
    # loading parameters from Divergence Compact        
    N_A, N_B, K = compACTdiv.N_A, compACTdiv.N_B, compACTdiv.K
    nn_A, nn_B, ff = compACTdiv.nn_A, compACTdiv.nn_B, compACTdiv.ff
    
    # double sum term 
    double_sum = D_polyGmm(0, nn_A+a+1, N_A+K*a+2)[:,None] * D_polyGmm(0, nn_B+b, N_B+K*b)
    # single sum term
    single_sum = D_polyGmm(0, nn_A+a+2, N_A+K*a+2) * D_polyGmm(0, nn_B+b, N_B+K*b)

    return single_sum, double_sum


def Power2_Term3( b, compACTexp ) :
    '''
    squared term : q_i q_j ln(t_i) ln(t_j)
    '''
    # loading parameters from Experiment.compACT exp       
    N, nn, ff, K = compACTexp.N, compACTexp.nn, compACTexp.ff, compACTexp.K
    
    # double sum term 
    double_sum = D_polyGmm(0, nn+b, N+K*b)[:,None] * D_polyGmm(0, nn+b, N+K*b) - polygamma(1, N+K*b)
    # single sum term
    single_sum = D_polyGmm(1, nn+b, N+K*b)

    return single_sum, double_sum