#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Nemenmann-Shafee-Bialek Estimator
    Copyright (C) October 2021 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
from scipy.special import loggamma, polygamma
from scipy import optimize
import multiprocessing

def NemenmanShafeeBialek( compACT, bins=None, error=False ):
    '''
    NSB_entropy Function Description:
    '''

    N, K = compACT.N, compACT.K
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>
    
    if bins is None:
        # EMPIRICAL: choice of the number of bins for numerical integration
        n_bins = int( np.log(N) * 1e2 )       
    else :
        try :
            n_bins = int(bins)
        except :
            raise TypeError("The parameter `bins` requires an integer value.")

    # >>>>>>>>>>>>>>>>>
    #  Compute Alpha  #
    # >>>>>>>>>>>>>>>>>

    # multiprocessing (WARNING:)
    POOL = multiprocessing.Pool( multiprocessing.cpu_count() )   
    args = [ ( implicit_S_vs_Alpha, x + 0.5, n_bins, K ) for x in range(n_bins) ]
    Alpha_vec = POOL.starmap( get_from_implicit, args )
    POOL.close()
    Alpha_vec = np.asarray( Alpha_vec )
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  estimators vs beta  #
    # >>>>>>>>>>>>>>>>>>>>>>
    
    POOL = multiprocessing.Pool( multiprocessing.cpu_count() ) 
    args = [ ( alpha, compACT, error ) for alpha in Alpha_vec ]
    results = POOL.starmap( estimate_S_at_alpha, args )
    POOL.close()
    results = np.asarray( results )
    
    # >>>>>>>>>>>>>>>
    #   estimators  #
    # >>>>>>>>>>>>>>>
    
    # NOTE: the normalization integral is computed on the same bins 
    #       which simplifies the bin size 
    
    Zeta = np.sum( results[:,0] )        

    integral_S1 = np.dot( results[:,0],  results[:,1] )
    S = mp.fdiv( integral_S1, Zeta )     

    if error is True :
        integral_S2 = np.dot( results[:,0],  results[:,2] )
        S2 = mp.fdiv( integral_S2, Zeta )
        S_devStd = np.sqrt( S2 - np.power(S,2) )
        shannon_estimate = np.array([S, S_devStd], dtype=np.float)   
        
    else :       
        shannon_estimate = np.array(S, dtype=np.float)        

    return shannon_estimate


##############
#  NOTATION  #
##############

def Delta_polyGmm(order, x, y):
    '''
    Difference between same `order` polygamma functions, computed in `x` and `y`.
    '''
    return polygamma(order,x) - polygamma(order,y)                                                     

def implicit_S_vs_Alpha( alpha, S, K ):
    '''
    implicit relation to be inverted.
    '''
    return Delta_polyGmm( 0, K * alpha + 1, alpha + 1 ) - S   

#################
#  _MEASURE MU  #
#################

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
    LogMu += np.dot( ff, LogGmm(nn+a) ) - LogGmm( N + K*a )   # posterior contribution

    return mp.exp( LogMu )

########################
#  get_from_implicit  #
########################

def get_from_implicit( implicit_relation, x, n_bins, K, maxiter=100 ):
    '''
    Numerical integration in (0, log(K)), domain of the explicit variable, 
    requires to solve an implicit relation to get the implicit variable  in(0, infty)
    ''' 

    # generalize to use also on H
    # bounds for the implicit variable
    lower_bound = 0
    # EMPIRICAL: right edge of the interval as approx of infty (WARNING:)
    upper_bound = np.log(K) * 1.e3    
    
    # EMPIRICAL: tolerance for brentq (WARNING:)
    xtol = 1.e-2  / ( K * n_bins )        
    # arguments of `implicit_relation` are explicit variable and categories       
    args = ( np.log(K) * x / n_bins , K )

    # brentq algorithm for implicit relation
    # NOTE : the implicit_realtion must have opposite signs in 0 and up_bound
    output = optimize.brentq( implicit_relation, lower_bound, upper_bound,
                             args=args, xtol=xtol, maxiter=maxiter )
    
    return output



######################################
#  S estimation vs Dirichelet param  #
######################################

def estimate_S_at_alpha( a, compACT, error ):
    '''
    It returns an array [ measureMu, entropy S and S^2 (if `error` is True) ] at the given `a` for `compACT`.
    '''
    
    # loading parameters from compACT        
    N, nn, ff, K = compACT.N, compACT.nn, compACT.ff, compACT.K
    
    mu_a = measureMu( a, compACT )
    
    # entropy computation
    temp = np.dot( ff, (nn+a) * Delta_polyGmm(0, N+K*a+1, nn+a+1) )     
    S1_a = mp.fdiv( temp, N + K*a )
    
    # compute squared entropy if error is required
    if error is False :
        output = np.array( [ mu_a, S1_a ] )

    else :    
        # term i != j
        S2_temp1 = np.zeros(len(ff))  
        for i in range( len(ff) ):   
            temp = (nn+a) * (nn[i]+a) * ( Delta_polyGmm(0, nn+a+1, N+K*a+2) * Delta_polyGmm(0, nn[i]+a+1, N+K*a+2) - polygamma(1, N+K*a+2) )
            S2_temp1[i] = np.dot( ff, temp )
        # correction # WARNING!: I could avoid this eliminating i-th term from nn and ff above
        correction = np.power(nn+a, 2) * ( np.power(Delta_polyGmm(0, nn+a+1, N+K*a+2), 2) - polygamma(1, N+K*a+2) )
        S2_temp1 = S2_temp1 - correction
        
        # term i == j
        S2_temp2 = (nn+a) * (nn+a+1) * ( np.power(Delta_polyGmm(0, nn+a+2, N+K*a+2), 2) + Delta_polyGmm(1, nn+a+2, N+K*a+2) ) 
        
        # total
        S2_a = mp.fdiv( np.dot( ff, S2_temp1 + S2_temp2 ), mp.fmul(N + K*a+1, N + K*a) )

        output = np.array( [ mu_a, S1_a, S2_a ] )

    return output
