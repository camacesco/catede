#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) October 2021 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
import multiprocessing
import itertools

from kamapack.nsb_entropy import *

def NSBDKL( compACT, bins=1e4, cutoff_ratio=5 ):
    '''
    '''

    K = compACT.K
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>
        
    try :
        n_bins = int(bins)
    except :
        raise TypeError("The parameter `bins` requires an integer value.")

    # >>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>
 
    S_vec = np.linspace(0, np.log(K), n_bins)[1:-1]
    H_cutoff = cutoff_ratio * np.log(K)
    H_vec = np.linspace(np.log(K), H_cutoff, n_bins)[1:-1]
    
    args = [ (implicit_S_vs_Alpha, S, 0, 1e15, K) for S in S_vec ]
    args = args + [ (implicit_H_vs_Beta, H, 1.e-15, 1e15, K) for H in H_vec ]
    
    POOL = multiprocessing.Pool( multiprocessing.cpu_count() )
    results = POOL.starmap( get_from_implicit, args )
    POOL.close()
    results = np.asarray( results )
    
    Alpha_vec = results[:len(S_vec)]
    Beta_vec = results[len(S_vec):]
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute MeasureMu Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    args = [ (a, compACT.compact_A ) for a in Alpha_vec ]
    args = args + [ (b, compACT.compact_B ) for b in Beta_vec ]
    
    POOL = multiprocessing.Pool( multiprocessing.cpu_count() )
    results = POOL.starmap( measureMu, args )
    POOL.close()
    results = np.asarray( results )  
    
    mu_a = results[:len(Alpha_vec)]
    mu_b = results[len(Alpha_vec):]
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  estimator vs alpha,beta  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # WARNING!: this itertool operation is a bottleneck
    args = [ x for x in itertools.product(zip(Alpha_vec,S_vec),zip(Beta_vec,H_vec))]
    args = [ x[0] + x[1] + (compACT,) for x in args]
    
    POOL = multiprocessing.Pool( multiprocessing.cpu_count() ) 
    results = POOL.starmap( estimate_DKL_at_alpha_beta, args )
    POOL.close()
    results = np.asarray(results)
    
    # >>>>>>>>>>>>>>>>>
    #   integrations  #
    # >>>>>>>>>>>>>>>>>

    temp = list(map(lambda i: np.trapz(np.multiply(results[:,0][i:i+len(Beta_vec)], mu_b)), np.arange(len(Alpha_vec)) ))
    zeta = np.trapz(np.multiply(temp, mu_a))
    temp = list(map(lambda i: np.trapz(np.multiply(results[:,1][i:i+len(Beta_vec)], mu_b)), np.arange(len(Alpha_vec)) ))
    estimate = mp.fdiv(np.trapz(np.multiply(temp, mu_a)), zeta)  

    return estimate

##############
#  NOTATION  #
##############

def implicit_H_vs_Beta( beta, x, K ):
    '''
    implicit relation to be inverted.
    '''
    return D_polyGmm( 0, K * beta , beta ) - x

###########
#  SIGMA  #
###########

def Sigma( H, S, K ) :
    z = H - S
    if z >= np.log(K) :
        return 1. / np.log(K)
    else :
        return 1. / z 
    
######################################
#  S estimation vs Dirichelet param  #
######################################

def estimate_DKL_at_alpha_beta( a, S, b, H, compACT ):
    '''
    It returns an array [ measureMu, entropy S and S^2 (if `error` is True) ] at the given `a` for `compACT`.
    '''
    
    # loading parameters from compACT        
    N_A, N_B = compACT.N_A, compACT.N_B
    nn_A, nn_B, ff = compACT.nn_A, compACT.nn_B, compACT.ff
    K = compACT.K
    
    sigma = Sigma( H, S, K )
    # DKL computation
    temp = np.dot( ff, (nn_A+a) * ( D_polyGmm(0, N_B+K*b, nn_B+b) - D_polyGmm(0, N_A+K*a+1, nn_A+a+1) ) )
    output = mp.fdiv( sigma * temp, N_A + K*a )  
    
    return np.array([sigma, output])
