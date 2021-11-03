#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) November 2021 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
import multiprocessing
import itertools
import tqdm

from kamapack.nsb_entropy import D_polyGmm, implicit_S_vs_Alpha, LogGmm, measureMu, get_from_implicit

def NemenmanShafeeBialek( compACT, bins=1e3, cutoff_ratio=5 ):
    '''
    '''

    K = compACT.K
    CPU_Count = multiprocessing.cpu_count()
    
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
    
    args = [ (implicit_S_vs_Alpha, S, 0, 1e20, K) for S in S_vec ]
    args = args + [ (implicit_H_vs_Beta, H, 1.e-20, 1e20, K) for H in H_vec ]
    
    POOL = multiprocessing.Pool( CPU_Count )
    results = POOL.starmap( get_from_implicit, tqdm.tqdm(args, total=len(args), desc='Pre-computations 1/2') )
    POOL.close()
    results = np.asarray( results )
    
    Alpha_vec = results[:len(S_vec)]
    Beta_vec = results[len(S_vec):]
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute MeasureMu Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    args = [ (a, compACT.compact_A ) for a in Alpha_vec ]
    args = args + [ (b, compACT.compact_B ) for b in Beta_vec ]
    
    POOL = multiprocessing.Pool( CPU_Count )
    results = POOL.starmap( measureMu, tqdm.tqdm(args, total=len(args), desc='Pre-computations 2/2') )
    POOL.close()
    results = np.asarray( results )  
    
    mu_a = results[:len(Alpha_vec)]
    mu_b = results[len(Alpha_vec):]
        
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  estimator vs alpha,beta  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    # WARNING!: this itertools operation is a bottleneck!
    args = [ x[0] + x[1] + (compACT,) for x in itertools.product(zip(Alpha_vec,S_vec),zip(Beta_vec,H_vec))]
            
    POOL = multiprocessing.Pool( CPU_Count ) 
    results = POOL.starmap( estimate_DKL_at_alpha_beta, tqdm.tqdm(args, total=len(args), desc='Grid Evaluations') )
    POOL.close()
    results = np.asarray(results)
    
    # >>>>>>>>>>>>>>>>>
    #   integrations  #
    # >>>>>>>>>>>>>>>>>
    
    args = [ (x, mu_b, H_vec) for x in results[:,0].reshape(len(Alpha_vec), len(Beta_vec)) ]
    args = args + [ (x, mu_b, H_vec) for x in results[:,1].reshape(len(Alpha_vec), len(Beta_vec)) ]
    
    POOL = multiprocessing.Pool( CPU_Count ) 
    results = POOL.starmap( integral_with_mu, tqdm.tqdm(args, total=len(args), desc='Final Integration') )
    POOL.close()
    results = np.asarray(results)
    
    zeta = integral_with_mu(results[:len(Alpha_vec)], mu_a, S_vec)
    estimate = mp.fdiv(integral_with_mu(results[len(Alpha_vec):], mu_a, S_vec), zeta)  
    
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
#     #
######################################

def estimate_DKL_at_alpha_beta( a, S, b, H, compACT ):
    '''
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

#########################
#   integral with mu  #
############################

def integral_with_mu( func_ab, mu, x ) :
    
    return np.trapz(np.multiply(func_ab, mu), x=x)
    