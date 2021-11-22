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

from ._nsb_aux_definitions import *

def CamagliaMoraWalczak( compACTdiv, bins=1e3, cutoff_ratio=5, error=False, progressbar=False ):
    '''
    CMW Kullback-Leibler divergence estimator description:
    '''

    K = compACTdiv.K
    CPU_Count = multiprocessing.cpu_count()
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>
        
    try :
        n_bins = int(bins)
    except :
        raise TypeError("The parameter `bins` requires an integer value.")
        
    disable = not progressbar 
        
    # >>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>
    
    S_vec = np.linspace(0, np.log(K), n_bins)[1:-1]
    H_cutoff = cutoff_ratio * np.log(K)
    H_vec = np.linspace(np.log(K), H_cutoff, n_bins)[1:-1]
    
    args = [ (implicit_S_vs_Alpha, S, 0, 1e20, K) for S in S_vec ]
    args = args + [ (implicit_H_vs_Beta, H, 1.e-20, 1e20, K) for H in H_vec ]
    
    POOL = multiprocessing.Pool( CPU_Count )
    results = POOL.starmap( get_from_implicit, tqdm.tqdm(args, total=len(args), desc='Pre-computations 1/2', disable=disable) )
    POOL.close()
    results = np.asarray( results )
    
    Alpha_vec = results[:len(S_vec)]
    Beta_vec = results[len(S_vec):]
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute MeasureMu Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    args = [ (a, compACTdiv.compact_A ) for a in Alpha_vec ]
    args = args + [ (b, compACTdiv.compact_B ) for b in Beta_vec ]
    
    POOL = multiprocessing.Pool( CPU_Count )
    results = POOL.starmap( measureMu, tqdm.tqdm(args, total=len(args), desc='Pre-computations 2/2', disable=disable) )
    POOL.close()
    results = np.asarray( results )  
    
    mu_a = results[:len(Alpha_vec)]
    mu_b = results[len(Alpha_vec):]
        
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  estimator vs alpha,beta  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    # WARNING!: this itertools operation is a bottleneck!
    args = [ x[0] + x[1] + (compACTdiv, error,) for x in itertools.product(zip(Alpha_vec,S_vec),zip(Beta_vec,H_vec))]
            
    POOL = multiprocessing.Pool( CPU_Count ) 
    results = POOL.starmap( estimates_at_alpha_beta, tqdm.tqdm(args, total=len(args), desc='Grid Evaluations', disable=disable) )
    POOL.close()
    results = np.asarray(results)
    
    # >>>>>>>>>>>>>>>>>
    #   integrations  #
    # >>>>>>>>>>>>>>>>>
    
    # WARNING!: these operations are a bottleneck!
    all_sigma = results[:,0].reshape(len(Alpha_vec), len(Beta_vec))
    all_DKL_ab_times_sigma = results[:,1].reshape(len(Alpha_vec), len(Beta_vec))
            
    args = [ (mu_b, x, H_vec) for x in all_sigma ]
    args = args + [ (mu_b, x, H_vec) for x in all_DKL_ab_times_sigma ]
    
    if error is True :    
        all_DKL2_ab_times_sigma = results[:,2].reshape(len(Alpha_vec), len(Beta_vec))
        args = args + [ (mu_b, x, H_vec) for x in all_DKL2_ab_times_sigma ]
        
    POOL = multiprocessing.Pool( CPU_Count ) 
    results = POOL.starmap( integral_with_mu, tqdm.tqdm(args, total=len(args), desc='Final Integrations', disable=disable) )
    POOL.close()
    results = np.asarray(results)
    
    Zeta = integral_with_mu(mu_a, results[:len(Alpha_vec)], S_vec)
    DKL1 = mp.fdiv(integral_with_mu(mu_a, results[len(Alpha_vec):2*len(Alpha_vec)], S_vec), Zeta)  
    
    if error is False :       
        kullback_leibler_estimate = np.array(DKL1, dtype=np.float) 
        
    else :
        DKL2 = mp.fdiv(integral_with_mu(mu_a, results[2*len(Alpha_vec):], S_vec), Zeta)  
        DKL_devStd = np.sqrt(DKL2 - np.power(DKL1, 2))  
        kullback_leibler_estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
    
    return kullback_leibler_estimate


###########
#  SIGMA  #
###########

def Sigma( H, S, K ) :
    z = H - S
    if z >= np.log(K) :
        return 1. / np.log(K)
    else :
        return 1. / z 
    
#########################################
#  DKL estimation vs Dirichelet params  #
#########################################

def estimates_at_alpha_beta( a, S, b, H, compACTdiv, error ):
    '''
    It returns an array [ measureMu, D_KL and D_KL^2 (if `error` is True) ] at the given `a` for `compACTdiv`.
    '''
    
    # loading parameters from Divergence Compact        
    N_A, N_B = compACTdiv.N_A, compACTdiv.N_B
    nn_A, nn_B, ff = compACTdiv.nn_A, compACTdiv.nn_B, compACTdiv.ff
    K = compACTdiv.K
    
    sigma = Sigma( H, S, K )
    # DKL computation
    temp = ff.dot( (nn_A+a) * ( D_polyGmm(0, N_B+K*b, nn_B+b) - D_polyGmm(0, N_A+K*a+1, nn_A+a+1) ) ) 
    DKL_ab_times_sigma = sigma * mp.fdiv( temp, N_A+K*a )    
        
        
    # compute squared entropy if error is required
    if error is False :
        output = np.array( [ sigma, DKL_ab_times_sigma ] )
    else :    
        DKL2_ab_times_sigma = sigma * estimate_DKL2_at_alpha_beta(a, b, compACTdiv)
        output = np.array( [ sigma, DKL_ab_times_sigma, DKL2_ab_times_sigma ] )
    
    return output

    