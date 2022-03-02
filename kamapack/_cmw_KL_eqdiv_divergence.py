#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) February 2022 Francesco Camaglia, LPENS 
'''

import warnings
import numpy as np
from mpmath import mp
import multiprocessing
import tqdm

from ._aux_definitions import *
from .cmw_KL_divergence import estimate_DKL_at_alpha_beta, estimate_DKL2_at_alpha_beta

def Kullback_Leibler_CMW_eqdiv( compACTdiv, n_bins=5e2, cutoff_ratio=5, error=False, CPU_Count=None, prior="unif", verbose=False ):
    '''Kullback-Leibler divergence estimation with CMW method in the case of equal diversity assumption.
    '''

    K = compACTdiv.K            
    disable = not verbose 
        
    # >>>>>>>>>>>>>>>>
    #  Compute Alpha #
    # >>>>>>>>>>>>>>>>
    
    D_cutoff = cutoff_ratio * np.log(K)

    # linear spacing
    D_vec = np.linspace(0, D_cutoff, n_bins+2)[1:-1]

    # logarithmic spacing
    D_vec = np.logspace(0, np.log10( D_cutoff + 1 ), n_bins+2) - 1
    D_vec = D_vec[1:-1]
    
    Alpha_vec = ( 1. - 1./K ) / D_vec
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute MeasureMu Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # multiprocessing (WARNING:)
    POOL = multiprocessing.Pool( CPU_Count )  
    
    args = [ (a, compACTdiv.compact_1 ) for a in Alpha_vec ]
    args = args + [ (a, compACTdiv.compact_2 ) for a in Alpha_vec ]
    
    measures = POOL.starmap( measureMu, tqdm.tqdm(args, total=len(args), desc='Pre-computations', disable=disable) )
    measures = np.asarray( measures )  
    
    mu_alpha_1 = measures[:n_bins]
    mu_alpha_2 = measures[n_bins:]

    # regualarization 
    mu_alpha_1 /= np.max( mu_alpha_1 )
    mu_alpha_2 /= np.max( mu_alpha_2 )

    # prior choice
    if prior == "unif" :
        pass
    elif prior == "log-unif" :
        mu_alpha_2 = mu_alpha_2 / D_vec
    elif prior == "linear" :
        mu_alpha_2 = mu_alpha_2 * D_vec
    else :
        pass
                    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>
    #  DKL estimator vs alpha  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>
        
    args = [ x + (compACTdiv,) for x in zip(Alpha_vec, Alpha_vec) ]
    all_DKL_a = POOL.starmap( estimate_DKL_at_alpha_beta, tqdm.tqdm(args, total=len(args), 
                                                                     desc='Evaluations', disable=disable) )
    all_DKL_a = np.asarray( all_DKL_a )
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  DKL2 estimator vs alpha  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    if error is True :
        all_DKL2_a = POOL.starmap( estimate_DKL2_at_alpha_beta, tqdm.tqdm(args, total=len(args), 
                                                                         desc='Error evaluations', disable=disable) )
        all_DKL2_a = np.asarray( all_DKL2_a )
    
    POOL.close()

    # >>>>>>>>>>>>>>>>>
    #   integrations  #
    # >>>>>>>>>>>>>>>>>

    Zeta = integral_with_mu( mu_alpha_1, mu_alpha_2, D_vec )
    DKL1 = integral_with_mu( mu_alpha_1, np.multiply(mu_alpha_2, all_DKL_a), D_vec ) 
    DKL1 = mp.fdiv( DKL1, Zeta )  
    if error is False :  
        kullback_leibler_ed_estimate = np.array(DKL1, dtype=np.float) 
    else :
        DKL2 = mp.fdiv( integral_with_mu(mu_alpha_1, np.multiply(mu_alpha_2, all_DKL2_a), D_vec ), Zeta )  
        DKL_devStd = np.sqrt(DKL2 - np.power(DKL1, 2))  
        kullback_leibler_ed_estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
    
    return kullback_leibler_ed_estimate