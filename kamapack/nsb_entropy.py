#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Nemenmann-Shafee-Bialek Estimator
    Copyright (C) November 2021 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
from scipy import optimize
import multiprocessing
import tqdm

from ._nsb_aux_definitions import *

def NemenmanShafeeBialek( compACTexp, error=False, bins=1e4, CPU_Count=None, progressbar=False ):
    '''
    NSB entropy estimator description:
    '''

    K = compACTexp.K
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>
        
    try :
        n_bins = int(bins)
    except :
        raise TypeError("The parameter `bins` requires an integer value.")
        
    try :
        CPU_Count = int(CPU_Count)
        if CPU_Count < 1 :
            raise IOError("The parameter `CPU_Count` requires an integer value greater than 0.")     
    except :
        CPU_Count = multiprocessing.cpu_count()
        
    disable = not progressbar

    # >>>>>>>>>>>>>>>>>
    #  Compute Alpha  #
    # >>>>>>>>>>>>>>>>>

    # multiprocessing (WARNING:)
    POOL = multiprocessing.Pool( CPU_Count )   
    S_vec = np.linspace(0, np.log(K), n_bins)[1:-1]
    args = [ (implicit_S_vs_Alpha, S, 0, 1e15, K) for S in S_vec ]
    Alpha_vec = POOL.starmap( get_from_implicit, tqdm.tqdm(args, total=len(args), desc="Pre-computation", disable=disable) )
    POOL.close()
    Alpha_vec = np.asarray( Alpha_vec )
    
    # >>>>>>>>>>>>>>>>>>>>>>>
    #  estimators vs alpha  #
    # >>>>>>>>>>>>>>>>>>>>>>>
    
    POOL = multiprocessing.Pool( CPU_Count ) 
    args = [ ( alpha, compACTexp, error ) for alpha in Alpha_vec ]
    results = POOL.starmap( estimates_at_alpha, tqdm.tqdm(args, total=len(args), desc="Evaluation", disable=disable) )
    POOL.close()
    results = np.asarray(results)
    
    # >>>>>>>>>>>>>>>
    #   estimators  #
    # >>>>>>>>>>>>>>>
    
    # NOTE: the normalization integral is computed on the same bins 
    #       which simplifies the bin size 
    
    Zeta = integral_with_mu(results[:,0], 1, S_vec)

    integral_S1 = integral_with_mu(results[:,0], results[:,1], S_vec)
    S1 = mp.fdiv(integral_S1, Zeta)     

    if error is False :       
        shannon_estimate = np.array(S1, dtype=np.float) 
        
    else :
        S2 = mp.fdiv(integral_with_mu(results[:,0], results[:,2], S_vec), Zeta)
        S_devStd = np.sqrt(S2 - np.power(S1, 2))
        shannon_estimate = np.array([S1, S_devStd], dtype=np.float)   
        
    return shannon_estimate



######################################
#  S estimation vs Dirichelet param  #
######################################

def estimates_at_alpha( a, compACTexp, error ):
    '''
    It returns an array [ measureMu, entropy S and S^2 (if `error` is True) ] at the given `a` for `compACTexp`.
    '''
    
    # loading parameters from Experiment Compact        
    N, nn, ff, K = compACTexp.N, compACTexp.nn, compACTexp.ff, compACTexp.K
    
    mu_a = measureMu( a, compACTexp )
    
    # entropy computation
    temp = ff.dot( (nn+a) * D_polyGmm(0, N+K*a+1, nn+a+1) )     
    S1_a = mp.fdiv( temp, N+K*a )
    
    # compute squared entropy if error is required
    if error is False :
        output = np.array( [ mu_a, S1_a ] )
    else :    
        output = np.array( [ mu_a, S1_a, estimate_S2_at_alpha(a, compACTexp) ] )

    return output
