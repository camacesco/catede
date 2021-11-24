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
    Alpha_vec = POOL.starmap( get_from_implicit, tqdm.tqdm(args, total=len(args), 
                                                           desc="Pre-computation 1/2", disable=disable) )
    Alpha_vec = np.asarray( Alpha_vec )
    
    args = [ (a, compACTexp ) for a in Alpha_vec ]
    measures = POOL.starmap( measureMu, tqdm.tqdm(args, total=len(args), desc='Pre-computations 2/2', disable=disable) )
    mu_a = np.asarray( measures )  
        
    # >>>>>>>>>>>>>>>>>>>>>>>
    #  estimators vs alpha  #
    # >>>>>>>>>>>>>>>>>>>>>>>
    
    args = [ ( alpha, compACTexp ) for alpha in Alpha_vec ]
    all_S1_a = POOL.starmap( estimate_S_at_alpha, tqdm.tqdm(args, total=len(args), desc="Evaluation", disable=disable) )
    all_S1_a = np.asarray(all_S1_a)
    
    if error is True :
        all_S2_a = POOL.starmap( estimate_S2_at_alpha, tqdm.tqdm(args, total=len(args), desc="Evaluation", disable=disable) )
        all_S2_a = np.asarray(all_S2_a)
    
    # multiprocessing (WARNING:)    
    POOL.close()
    
    # >>>>>>>>>>>>>>>
    #   estimators  #
    # >>>>>>>>>>>>>>>
    
    # NOTE: the normalization integral is computed on the same bins 
    #       which simplifies the bin size 
    
    Zeta = integral_with_mu( mu_a, 1, S_vec )

    integral_S1 = integral_with_mu(mu_a, all_S1_a, S_vec)
    S1 = mp.fdiv(integral_S1, Zeta)     

    if error is False :       
        shannon_estimate = np.array(S1, dtype=np.float) 
        
    else :
        S2 = mp.fdiv(integral_with_mu(mu_a, all_S2_a, S_vec), Zeta)
        S_devStd = np.sqrt(S2 - np.power(S1, 2))
        shannon_estimate = np.array([S1, S_devStd], dtype=np.float)   
        
    return shannon_estimate



######################################
#  S estimation vs Dirichelet param  #
######################################

def estimate_S_at_alpha( a, compACTexp ):
    '''
    It returns entropy S at the given `a` for `compACTexp`.
    '''
    
    # loading parameters from Experiment Compact        
    N, nn, ff, K = compACTexp.N, compACTexp.nn, compACTexp.ff, compACTexp.K
    
    # entropy computation
    temp = ff.dot( (nn+a) * D_polyGmm(0, N+K*a+1, nn+a+1) )     
    S1_a = mp.fdiv( temp, N+K*a )

    return S1_a

def estimate_S2_at_alpha( a, compACTexp ) :
    '''
    It returns squared entropy S2 at the given `a` for `compACTexp`.
    '''
    # loading parameters from Experiment.compACT exp       
    N, nn, ff, K = compACTexp.N, compACTexp.nn, compACTexp.ff, compACTexp.K
    
    # single sum term
    single_sum = np.power(D_polyGmm(0, nn+a+2, N+K*a+2), 2) + D_polyGmm(1, nn+a+2, N+K*a+2)
    Ss = (nn+a+1) * (nn+a) * single_sum
    
    # double sum term 
    double_sum = D_polyGmm(0, nn+a+1, N+K*a+2)[:,None] * D_polyGmm(0, nn+a+1, N+K*a+2) - polygamma(1, N+K*a+2)
    Ds = ( (nn+a)[:,None] * (nn+a) ) * double_sum
            
    output = ff.dot( Ss - Ds.diagonal() + Ds.dot(ff) )
    output = mp.fdiv( output, mp.fmul(N+K*a+1, N+K*a) ) 
    
    return output
