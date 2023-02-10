#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Nemenmann-Shafee-Bialek Estimator
    Copyright (C) August 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
import multiprocessing
from tqdm import tqdm

from .new_calculus import optimal_simpson_param_
from .beta_func_multivar import *

def Simpson_NSB(
    compACTexp, error=False, n_bins=1, CPU_Count=None, verbose=False
    ):
    ''' Nemenamn-Shafee-Bialekd Simpson index estimator '''
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>
        
    # number of bins
    try :
        n_bins = int(n_bins)
    except :
        raise TypeError("The parameter `bins` requires an integer value.")
    
    # number of jobs
    try :
        CPU_Count = int(CPU_Count)
        if CPU_Count < 1 :
            raise TypeError("`CPU_Count` requires an integer greater than 0. Falling back to 1.") 
    except :
        CPU_Count = multiprocessing.cpu_count()
    CPU_Count = min( CPU_Count, n_bins**2 )

    run_parallel = ( CPU_Count == 1 )

    # verbose
    disable = not verbose

    # >>>>>>>>>>>>>>>>>
    #  Compute Alpha  #
    # >>>>>>>>>>>>>>>>>

    a_NSB_star = optimal_simpson_param_( compACTexp )
    aux_range = np.append(
        np.logspace( -0.25, 0, np.ceil( n_bins / 2 + 0.5 ).astype(int) )[:-1],
        np.logspace( 0, 0.25, np.floor( n_bins / 2 + 0.5 ).astype(int) )
        )
    alpha_vec = a_NSB_star * aux_range
            
    # >>>>>>>>>>>>>>>>>>>>>>>
    #  estimators vs alpha  #
    # >>>>>>>>>>>>>>>>>>>>>>>

    args = alpha_vec

    # entropy( a ) computation
    if run_parallel is True :
        POOL = multiprocessing.Pool( CPU_Count )  
        tqdm_args = tqdm( args, total=len(args), desc="Error Eval", disable=disable ) 
        all_S1_a = POOL.map( compACTexp.simpson, tqdm_args )
    else :
        all_S1_a = [ compACTexp.simpson(args[0]) ]
    all_S1_a = np.asarray(all_S1_a)
    
    # squared-entropy (a) computation
    if error is True :
        if run_parallel is True :
            tqdm_args = tqdm( args, total=len(args), desc="Error Eval", disable=disable )
            all_S2_a = POOL.map( compACTexp.squared_simpson, tqdm_args )   
        else :
            all_S2_a = [ compACTexp.squared_simpson(args[0]) ]
        all_S2_a = np.asarray(all_S2_a)
        
    if run_parallel is True :
        POOL.close()

    # >>>>>>>>>>>>>>>
    #   estimators  #
    # >>>>>>>>>>>>>>>

    K = compACTexp.K
    A_vec = prior_simpson_vs_alpha_( alpha_vec, K ) 
    mu_a = np.asarray( list(map( compACTexp.alphaLikelihood,  alpha_vec )) )
        
    Zeta = integral_with_mu_(mu_a, 1., A_vec)
    S1 = mp.fdiv( integral_with_mu_(mu_a, all_S1_a, A_vec), Zeta ) 
    if error is False :       
        estimate = np.array(S1, dtype=np.float) 
    else :
        S2 = mp.fdiv( integral_with_mu_(mu_a, all_S2_a, A_vec), Zeta )
        S_devStd = np.sqrt(S2 - np.power(S1, 2))
        estimate = np.array([S1, S_devStd], dtype=np.float)   
        
    return estimate

def integral_with_mu_( m, y, x ) :
    # alias for trapz integration
    if len(x) == 1 : output = np.multiply(m, y)[0]
    else : output = np.trapz( np.multiply(m, y), x=x )
    return output