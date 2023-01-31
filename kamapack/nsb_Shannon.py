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

from .new_calculus import *
from .beta_func_multivar import *


def Shannon_NSB(
    compExp, error=False, n_bins="default", CPU_Count=None, verbose=False
    ):
    ''' Nemenamn-Shafee-Bialekd Shannon entropy estimator '''
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>

    # number of categories
    K = compExp.K
        
    # number of bins
    if n_bins == "default" :
        # empirical choice ~
        n_bins = max( 1, np.round(5 * np.power(K / compExp.N, 2)) )
        n_bins = min( n_bins, 100 ) 
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

    run_parallel = ( CPU_Count > 1 ) # FIXME
    #  saddle_point_method = (n_bins < 2) # only saddle

    # verbose
    disable = not verbose

    # >>>>>>>>>>>>>>>>>
    #  Compute Alpha  #
    # >>>>>>>>>>>>>>>>>

    a_star = optimal_entropy_param_( compExp )
    hess_LogPosterior = Posterior(a_star, compExp).log_hess()

    std_a = np.power( - hess_LogPosterior, -0.5 )
    alpha_vec = np.append(
        np.logspace( min(BOUND_DIR[0], np.log10(a_star-N_SIGMA*std_a)), np.log10(a_star), n_bins//2 )[:-1],
        np.logspace( np.log10(a_star), np.log10(a_star+N_SIGMA*std_a), n_bins//2+1 )
    )

    #  Compute Posterior (old ``Measure Mu``) for alpha #
    log_mu_alpha = list(map(lambda a : Posterior(a, compExp).log(), alpha_vec ))   
    log_mu_alpha -= np.max( log_mu_alpha ) # regularization
    mu_a = np.exp( log_mu_alpha )

    # for uniform binning in prior expected entropy
    A_vec = prior_entropy_vs_alpha_(alpha_vec, compExp.K)
    
    # >>>>>>>>>>>>>>>>>>>>>>>
    #  estimators vs alpha  #
    # >>>>>>>>>>>>>>>>>>>>>>>

    args = alpha_vec

    # entropy( a ) computation
    if run_parallel is True :
        POOL = multiprocessing.Pool( CPU_Count )  
        tqdm_args = tqdm( args, total=len(args), desc="Error Eval", disable=disable ) 
        all_S1_a = POOL.map( compExp.entropy, tqdm_args )
    else :
        all_S1_a = [ compExp.entropy(args[0]) ]
    all_S1_a = np.asarray(all_S1_a)
    
    # squared-entropy (a) computation
    if error is True :
        if run_parallel is True :
            tqdm_args = tqdm( args, total=len(args), desc="Error Eval", disable=disable )
            all_S2_a = POOL.map( compExp.squared_entropy, tqdm_args )   
        else :
            all_S2_a = [ compExp.squared_entropy(args[0]) ]
        all_S2_a = np.asarray(all_S2_a)
        
    if run_parallel is True :
        POOL.close()

    # >>>>>>>>>>>>>>>
    #   estimators  #
    # >>>>>>>>>>>>>>>
        
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