#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Nemenmann-Shafee-Bialek Estimator
    Copyright (C) February 2023 Francesco Camaglia, LPENS 

    ref: 
    Ilya Nemenman, F. Shafee, and William Bialek. Entropy and Inference, Revisited. 
    Advances in Neural Infor- mation Processing Systems, volume 14. MIT Press, 2001.
'''

import numpy as np
from mpmath import mp
import multiprocessing
from tqdm import tqdm

from .new_calculus import *
from .beta_func_multivar import *


def main(
    compExp, error=False, n_bins="default", 
    CPU_Count=None, verbose=False,
    ):
    ''' Nemenman-Shafee-Bialek estimator for Shannon entropy.'''
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>

    # number of categories #
    K = compExp.K
        
    # number of bins #
    if n_bins == "default" :
        # empirical choice ~
        n_bins = max( 1, np.round(5 * np.power(K / compExp.N, 2)) )
        n_bins = min( n_bins, 100 ) 
    try :
        n_bins = int(n_bins)
    except :
        raise TypeError("The parameter `bins` requires an integer value.")

    # number of jobs #
    try :
        CPU_Count = int(CPU_Count)
        if CPU_Count < 1 :
            raise TypeError("`CPU_Count` requires an integer greater than 0. Falling back to 1.") 
    except :
        CPU_Count = multiprocessing.cpu_count()
    CPU_Count = min( CPU_Count, n_bins**2 )

    # only saddle
    saddle_point_method = (n_bins < 2) 

    # verbose #
    disable = not verbose

    #  Find Point for Maximum Likelihood #  
    a_star = optimal_entropy_param( compExp ) 

    if saddle_point_method is True :

        # >>>>>>>>>>>>>>>>>>>>>>>
        #  SADDLE POINT METHOD  #
        # <<<<<<<<<<<<<<<<<<<<<<<

        S1_star = compExp.entropy(a_star) 
        if error is True :
            S2_star = compExp.squared_entropy(a_star)
            S_StdDev_star = np. sqrt(S2_star - np.power(S1_star, 2))  
            estimate = np.array([S1_star, S_StdDev_star], dtype=np.float64) 
        else :
            estimate = np.array([S1_star], dtype=np.float64) 

    else :

        # >>>>>>>>>>>>>>>>>>>>
        #  PRE COMPUTATIONS  #
        # <<<<<<<<<<<<<<<<<<<<

        hess_LogPosterior = Polya(a_star, compExp).log_hess() + DirEntr(a_star, K).logMetapr_hess()
        std_a = np.power( - hess_LogPosterior, -0.5 )
        alpha_vec = centered_logspaced_binning( a_star, std_a, n_bins )

        #  Compute Posterior (old ``Measure Mu``) for alpha #
        log_mu_alpha = list(map(lambda a : Polya(a, compExp).log(), alpha_vec ))   
        log_mu_alpha -= np.max( log_mu_alpha ) # regularization
        mu_a = np.exp( log_mu_alpha )

        # for uniform binning in prior expected entropy
        A_vec = DirEntr(alpha_vec, compExp.K).aPrioriExpec()
        
        # >>>>>>>>>>>>>>>>>>>>>>>
        #  estimators vs alpha  #
        # >>>>>>>>>>>>>>>>>>>>>>>

        POOL = multiprocessing.Pool( CPU_Count )  

        # entropy( a ) computation
        tqdm_args = tqdm( alpha_vec, total=len(alpha_vec), desc="Error Eval", disable=disable ) 
        all_S1_a = POOL.map( compExp.entropy, tqdm_args )
        all_S1_a = np.asarray(all_S1_a)
        
        # squared-entropy (a) computation
        if error is True :
            tqdm_args = tqdm( alpha_vec, total=len(alpha_vec), desc="Error Eval", disable=disable )
            all_S2_a = POOL.map( compExp.squared_entropy, tqdm_args )   
            all_S2_a = np.asarray(all_S2_a)
            
        POOL.close()

        # >>>>>>>>>>>>>>>
        #   estimators  #
        # >>>>>>>>>>>>>>>
            
        Zeta = np.trapz( mu_a, x=A_vec )
        S1 = mp.fdiv( np.trapz( np.multiply(mu_a, all_S1_a), x=A_vec ), Zeta ) 

        if error is False :       
            estimate = np.array(S1, dtype=np.float) 
        else :
            S2 = mp.fdiv( np.trapz( np.multiply( mu_a, all_S2_a ), x=A_vec ), Zeta ) 
            S_devStd = np.sqrt(S2 - np.power(S1, 2))
            estimate = np.array([S1, S_devStd], dtype=np.float)   
        
    return estimate
