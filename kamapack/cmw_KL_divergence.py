#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) August 2022 Francesco Camaglia, LPENS 
'''

import itertools
import numpy as np
from mpmath import mp
import multiprocessing
from tqdm import tqdm
from .new_calculus import *
from .nsb_entropy import integral_with_mu_

def Kullback_Leibler_CMW(
    compACTdiv, n_bins=1, cutoff_ratio=4, error=False,
    CPU_Count=None, verbose=False, equal_prior=False,
    ) :
    '''Kullback-Leibler divergence estimation with Camaglia Mora Walczak method.'''

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

    aux_range = np.append(
        np.logspace( -0.25, 0, np.ceil( n_bins / 2 + 0.5 ).astype(int) )[:-1],
        np.logspace( 0, 0.25, np.floor( n_bins / 2 + 0.5 ).astype(int) )
        )

    # Equal Dirichlet Priors
    if equal_prior is True : # alpha == beta

        a_NSB_star = optimal_divergence_EP_param_( compACTdiv )
        alpha_vec = a_NSB_star * aux_range
        
        mu_alpha_1 = np.asarray(list(map( compACTdiv.compact_1.alphaLikelihood, alpha_vec )))  
        mu_alpha_1 /= np.max( mu_alpha_1 )
        mu_alpha_2 = np.asarray(list(map( compACTdiv.compact_2.alphaLikelihood, alpha_vec )))  
        mu_alpha_2 /= np.max( mu_alpha_2 )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>
            
        args = [ x for x in zip(alpha_vec, alpha_vec) ]
        if run_parallel is True :
            POOL = multiprocessing.Pool( CPU_Count ) 
            tqdm_args = tqdm(args, total=len(args), desc='Evaluations', disable=disable)
            all_DKL_a = POOL.starmap( compACTdiv.divergence, tqdm_args )
        else :
            all_DKL_a = [ compACTdiv.divergence( args[0] ) ]
        all_DKL_a = np.asarray( all_DKL_a )
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL2 estimator vs alpha  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error is True :
            if run_parallel is True :
                tqdm_args = tqdm(args, total=len(args), desc='Squared', disable=disable)
                all_DKL2_a = POOL.starmap( compACTdiv.squared_divergence, tqdm_args )
            else :
                all_DKL2_a = [ compACTdiv.squared_divergence( args[0] ) ]
            all_DKL2_a = np.asarray( all_DKL2_a )
    
        if run_parallel is True :
            POOL.close()

        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>

        D_vec = ( 1. - 1. / compACTdiv.K ) / alpha_vec
        mu_alpha_1_2 = np.multiply( mu_alpha_1, mu_alpha_2 )

        Zeta = np.trapz( mu_alpha_1_2, x=D_vec )
        DKL1 = mp.fdiv( integral_with_mu_( mu_alpha_1_2, all_DKL_a, D_vec ) , Zeta )  
        if error is False :  
            estimate = np.array(DKL1, dtype=np.float) 
        else :
            DKL2 = mp.fdiv( integral_with_mu_( mu_alpha_1_2, all_DKL2_a, D_vec ), Zeta )  
            DKL_devStd = np.sqrt(DKL2 - np.power(DKL1, 2))  
            estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
    
    elif equal_prior is False : # standard alpha != beta

        # >>>>>>>>>>>>>>>>>>>>
        #  PRE COMPUTATIONS  #
        # <<<<<<<<<<<<<<<<<<<<

        #  Compute Alpha and Beta #   
        a_NSB_star = optimal_entropy_param_( compACTdiv.compact_1 )
        alpha_vec = a_NSB_star * aux_range
        b_NSB_star = optimal_crossentropy_param_( compACTdiv.compact_2 )
        beta_vec = b_NSB_star * aux_range
        # FIXME : this choice is arbitrary! works fine if the meta prior is negligible

        #  Compute MeasureMu alpha and beta #
        mu_alpha = np.asarray(list(map( compACTdiv.compact_1.alphaLikelihood, alpha_vec )))  
        mu_alpha /= np.max( mu_alpha )
        mu_beta = np.asarray(list(map( compACTdiv.compact_2.alphaLikelihood, beta_vec )))  
        mu_beta /= np.max( mu_beta )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        args = [ x for x in itertools.product(alpha_vec, beta_vec)]
        if run_parallel is True :
            POOL = multiprocessing.Pool( CPU_Count ) 
            tqdm_args = tqdm(args, total=len(args), desc='Evaluations...', disable=disable)
            all_DKL_ab = POOL.starmap( compACTdiv.divergence, tqdm_args )
        else :
            all_DKL_ab = [ compACTdiv.divergence( args[0] ) ]
        all_DKL_ab = np.asarray( all_DKL_ab ).reshape(len(alpha_vec), len(beta_vec))
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL2 estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error is True :
            if run_parallel is True :
                tqdm_args = tqdm(args, total=len(args), desc='Squared', disable=disable)
                all_DKL2_ab = POOL.starmap( compACTdiv.squared_divergence, tqdm_args )
            else :
                all_DKL2_ab = [ compACTdiv.squared_divergence( args[0] ) ]
            all_DKL2_ab = np.asarray( all_DKL2_ab ).reshape(len(alpha_vec), len(beta_vec))
        
        if run_parallel is True :
            POOL.close()

        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>

        K = compACTdiv.K
        A_vec = prior_entropy_vs_alpha_( alpha_vec, K ) 
        B_vec = prior_crossentropy_vs_beta_( beta_vec, K ) 

        #  Compute MetaPrior_DKL   #
        args = [ x for x in itertools.product(A_vec, B_vec) ]
        all_phi = list(map( lambda x : MetaPrior_DKL(x[0], x[1], K, cutoff_ratio), args ))
        all_phi = np.asarray( all_phi ).reshape(len(alpha_vec), len(beta_vec))
        all_DKL_ab_times_phi = np.multiply( all_phi, all_DKL_ab )
        args = np.concatenate([all_phi, all_DKL_ab_times_phi])
        if error is True :
            all_DKL2_ab_times_phi = np.multiply( all_phi, all_DKL2_ab )
            args = np.concatenate([args, all_DKL2_ab_times_phi])

        integrations_a = list(map( lambda x : integral_with_mu_( mu_beta, x, B_vec ), args ))
        integrations_a = np.asarray(  integrations_a )
        Zeta = integral_with_mu_( mu_alpha, integrations_a[:len(A_vec)], A_vec )
        DKL1 = integral_with_mu_( mu_alpha, integrations_a[len(A_vec):2*len(A_vec)], A_vec ) 
        DKL1 = mp.fdiv( DKL1, Zeta )  
        
        if error is False :  
            estimate = np.array(DKL1, dtype=np.float) 
        else :
            DKL2 = mp.fdiv( integral_with_mu_(mu_alpha, integrations_a[2*len(A_vec):], A_vec ), Zeta )  
            DKL_devStd = np. sqrt(DKL2 - np.power(DKL1, 2))  
            estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
        
    return estimate