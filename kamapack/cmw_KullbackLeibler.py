#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) November 2022 Francesco Camaglia, LPENS 
'''

import itertools
import numpy as np
from mpmath import mp
import multiprocessing
from tqdm import tqdm
from .new_calculus import *
from .nsb_Shannon import integral_with_mu_

def Kullback_Leibler_CMW(
    CompDiv, n_bins=1, cutoff_ratio=4, error=False,
    CPU_Count=None, verbose=False,
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
    if n_bins == 1 : saddle_point_method = True
    saddle_point_method = True # FLAG : FIXME developer

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

    # standard alpha != beta
    equal_prior = False # FLAG : FIXME developer
    


    #  Find Point for Maximum Likelihood #   
    a_NSB_star, b_NSB_star = optimal_divergence_params_( CompDiv )

    if saddle_point_method is True :

        DKL1_star = CompDiv.divergence(a_NSB_star, b_NSB_star)
        if error is True :
            DKL2_star = CompDiv.squared_divergence(a_NSB_star, b_NSB_star)
            DKL_StdDev_star = np. sqrt(DKL2_star - np.power(DKL1_star, 2))  
            estimate = np.array([DKL1_star, DKL_StdDev_star], dtype=np.float64) 
        else :
            estimate = np.array([DKL1_star], dtype=np.float64) 
    else :    
        # >>>>>>>>>>>>>>>>>>>>
        #  PRE COMPUTATIONS  #
        # <<<<<<<<<<<<<<<<<<<<

        # FIXME : to be optimized; also, the parallelization seems to be weak

        aux_range = np.append(
            np.logspace( -0.25, 0, np.ceil( n_bins / 2 + 0.5 ).astype(int) )[:-1],
            np.logspace( 0, 0.25, np.floor( n_bins / 2 + 0.5 ).astype(int) )
        )
        #  Compute MeasureMu alpha and beta #
        alpha_vec = a_NSB_star * aux_range
        beta_vec = b_NSB_star * aux_range
        mu_alpha = np.asarray(list(map( CompDiv.compact_1.alphaLikelihood, alpha_vec )))  
        mu_alpha /= np.max( mu_alpha )
        mu_beta = np.asarray(list(map( CompDiv.compact_2.alphaLikelihood, beta_vec )))  
        mu_beta /= np.max( mu_beta )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        args = [ x for x in itertools.product(alpha_vec, beta_vec)]
        if run_parallel is True : # FIXME
            POOL = multiprocessing.Pool( CPU_Count ) 
            tqdm_args = tqdm(args, total=len(args), desc='Evaluations...', disable=disable)
            all_DKL_ab = POOL.starmap( CompDiv.divergence, tqdm_args )
        else :
            all_DKL_ab = [ CompDiv.divergence( args[0] ) ]
        all_DKL_ab = np.asarray( all_DKL_ab ).reshape(len(alpha_vec), len(beta_vec))
    
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL2 estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error is True :
            if run_parallel is True :
                tqdm_args = tqdm(args, total=len(args), desc='Squared', disable=disable)
                all_DKL2_ab = POOL.starmap( CompDiv.squared_divergence, tqdm_args )
            else :
                all_DKL2_ab = [ CompDiv.squared_divergence( args[0] ) ]
            all_DKL2_ab = np.asarray( all_DKL2_ab ).reshape(len(alpha_vec), len(beta_vec))
        
        if run_parallel is True :
            POOL.close()

        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>

        K = CompDiv.K
        A_vec = prior_entropy_vs_alpha_( alpha_vec, K ) 
        B_vec = prior_crossentropy_vs_beta_( beta_vec, K ) 
        # FIXME

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