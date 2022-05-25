#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) April 2022 Francesco Camaglia, LPENS 
'''

import itertools
import numpy as np
from mpmath import mp
import multiprocessing

from typing import IO
import warnings
import tqdm
from ._wolpert_wolf_calculus import *

def Kullback_Leibler_CMW(
    compACTdiv, n_bins=2e1, cutoff_ratio=4, error=False,
    CPU_Count=None, verbose=False, equal_prior=False,
    ) :
    '''Kullback-Leibler divergence estimation with Camaglia Mora Walczak method.'''

    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>

    try :
        n_bins = int(n_bins)
    except :
        raise TypeError("The parameter `bins` requires an integer value.")

    try :
        cutoff_ratio = float( cutoff_ratio )
        if cutoff_ratio < 1. :
            if cutoff_ratio > 0. :
                warnings.warn("The parameter `cutoff_ratio` should be >=1.")
            else :
                raise IOError("The parameter `cutoff_ratio` must be >=1.")
    except :
        raise TypeError("The parameter `cutoff_ratio` requires a scalar value.")

    try :
        CPU_Count = int(CPU_Count)
        if CPU_Count < 1 :
            raise IOError("The parameter `CPU_Count` requires an integer value greater than 0.")     
    except :
        CPU_Count = multiprocessing.cpu_count()

    K = compACTdiv.K
    disable = not verbose 

    aux_range = np.append(
        np.logspace( -0.1, 0, np.floor(n_bins/2).astype(int) )[:-1],
        np.logspace( 0, 0.1, np.ceil(n_bins/2).astype(int) )
        )

    if equal_prior is True : # alpha == beta
        # FIXME :
        pass

        a_NSB_star = optimal_divergence_EP_param_( compACTdiv )
        alpha_vec = a_NSB_star * aux_range
        D_vec = ( 1. - 1./K ) / alpha_vec

        mu_alpha_1 = np.asarray(list(map( compACTdiv.compact_1._measureMu, alpha_vec )))  
        mu_alpha_1 /= np.max( mu_alpha_1 )
        mu_alpha_2 = np.asarray(list(map( compACTdiv.compact_2._measureMu, alpha_vec )))  
        mu_alpha_2 /= np.max( mu_alpha_2 )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>

        # multiprocessing (WARNING:)
        POOL = multiprocessing.Pool( CPU_Count )  
            
        args = [ x for x in zip(alpha_vec, alpha_vec) ]
        all_DKL_a = POOL.starmap(
            compACTdiv._post_divergence,
            tqdm.tqdm(args, total=len(args), desc='Evaluations', disable=disable)
            )
        all_DKL_a = np.asarray( all_DKL_a )
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL2 estimator vs alpha  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error is True :
            all_DKL2_a = POOL.starmap(
                compACTdiv._post_divergence_squared,
                tqdm.tqdm(args, total=len(args), desc='Squared', disable=disable)
                )
            all_DKL2_a = np.asarray( all_DKL2_a )
        
        POOL.close()

        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>

        Zeta = integral_with_mu_( mu_alpha_1, mu_alpha_2, D_vec )
        DKL1 = integral_with_mu_( mu_alpha_1, np.multiply(mu_alpha_2, all_DKL_a), D_vec ) 
        DKL1 = mp.fdiv( DKL1, Zeta )  
        if error is False :  
            kullback_leibler_estimate = np.array(DKL1, dtype=np.float) 
        else :
            DKL2 = mp.fdiv( integral_with_mu_(mu_alpha_1, np.multiply(mu_alpha_2, all_DKL2_a), D_vec ), Zeta )  
            DKL_devStd = np.sqrt(DKL2 - np.power(DKL1, 2))  
            kullback_leibler_estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
    
    else : # standard alpha != beta

        # >>>>>>>>>>>>>>>>>>>>
        #  PRE COMPUTATIONS  #
        # <<<<<<<<<<<<<<<<<<<<

        #  Compute Alpha and Beta #   

        # FIXME : this choice is arbitrary! works fine if the meta prior is negligible

        a_NSB_star = optimal_entropy_param_( compACTdiv.compact_1 )
        alpha_vec = a_NSB_star * aux_range
        A_vec = list(map( lambda a : implicit_entropy_vs_alpha_(a, 0, K), alpha_vec ) )

        b_NSB_star = optimal_crossentropy_param_( compACTdiv.compact_2 )
        beta_vec = b_NSB_star * aux_range
        B_vec = list(map( lambda b : implicit_crossentropy_vs_beta_(b, 0, K), beta_vec ) )

        #  Compute MeasureMu alpha and beta #
        mu_alpha = np.asarray(list(map( compACTdiv.compact_1._measureMu, alpha_vec )))  
        mu_alpha /= np.max( mu_alpha )
        mu_beta = np.asarray(list(map( compACTdiv.compact_2._measureMu, beta_vec )))  
        mu_beta /= np.max( mu_beta )
                    
        #  Compute MetaPrior_DKL   #
        args = [ x for x in itertools.product(A_vec, B_vec)]
        all_phi = list(map( lambda x : MetaPrior_DKL(x[0], x[1], K, cutoff_ratio), args ))
        all_phi = np.asarray( all_phi ).reshape(len(A_vec), len(B_vec))

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # multiprocessing (WARNING:)
        POOL = multiprocessing.Pool( CPU_Count )  
            
        # FIXME : choose args here only where MetaPrior_DKL > 0
        args = [ x for x in itertools.product(alpha_vec, beta_vec)]
        all_DKL_ab = POOL.starmap(
            compACTdiv._post_divergence,
            tqdm.tqdm(args, total=len(args), desc='Evaluations...', disable=disable)
            )
        all_DKL_ab = np.asarray( all_DKL_ab ).reshape(len(A_vec), len(B_vec))
        all_DKL_ab_times_phi = np.multiply( all_phi, all_DKL_ab )
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL2 estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error is True :
            # FIXME : choose args here only where all_DKL_ab > a certain threshold
            all_DKL2_ab = POOL.starmap(
                compACTdiv._post_divergence_squared,
                tqdm.tqdm(args, total=len(args), desc='Squared', disable=disable)
                )
            all_DKL2_ab = np.asarray( all_DKL2_ab ).reshape(len(A_vec), len(B_vec))
            all_DKL2_ab_times_phi = np.multiply( all_phi, all_DKL2_ab )

        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>
                    
        args = [ (mu_beta, x, B_vec) for x in all_phi ]
        args = args + [ (mu_beta, x, B_vec) for x in all_DKL_ab_times_phi ]
        if error is True : 
            args = args + [ (mu_beta, x, B_vec) for x in all_DKL2_ab_times_phi ]
            
        integrations_a = POOL.starmap(
            integral_with_mu_,
            tqdm.tqdm(args, total=len(args), desc='Final Integration', disable=disable)
            )
        integrations_a = np.asarray(  integrations_a )
                
        # multiprocessing (WARNING:)    
        POOL.close()
        
        Zeta = integral_with_mu_( mu_alpha, integrations_a[:len(A_vec)], A_vec )
        DKL1 = integral_with_mu_( mu_alpha, integrations_a[len(A_vec):2*len(A_vec)], A_vec ) 
        DKL1 = mp.fdiv( DKL1, Zeta )  
        
        if error is False :  
            kullback_leibler_estimate = np.array(DKL1, dtype=np.float) 
        else :
            DKL2 = mp.fdiv( integral_with_mu_(mu_alpha, integrations_a[2*len(A_vec):], A_vec ), Zeta )  
            DKL_devStd = np. sqrt(DKL2 - np.power(DKL1, 2))  
            kullback_leibler_estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
        
    return kullback_leibler_estimate
