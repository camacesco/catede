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
    compACTdiv, n_bins=5e1, cutoff_ratio=4, error=False,
    CPU_Count=None, verbose=False, equal_diversity=False,
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

    # FIXME :
    if equal_diversity is True : # alpha = beta
        from ._cmw_KL_eqdiv_divergence import Kullback_Leibler_CMW_eqdiv
        return Kullback_Leibler_CMW_eqdiv(
            compACTdiv, n_bins=n_bins, cutoff_ratio=cutoff_ratio, error=error,
            CPU_Count=CPU_Count, verbose=verbose
            )

    else : # standard alpha != beta

        K = compACTdiv.K
        disable = not verbose 
            
        # >>>>>>>>>>>>>>>>>>>>>>>>>
        #  Compute Alpha and Beta #
        # >>>>>>>>>>>>>>>>>>>>>>>>>       

        # FIXME : this choice makes sense ny if the meta prior is negligible

        aux_range = np.append(
            np.logspace( -0.01, 0, np.floor(n_bins/2).astype(int) ),
            np.logspace( 0, 0.01, np.ceil(n_bins/2).astype(int) )[1:]
            )

        a_NSB_star = optimal_entropy_param_( compACTdiv.compact_1, upper=1e-5, lower=1e3 )
        alpha_vec = a_NSB_star * aux_range
        A_vec = list(map( lambda a : implicit_entropy_vs_alpha_(a, 0, K), alpha_vec ) )

        b_NSB_star = optimal_crossentropy_param_( compACTdiv.compact_2, upper=1e-5, lower=1e3 )
        beta_vec = b_NSB_star * aux_range
        B_vec = list(map( lambda b : implicit_crossentropy_vs_beta_(b, 0, K), beta_vec ) )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  Compute MeasureMu alpha and beta #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        # multiprocessing (WARNING:)
        POOL = multiprocessing.Pool( CPU_Count )  
        
        args = alpha_vec     
        measures = POOL.map(
            compACTdiv.compact_1._measureMu,
            tqdm.tqdm(args, total=len(args), desc='Pre-computations 1/3', disable=disable)
            )
        mu_alpha = np.asarray( measures )  
        mu_alpha /= np.max( mu_alpha )

        args = beta_vec     
        measures = POOL.map(
            compACTdiv.compact_2._measureMu,
            tqdm.tqdm(args, total=len(args), desc='Pre-computations 2/3', disable=disable)
            )
        mu_beta = np.asarray( measures )  
        mu_beta /= np.max( mu_beta )
                    
        # >>>>>>>>>>>>>>>>
        #  Compute MetaPrior_DKL   #
        # >>>>>>>>>>>>>>>>
            
        args = [ x + (compACTdiv.K, cutoff_ratio,) for x in itertools.product(A_vec, B_vec)]
        all_phi = POOL.starmap(
            MetaPrior_DKL,
            tqdm.tqdm(args, total=len(args), desc='Pre-computations 3/3', disable=disable)
            )
        all_phi = np.asarray( all_phi ).reshape(len(A_vec), len(B_vec))

        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        # FIXME : choose args here only where MetaPrior_DKL > 0
        args = [ x for x in itertools.product(alpha_vec, beta_vec)]
        all_DKL_ab = POOL.starmap(
            compACTdiv._post_divergence,
            tqdm.tqdm(args, total=len(args), desc='Evaluations', disable=disable)
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
                tqdm.tqdm(args, total=len(args), desc='Error evaluations', disable=disable)
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
            tqdm.tqdm(args, total=len(args), desc='Integration', disable=disable)
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
