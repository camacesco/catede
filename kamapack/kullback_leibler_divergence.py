#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) February 2023 Francesco Camaglia, LPENS 
'''

import itertools
import numpy as np
from mpmath import mp
import multiprocessing
from tqdm import tqdm
from .new_calculus import *
from .nsb_shannon_entropy import integral_with_mu

def main(
    CompDiv, n_bins="default", error=False,
    CPU_Count=None, verbose=False, choice="uniform", scaling=1.
    ) :
    '''Kullback-Leibler divergence estimation with Camaglia Mora Walczak method.'''

    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>

    # number of categories
    K = CompDiv.K

    # number of bins
    if n_bins == "default" :
        # empirical choice ~
        n_bins = max( 1, np.round(5 * np.power(K / np.min([CompDiv.N_1, CompDiv.N_2]), 2)) )
        n_bins = min( n_bins, 100 ) 
    try :
        n_bins = int(n_bins)
    except :
        raise TypeError("The parameter `bins` requires an integer value or `default`.")
    if n_bins > 1 :
        n_bins += (n_bins%2)==0 # only odd numbers of bins
    saddle_point_method = (n_bins < 2) # only saddle

    # number of jobs
    try :
        CPU_Count = int(CPU_Count)
        if CPU_Count < 1 :
            raise TypeError("`CPU_Count` requires an integer greater than 0. Falling back to 1.") 
    except :
        CPU_Count = multiprocessing.cpu_count()
    CPU_Count = min( CPU_Count, n_bins**2 )

    # verbose 
    disable = not verbose

    #  Find Point for Maximum Likelihood #   
    a_star, b_star = optimal_KL_divergence_params( CompDiv, choice=choice, scaling=scaling )
    # FIXME : optimal_divergence_params_ arguments are messy

    if saddle_point_method is True :

        # >>>>>>>>>>>>>>>>>>>>>>>
        #  SADDLE POINT METHOD  #
        # <<<<<<<<<<<<<<<<<<<<<<<

        DKL1_star = CompDiv.divergence( a_star, b_star )
        if error is True :
            DKL2_star = CompDiv.squared_divergence( a_star, b_star )
            DKL_StdDev_star = np. sqrt(DKL2_star - np.power(DKL1_star, 2))  
            estimate = np.array([DKL1_star, DKL_StdDev_star], dtype=np.float64) 
        else :
            estimate = np.array([DKL1_star], dtype=np.float64) 

    else :    

        # >>>>>>>>>>>>>>>>>>>>
        #  PRE COMPUTATIONS  #
        # <<<<<<<<<<<<<<<<<<<<

        hess_LogPosterior = log_meta_posterior_hess([a_star, b_star], CompDiv, choice, {"scaling" : scaling})

        # FIXME : this bin choice may be wrong : 
        std_a = np.power( - hess_LogPosterior[:,0,0], -0.5 )
        alpha_vec = np.append(
            np.logspace( min(BOUND_DIR[0], np.log10(a_star-N_SIGMA*std_a)), np.log10(a_star), n_bins//2 )[:-1],
            np.logspace( np.log10(a_star), np.log10(a_star+N_SIGMA*std_a), n_bins//2+1 )
        )
        std_b = np.power( - hess_LogPosterior[:,1,1], -0.5 )
        beta_vec = np.append(
            np.logspace( min(BOUND_DIR[0], np.log10(b_star-N_SIGMA*std_b)), np.log10(b_star), n_bins//2 )[:-1],
            np.logspace( np.log10(b_star), np.log10(b_star+N_SIGMA*std_b), n_bins//2+1 )
        )

        #  Compute Posterior (old ``Measure Mu``) for alpha and beta #
        log_mu_alpha = list(map(lambda a : Polya(a, CompDiv.compact_1).log(), alpha_vec ))   
        log_mu_alpha -= np.max( log_mu_alpha )
        mu_alpha = np.exp( log_mu_alpha )
        log_mu_beta = list(map(lambda b : Polya(b, CompDiv.compact_2).log(), beta_vec )) 
        log_mu_beta -= np.max( log_mu_beta )
        mu_beta = np.exp( log_mu_beta )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # FIXME : parallelization is weak...
        
        args = [ x for x in itertools.product(alpha_vec, beta_vec) ]
        POOL = multiprocessing.Pool( CPU_Count ) 
        tqdm_args = tqdm(args, total=len(args), desc='Evaluations...', disable=disable)
        all_DKL_ab = POOL.starmap( CompDiv.divergence, tqdm_args )
        all_DKL_ab = np.asarray( all_DKL_ab ).reshape(len(alpha_vec), len(beta_vec))
    
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL2 estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error is True :
            tqdm_args = tqdm(args, total=len(args), desc='Squared', disable=disable)
            all_DKL2_ab = POOL.starmap( CompDiv.squared_divergence, tqdm_args )
            all_DKL2_ab = np.asarray( all_DKL2_ab ).reshape(len(alpha_vec), len(beta_vec))
        
        POOL.close()

        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>

        #  Compute MetaPrior_DKL   #
        X, Y = np.meshgrid(alpha_vec, beta_vec)
        all_phi = DirKLdiv( [X, Y], K, choice ).Metapr().reshape(len(alpha_vec), len(beta_vec))
        all_DKL_ab_times_phi = np.multiply( all_phi, all_DKL_ab )
        args = np.concatenate([all_phi, all_DKL_ab_times_phi])
        if error is True :
            all_DKL2_ab_times_phi = np.multiply( all_phi, all_DKL2_ab )
            args = np.concatenate([args, all_DKL2_ab_times_phi])

        integrations_a = list(map( lambda x : integral_with_mu( mu_beta, x, beta_vec ), args ))
        integrations_a = np.asarray(  integrations_a )
        Zeta = integral_with_mu( mu_alpha, integrations_a[:len(alpha_vec)], alpha_vec )
        DKL1 = integral_with_mu( mu_alpha, integrations_a[len(alpha_vec):2*len(alpha_vec)], alpha_vec ) 
        DKL1 = mp.fdiv( DKL1, Zeta )  
        
        if error is False :  
            estimate = np.array(DKL1, dtype=np.float) 
        else :
            DKL2 = mp.fdiv( integral_with_mu(mu_alpha, integrations_a[2*len(alpha_vec):], alpha_vec ), Zeta )  
            DKL_devStd = np.sqrt(DKL2 - np.power(DKL1, 2))  
            estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
        
    return estimate