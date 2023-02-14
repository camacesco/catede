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

def main(
    CompDiv, error=False, n_bins="default", equal_prior=False,
    choice="uniform", scaling=1.,
    CPU_Count=None, verbose=False, 
    ) :
    '''Kullback-Leibler divergence estimation with Camaglia Mora Walczak method.'''

    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>

    # number of categories
    K = CompDiv.K

    # number of bins
    if n_bins == "default" :
        n_bins = empirical_n_bins( min(CompDiv.N_1, CompDiv.N_2), K )
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
    if equal_prior is True :
        a_star = optimal_equal_KLdiv_param( CompDiv )
    else :
        a_star, b_star = optimal_KL_divergence_params( CompDiv, choice=choice, scaling=scaling )

    if saddle_point_method is True :

        # >>>>>>>>>>>>>>>>>>>>>>>
        #  SADDLE POINT METHOD  #
        # <<<<<<<<<<<<<<<<<<<<<<<

        if equal_prior is True :
            DKL1= CompDiv.divergence( a_star, a_star )
            if error is True :
                DKL2 = CompDiv.squared_divergence( a_star, a_star )
        else :
            DKL1= CompDiv.divergence( a_star, b_star )
            if error is True :
                DKL2 = CompDiv.squared_divergence( a_star, b_star )
    else :    

        # >>>>>>>>>>>>>>>>>>>>
        #  PRE COMPUTATIONS  #
        # <<<<<<<<<<<<<<<<<<<<

        if equal_prior is True :
            hess_LogPosterior = log_equal_KLdiv_meta_posterior_hess(a_star, CompDiv)
            std_a = np.power( - hess_LogPosterior, -0.5 )
            # alpha
            alpha_vec = centered_logspaced_binning( a_star, std_a, n_bins )
            log_mu_alpha = list(map(lambda a : Polya(a, CompDiv.compact_1).log() + Polya(a, CompDiv.compact_2).log(), alpha_vec ))   
            log_mu_alpha -= np.max( log_mu_alpha ) # regularization
            mu_alpha = np.exp( log_mu_alpha )

        else :
            hess_LogPosterior = log_KL_divergence_meta_posterior_hess([a_star, b_star], CompDiv, choice, {"scaling" : scaling})
            std_a = np.power( - hess_LogPosterior[:,0,0], -0.5 )
            std_b = np.power( - hess_LogPosterior[:,1,1], -0.5 )
            # alpha
            alpha_vec = centered_logspaced_binning( a_star, std_a, n_bins )
            log_mu_alpha = list(map(lambda a : Polya(a, CompDiv.compact_1).log(), alpha_vec ))   
            log_mu_alpha -= np.max( log_mu_alpha ) # regularization
            mu_alpha = np.exp( log_mu_alpha )
            # beta
            beta_vec = centered_logspaced_binning( b_star, std_b, n_bins )
            log_mu_beta = list(map(lambda b : Polya(b, CompDiv.compact_2).log(), beta_vec )) 
            log_mu_beta -= np.max( log_mu_beta ) # regularization
            mu_beta = np.exp( log_mu_beta )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # FIXME : parallelization is weak...
        POOL = multiprocessing.Pool( CPU_Count ) 

        if equal_prior is True :
            args = [ x for x in zip(alpha_vec, alpha_vec) ]
            tqdm_args = tqdm(args, total=len(args), desc='Evaluations...', disable=disable)
            all_DKL_a = POOL.starmap( CompDiv.divergence, tqdm_args )

        else :
            args = [ x for x in itertools.product(alpha_vec, beta_vec) ]
            tqdm_args = tqdm(args, total=len(args), desc='Evaluations...', disable=disable)
            all_DKL_ab = POOL.starmap( CompDiv.divergence, tqdm_args )
            all_DKL_ab = np.asarray( all_DKL_ab ).reshape(len(alpha_vec), len(beta_vec))
    
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL2 estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error is True :
            if equal_prior is True :
                tqdm_args = tqdm(args, total=len(args), desc='Squared', disable=disable)
                all_DKL2_a = POOL.starmap( CompDiv.squared_divergence, tqdm_args )
            else :
                tqdm_args = tqdm(args, total=len(args), desc='Squared', disable=disable)
                all_DKL2_ab = POOL.starmap( CompDiv.squared_divergence, tqdm_args )
                all_DKL2_ab = np.asarray( all_DKL2_ab ).reshape(len(alpha_vec), len(beta_vec))
        
        POOL.close()

        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>

        if equal_prior is True :
            # for uniform binning in prior expected divergence
            A_vec = equalDirKLdiv( alpha_vec, CompDiv.K ).aPrioriExpec()
            Zeta = np.trapz( mu_alpha, x=A_vec )
            DKL1 = mp.fdiv( np.trapz( np.multiply( mu_alpha, all_DKL_a), x=A_vec ), Zeta ) 

            if error is True :      
                DKL2 = mp.fdiv( np.trapz( np.multiply( mu_alpha, all_DKL2_a ), x=A_vec ), Zeta ) 

        else :
            #  Compute MetaPrior_DKL   #
            X, Y = np.meshgrid(alpha_vec, beta_vec)
            all_phi = DirKLdiv( [X, Y], K, choice ).Metapr().reshape(len(alpha_vec), len(beta_vec))
            all_DKL_ab_times_phi = np.multiply( all_phi, all_DKL_ab )
            args = np.concatenate([all_phi, all_DKL_ab_times_phi])
            # FIXME : maybe it's better to rewrite for it to be clearer and collapse to equal_prior
            if error is True :
                all_DKL2_ab_times_phi = np.multiply( all_phi, all_DKL2_ab )
                args = np.concatenate([args, all_DKL2_ab_times_phi])
            integr_a = list(map( lambda i : np.trapz( np.multiply(mu_beta, i), x=beta_vec ), args ))
            integr_a = np.asarray(  integr_a )
            Zeta = np.trapz( np.multiply(mu_alpha, integr_a[:len(alpha_vec)]), x=alpha_vec )
            DKL1 = np.trapz( np.multiply(mu_alpha, integr_a[len(alpha_vec):2*len(alpha_vec)]), x=alpha_vec ) 
            DKL1 = mp.fdiv( DKL1, Zeta )  
            
            if error is True :  
                DKL2 = mp.fdiv( np.trapz(np.multiply(mu_alpha, integr_a[2*len(alpha_vec):]), x=alpha_vec ), Zeta )  
        ####
    ####
    
    if error is True :
        DKL_devStd = np.sqrt( DKL2 - np.power(DKL1, 2) )  
        estimate = np.array( [DKL1, DKL_devStd], dtype=np.float )   
    else :
        estimate = np.array( DKL1, dtype=np.float ) 
        
    return estimate