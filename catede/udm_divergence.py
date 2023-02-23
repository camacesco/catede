#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Unbiased Dirichlet Mixture Method - Divergence Estimator
Copyright (C) February 2023 Francesco Camaglia, LPENS 
'''

import itertools
import warnings
import numpy as np
from mpmath import mp
import multiprocessing
from tqdm import tqdm
from .new_calculus import *

def udm_estimator( udm_wrap, error=False, n_bins="default", equal_prior=False, cpu_count=None, verbose=False, ) :
    '''Divergence estimator with UDM method.'''

    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>

    #
    # verbose 
    #
    disable = not verbose

    #
    # number of bins
    #
    if n_bins == "default" :
        n_bins = empirical_n_bins( min(udm_wrap.comp_div.N_1, udm_wrap.comp_div.N_2), udm_wrap.comp_div.K )
        if verbose is True :
           warnings.warn("The precision is chosen by default.")
    try :
        n_bins = int(n_bins)
    except :
        raise TypeError("The parameter `n_bins` requires an integer value or `default`.")
    if n_bins > 1 :
        n_bins += (n_bins%2)==0 # only odd numbers of bins
    saddle_point_method = (n_bins < 2) # only saddle

    #
    # number of jobs
    #
    try :
        cpu_count = int(cpu_count)
        if cpu_count < 1 :
            raise TypeError("`cpu_count` requires an integer greater than 0.") 
    except :
        cpu_count = multiprocessing.cpu_count()

    if equal_prior is True :
        cpu_count = min( cpu_count, n_bins )
    else :
        cpu_count = min( cpu_count, n_bins**2 )

    # >>>>>>>>>>>>>>>>>>>>
    #  PRE COMPUTATIONS  #
    # <<<<<<<<<<<<<<<<<<<<

    #  Find Point of maximum evidence #   
    if equal_prior is True :
        a_star = udm_wrap.optimal_equal_param( )
    else :
        a_star, b_star = udm_wrap.optimal_divergence_params( )

    if saddle_point_method is True :

        # 
        #  SADDLE POINT  #
        # 

        if equal_prior is True :
            DIV1= udm_wrap.divergence( a_star, a_star )
            if error is True :
                DIV2 = udm_wrap.squared_divergence( a_star, a_star )
        else :
            DIV1= udm_wrap.divergence( a_star, b_star )
            if error is True :
                DIV2 = udm_wrap.squared_divergence( a_star, b_star )
    else :    

        # 
        #  FULL INTEGRATION  #
        # 

        # FIXME : these calls to Polya posterior are convoluted
        # they should be a method of compact experiments probably
        # and vectorialized

        if equal_prior is True :
            Log_evidence_hess = udm_wrap.log_equal_evidence_hess( a_star )
            std_a = np.power( - Log_evidence_hess, -0.5 )
            # alpha
            alpha_vec = centered_logspaced_binning( a_star, std_a, n_bins )
            log_mu_alpha = list(map(lambda a : Polya(a, udm_wrap.comp_div.compact_1).log() + Polya(a, udm_wrap.comp_div.compact_2).log(), alpha_vec ))   
            log_mu_alpha -= np.max( log_mu_alpha ) # regularization
            mu_alpha = np.exp( log_mu_alpha )

        else :
            
            Log_evidence_hess = udm_wrap.log_evidence_hess( a_star, b_star )
            std_a = np.power( - Log_evidence_hess[:,0,0], -0.5 )
            std_b = np.power( - Log_evidence_hess[:,1,1], -0.5 )
            # alpha
            alpha_vec = centered_logspaced_binning( a_star, std_a, n_bins )
            log_mu_alpha = list(map(lambda a : Polya(a, udm_wrap.comp_div.compact_1).log(), alpha_vec ))   
            log_mu_alpha -= np.max( log_mu_alpha ) # regularization
            mu_alpha = np.exp( log_mu_alpha )
            # beta
            beta_vec = centered_logspaced_binning( b_star, std_b, n_bins )
            log_mu_beta = list(map(lambda b : Polya(b, udm_wrap.comp_div.compact_2).log(), beta_vec )) 
            log_mu_beta -= np.max( log_mu_beta ) # regularization
            mu_beta = np.exp( log_mu_beta )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # FIXME : parallelization is weak...
        POOL = multiprocessing.Pool( cpu_count ) 

        if equal_prior is True :
            args = [ x for x in zip( alpha_vec, alpha_vec ) ]
            tqdm_args = tqdm( args, total=len(args), desc='Evaluations...', disable=disable )
            all_DIV_a = POOL.starmap( udm_wrap.divergence, tqdm_args )

        else :
            args = [ x for x in itertools.product( alpha_vec, beta_vec ) ]
            tqdm_args = tqdm( args, total=len(args), desc='Evaluations...', disable=disable )
            all_DIV_ab = POOL.starmap( udm_wrap.divergence, tqdm_args )
            all_DIV_ab = np.asarray( all_DIV_ab ).reshape( len(alpha_vec), len(beta_vec) )
    
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DIV2 estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error is True :
            tqdm_args = tqdm(args, total=len(args), desc='Squared...', disable=disable)
            if equal_prior is True :
                all_DIV2_a = POOL.starmap( udm_wrap.squared_divergence, tqdm_args )
            else :
                all_DIV2_ab = POOL.starmap( udm_wrap.squared_divergence, tqdm_args )
                all_DIV2_ab = np.asarray( all_DIV2_ab ).reshape( len(alpha_vec), len(beta_vec) )

        POOL.close()
    
        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>

        if equal_prior is True :
            # for uniform binning in prior expected divergence
            A_vec = udm_wrap.equal_prior( alpha_vec )
            Zeta = np.trapz( mu_alpha, x=A_vec )
            DIV1 = mp.fdiv( np.trapz( np.multiply( mu_alpha, all_DIV_a ), x=A_vec ), Zeta ) 

            if error is True :      
                DIV2 = mp.fdiv( np.trapz( np.multiply( mu_alpha, all_DIV2_a ), x=A_vec ), Zeta ) 

        else :
            #  Compute Prior   #
            X, Y = np.meshgrid(alpha_vec, beta_vec)
            all_phi = udm_wrap.udm_prior( [X, Y] ).reshape(len(alpha_vec), len(beta_vec))
            args = np.concatenate( [all_phi, np.multiply( all_phi, all_DIV_ab ) ] )

            if error is True :
                args = np.concatenate( [ args, np.multiply( all_phi, all_DIV2_ab ) ] )

            integr_a = np.asarray(list(map( lambda i : np.trapz( np.multiply(mu_beta, i), x=beta_vec ), args )))
            Zeta = np.trapz( np.multiply(mu_alpha, integr_a[:len(alpha_vec)]), x=alpha_vec )
            DIV1 = np.trapz( np.multiply(mu_alpha, integr_a[len(alpha_vec):2*len(alpha_vec)]), x=alpha_vec ) 
            DIV1 = mp.fdiv( DIV1, Zeta )  
            if error is True :  
                DIV2 = mp.fdiv( np.trapz(np.multiply(mu_alpha, integr_a[2*len(alpha_vec):]), x=alpha_vec ), Zeta ) 
        ####
    ####

    estimate = np.array( udm_wrap.estim_mean(DIV1) ) 
    if error is True :
        estimate = np.append( estimate, [udm_wrap.estim_std(DIV1, DIV2)] )   
        
    return np.float64( estimate )