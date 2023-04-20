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
from ..bayesian_calculus import *

class dpm_wrapper( ) :
    ''' Common functions for all wrapper'''
    def metapr( self, var ) :
        return self.dir_meta_obj.metapr(var)
    def logmetapr( self, var ) :
        return self.dir_meta_obj.logmetapr(var)
    def optimal_divergence_params(self) :
        guess_a = optimal_polya_param(self.cpct_div.compact_1)
        guess_b = optimal_polya_param(self.cpct_div.compact_2)
        return self.meta_likelihood.maximize([guess_a, guess_b])
    def neglog_evidence_hess(self, var) :
        return self.meta_likelihood.neglog_hess( var )
    def lgscl_optimal_divergence_params(self) :
        guess_a = np.log(optimal_polya_param(self.cpct_div.compact_1))
        guess_b = np.log(optimal_polya_param(self.cpct_div.compact_2))
        return self.meta_likelihood.lgscl_maximize([guess_a, guess_b])
    def lgscl_neglog_evidence_hess(self, var) :
        return self.meta_likelihood.lgscl_neglog_hess(var)
    
###################
#  DPM ESTIMATOR  #
###################

def dpm_estimator( dpm_wrap, error=False, n_bins="default", cpu_count=None, verbose=False, ) :
    '''Divergence estimator with DPM method.'''

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
        n_bins = empirical_n_bins( min(dpm_wrap.cpct_div.N_1, dpm_wrap.cpct_div.N_2), dpm_wrap.cpct_div.K )
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

    cpu_count = min( cpu_count, n_bins**2 )

    # >>>>>>>>>>>>>>>>>>>>
    #  PRE COMPUTATIONS  #
    # <<<<<<<<<<<<<<<<<<<<

    #  Find Point of maximum evidence #
    a_star, b_star = dpm_wrap.optimal_divergence_params()
   
    neglog_evidence_hess = dpm_wrap.neglog_evidence_hess([a_star, b_star])
    std_a = np.power(neglog_evidence_hess[:,0,0], -0.5)
    std_b = np.power(neglog_evidence_hess[:,1,1], -0.5)

    # np.all([std_a, std_n]) small : saddle_point_method = True
    n_bins_a = n_bins
    n_bins_b = n_bins

    if saddle_point_method == True :

        # 
        #  SADDLE POINT  #
        # 

        DIV1= dpm_wrap.divergence(a_star, b_star)
        if error is True :
            DIV2 = dpm_wrap.squared_divergence(a_star, b_star)
    else :    

        # 
        #  FULL INTEGRATION  #
        # 

        alpha_vec = centered_logspaced_binning(a_star, std_a, n_bins_a)
        beta_vec = centered_logspaced_binning(b_star, std_b, n_bins_b)

        # compute polya
        polya_a = Polya(dpm_wrap.cpct_div.compact_1)
        log_mu_alpha = list(map(lambda a : polya_a.log(a), alpha_vec ))   
        polya_b = Polya(dpm_wrap.cpct_div.compact_2)
        log_mu_beta = list(map(lambda b : polya_b.log(b), beta_vec )) 

        log_mu = np.add.outer(log_mu_alpha, log_mu_beta)
        #  compute metaprior   #
        X, Y = np.meshgrid(alpha_vec, beta_vec)
        log_mu += dpm_wrap.logmetapr([X, Y]).reshape(len(alpha_vec), len(beta_vec))

        log_mu -= np.max( log_mu ) # regularization
        mu = np.exp( log_mu )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # FIXME : parallelization is weak...
        POOL = multiprocessing.Pool( cpu_count ) 

        args = [x for x in itertools.product( alpha_vec, beta_vec )]
        tqdm_args = tqdm(args, total=len(args), desc='Evaluations...', disable=disable)
        all_DIV_ab = POOL.starmap( dpm_wrap.divergence, tqdm_args )
        all_DIV_ab = np.asarray(all_DIV_ab).reshape(len(alpha_vec), len(beta_vec))
    
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DIV2 estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error == True :
            tqdm_args = tqdm(args, total=len(args), desc='Squared...', disable=disable)
            all_DIV2_ab = POOL.starmap(dpm_wrap.squared_divergence, tqdm_args)
            all_DIV2_ab = np.asarray(all_DIV2_ab).reshape(len(alpha_vec), len(beta_vec))

        POOL.close()
    
        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>

        args = np.concatenate([mu, np.multiply(mu, all_DIV_ab)])
        if error is True :
            args = np.concatenate([args, np.multiply(mu, all_DIV2_ab)])
        integr_a = np.asarray(list(map(lambda i : np.trapz(i, x=beta_vec), args)))
        Zeta = np.trapz(integr_a[:len(alpha_vec)], x=alpha_vec)
        DIV1 = np.trapz(integr_a[len(alpha_vec):2*len(alpha_vec)], x=alpha_vec) 
        DIV1 = mp.fdiv( DIV1, Zeta )  
        if error is True :  
            DIV2 = mp.fdiv(np.trapz(integr_a[2*len(alpha_vec):], x=alpha_vec ), Zeta) 
        ####
    ####

    estimate = np.array(dpm_wrap.mean(DIV1)) 
    if error is True :
        estimate = np.append(estimate, [dpm_wrap.std(DIV1, DIV2)])
        
    return np.float64(estimate)