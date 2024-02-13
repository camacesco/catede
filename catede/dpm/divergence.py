#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Dirichlet Prior Mixture Method - Divergence Estimator
    Copyright (C) January 2024 Francesco Camaglia, LPENS 
'''

import itertools
import warnings
import numpy as np
from mpmath import mp
import multiprocessing
from tqdm.auto import tqdm
from ..bayesian_calculus import *

class dpm_wrapper( ) :
    ''' Common functions for all wrapper'''
    def metapr( self, var ) :
        return self.dir_meta_obj.metapr(var)
    def logmetapr( self, var ) :
        return self.dir_meta_obj.logmetapr(var)
    def optimal_divergence_params(self) :
        return self.meta_likelihood.maximize([INIT_GUESS, INIT_GUESS])
    def neglog_evidence(self, var) :
        return self.meta_likelihood.neglog( var )
    def neglog_evidence_hess(self, var) :
        return self.meta_likelihood.neglog_hess( var )
    def lgscl_optimal_divergence_params(self) :
        return self.meta_likelihood.lgscl_maximize([INIT_GUESS, INIT_GUESS])
    def lgscl_neglog_evidence(self, var) :
        return self.meta_likelihood.lgscl_neglog(var)
    def lgscl_neglog_evidence_hess(self, var) :
        return self.meta_likelihood.lgscl_neglog_hess(var)
    
###################
#  DPM ESTIMATOR  #
###################

def dpm_estimator(dpm_wrap, error=False, n_sigma=3, n_bins=None, cpu_count=None, 
                  verbose=False, logscaled=True) :
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
    if n_bins == None :
        n_bins = empirical_n_bins(min(dpm_wrap.cpct_div.N_1, dpm_wrap.cpct_div.N_2), dpm_wrap.cpct_div.K)
        if verbose == True :
           warnings.warn("The precision is chosen by default.")
    try :
        n_bins = int(n_bins)
    except :
        raise TypeError("The parameter `n_bins` requires an integer value.")
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
    if logscaled == True :
        a_star, b_star = dpm_wrap.lgscl_optimal_divergence_params()
        neglog_evidence_hess = dpm_wrap.lgscl_neglog_evidence_hess([a_star, b_star])
    else :
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

        if logscaled == True :
            alpha_vec = lgscl_binning(a_star, std_a, n_bins_a, n_sigma=n_sigma)
            beta_vec = lgscl_binning(b_star, std_b, n_bins_b, n_sigma=n_sigma)
        else :
            alpha_vec = centered_logspaced_binning(a_star, std_a, n_bins_a, n_sigma=n_sigma)
            beta_vec = centered_logspaced_binning(b_star, std_b, n_bins_b, n_sigma=n_sigma)

        # compute measure
        if logscaled == True :
            log_measure = dpm_wrap.lgscl_neglog_evidence([alpha_vec, beta_vec])
        else :
            log_measure = dpm_wrap.neglog_evidence([alpha_vec, beta_vec])
        log_measure -= np.min(log_measure) # regularization
        measure = np.exp(-log_measure)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # FIXME : parallelization can be improved

        POOL = multiprocessing.Pool(cpu_count) 

        args = [x for x in itertools.product(alpha_vec, beta_vec)]
        tqdm_args = tqdm(args, total=len(args), desc='Evaluations...', disable=disable)
        all_DIV_ab = POOL.starmap(dpm_wrap.divergence, tqdm_args)
        all_DIV_ab = np.asarray(all_DIV_ab).reshape(n_bins_a, n_bins_b)
    
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DIV2 estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error == True :
            tqdm_args = tqdm(args, total=len(args), desc='Squared...', disable=disable)
            all_DIV2_ab = POOL.starmap(dpm_wrap.squared_divergence, tqdm_args)
            all_DIV2_ab = np.asarray(all_DIV2_ab).reshape(n_bins_a, n_bins_b)

        POOL.close()
    
        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>

        if logscaled == True :
            x = np.log(alpha_vec)
            y = np.log(beta_vec)
        else :
            x = alpha_vec
            y = beta_vec

        args = np.concatenate([measure, np.multiply(measure, all_DIV_ab)])
        if error is True :
            args = np.concatenate([args, np.multiply(measure, all_DIV2_ab)])
        integr_a = np.trapz(args, x=y, axis=1)
        Zeta = np.trapz(integr_a[:n_bins_a], x=x)
        DIV1 = np.trapz(integr_a[n_bins_a:2*n_bins_a], x=x) 
        DIV1 = mp.fdiv(DIV1, Zeta)  
        if error is True :  
            DIV2 = mp.fdiv(np.trapz(integr_a[2*n_bins_a:], x=x ), Zeta) 
        ####
    ####

    estimate = np.array(dpm_wrap.mean(DIV1)) 
    if error is True :
        estimate = np.append(estimate, [dpm_wrap.std(DIV1, DIV2)])
        
    return np.float64(estimate)