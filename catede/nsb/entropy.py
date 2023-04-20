#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Nemenmann-Shafee-Bialek Method - Entropy Estimator
    Copyright (C) April 2023 Francesco Camaglia, LPENS 

    ref: 
    Nemenman, I., Shafee, F. & Bialek, W. Entropy and Inference, Revisited. 
    Advances in Neural Information Processing Systems vol. 14 (MIT Press, 2001).
'''

import itertools
import warnings
import numpy as np
from mpmath import mp
import multiprocessing
from tqdm import tqdm
from ..bayesian_calculus import *

class nsb_wrapper( ) :
    ''' Common functions for all wrapper'''
    def metapr( self, var ) :
        return self.dir_meta_obj.metapr(var)
    def logmetapr( self, var ) :
        return self.dir_meta_obj.logmetapr(var)
    def optimal_entropy_param(self) :
        init_guess = optimal_polya_param(self.cpct_exp)
        return self.meta_likelihood.maximize([init_guess])
    def neglog_evidence_hess(self, var) :
        return self.meta_likelihood.neglog_hess(var)
    def lgscl_optimal_entropy_param(self) :
        init_guess = np.log(optimal_polya_param(self.cpct_exp))
        return self.meta_likelihood.lgscl_maximize([init_guess])
    def lgscl_neglog_evidence_hess(self, var) :
        return self.meta_likelihood.lgscl_neglog_hess(var)

def nsb_estimator(nsb_wrap, error=False, n_bins=None, cpu_count=None, verbose=False) :
    '''Entropy estimators with NSB method.'''

    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>

    #
    # load user number of bins
    #
    if n_bins == None :
        if verbose == True :
           warnings.warn("The precision is chosen by default.")
    else :
        try :
            n_bins = int(n_bins)
        except :
            raise TypeError("The parameter `n_bins` requires an integer value.")

    #
    # load user number of jobs
    #
    try :
        cpu_count = int(cpu_count)
        if cpu_count < 1 :
            raise TypeError("Parameter `cpu_count` requires an integer greater than 0.") 
    except :
        cpu_count = multiprocessing.cpu_count()

    # >>>>>>>>>>>>>>>>
    #  RANGE CHOICE  #
    # <<<<<<<<<<<<<<<<

    #  Find Point for Maximum Likelihood of parameters #  
    a_star = nsb_wrap.optimal_entropy_param() 
    # std dev around the max
    Log_evidence_hess = nsb_wrap.neglog_evidence_hess(a_star)
    std_a = np.power(Log_evidence_hess, -0.5)

    # if std_a small : saddle_point_method = True
    n_bins = round(empirical_n_bins(nsb_wrap.cpct_exp.N, nsb_wrap.cpct_exp.K))
    saddle_point_method = (n_bins < 2) # only saddle

    if saddle_point_method == True :

        # 
        #  SADDLE POINT  #
        # 

        S1 = nsb_wrap.entropy(a_star) 
        if error == True :
            S2 = nsb_wrap.squared_entropy(a_star)

    else :

        n_bins += (n_bins%2)==0 # only odd numbers of bins
        cpu_count = min( cpu_count, n_bins**2 )

        # >>>>>>>>>>>>>>>>>>>>
        #  PRE COMPUTATIONS  #
        # <<<<<<<<<<<<<<<<<<<<

        alpha_vec = centered_logspaced_binning(a_star, std_a, n_bins)

        #  Compute Posterior (old `Measure Mu`) for alpha #
        polya = Polya(nsb_wrap.cpct_exp)
        log_mu_alpha = np.array(list(map(lambda a : polya.log(a), alpha_vec)))
        log_mu_alpha += nsb_wrap.logmetapr(alpha_vec)
        log_mu_alpha -= np.max(log_mu_alpha) # regularization
        mu_a = np.exp( log_mu_alpha )
        
        # >>>>>>>>>>>>>>>>>>>>>>>
        #  estimators vs alpha  #
        # >>>>>>>>>>>>>>>>>>>>>>>

        POOL = multiprocessing.Pool(cpu_count)  

        # entropy( a ) computation
        tqdm_args = tqdm(alpha_vec, total=len(alpha_vec), desc="Error Eval", disable=~verbose) 
        all_S1_a = POOL.map( nsb_wrap.entropy, tqdm_args )
        all_S1_a = np.asarray(all_S1_a)
        
        # squared-entropy (a) computation
        if error is True :
            tqdm_args = tqdm(alpha_vec, total=len(alpha_vec), desc="Error Eval", disable=~verbose)
            all_S2_a = POOL.map(nsb_wrap.squared_entropy, tqdm_args)   
            all_S2_a = np.asarray(all_S2_a)
            
        POOL.close()

        # >>>>>>>>>>>>>>>
        #   estimators  #
        # >>>>>>>>>>>>>>>
            
        Zeta = np.trapz(mu_a, x=alpha_vec)
        S1 = mp.fdiv(np.trapz(np.multiply(mu_a, all_S1_a), x=alpha_vec), Zeta) 
        if error == True :       
            S2 = mp.fdiv(np.trapz(np.multiply(mu_a, all_S2_a), x=alpha_vec), Zeta) 
    ####
    
    estimate = np.array(nsb_wrap.mean(S1)) 
    if error == True :
        estimate = np.append(estimate, [nsb_wrap.std(S1, S2)])

    return np.float64(estimate)
