#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Nemenmann-Shafee-Bialek Estimator
    Copyright (C) November 2021 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
from scipy import optimize
import multiprocessing
import tqdm

from ._wolpert_wolf_calculus import *

def NemenmanShafeeBialek(
    compACTexp, error=False, n_bins=2.5e1, CPU_Count=None, verbose=False
    ):
    ''' Nemenamn-Shafee-Bialekd entropy estimator '''

    K = compACTexp.K
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>
        
    try :
        n_bins = int(n_bins)
    except :
        raise TypeError("The parameter `bins` requires an integer value.")
        
    try :
        CPU_Count = int(CPU_Count)
        if CPU_Count < 1 :
            raise IOError("The parameter `CPU_Count` requires an integer value greater than 0.")     
    except :
        CPU_Count = multiprocessing.cpu_count()
        
    disable = not verbose

    # >>>>>>>>>>>>>>>>>
    #  Compute Alpha  #
    # >>>>>>>>>>>>>>>>>

    a_NSB_star = optimal_entropy_param_( compACTexp )
    aux_range = np.append(
        np.logspace( -0.25, 0, np.floor(n_bins/2).astype(int) )[:-1],
        np.logspace( 0, 0.25, np.ceil(n_bins/2).astype(int) )
    )
    alpha_vec = a_NSB_star * aux_range
    A_vec = list(map( lambda a : implicit_entropy_vs_alpha_(a, 0, K), alpha_vec ) )
    
    mu_a = np.asarray( list(map( compACTexp._measureMu,  alpha_vec )) )
        
    # >>>>>>>>>>>>>>>>>>>>>>>
    #  estimators vs alpha  #
    # >>>>>>>>>>>>>>>>>>>>>>>
    
    POOL = multiprocessing.Pool( CPU_Count )   

    args = alpha_vec
    all_S1_a = POOL.map(
        compACTexp._post_entropy,
        tqdm.tqdm(args, total=len(args), desc="Estimator Eval", disable=disable)
        )
    all_S1_a = np.asarray(all_S1_a)
    
    if error is True :
        args = alpha_vec
        all_S2_a = POOL.map(
            compACTexp._post_entropy_squared,
            tqdm.tqdm(args, total=len(args), desc="Error Eval", disable=disable)
            )
        all_S2_a = np.asarray(all_S2_a)
    
    POOL.close()
    
    # >>>>>>>>>>>>>>>
    #   estimators  #
    # >>>>>>>>>>>>>>>
        
    Zeta = integral_with_mu_( mu_a, 1, A_vec )

    integral_S1 = integral_with_mu_(mu_a, all_S1_a, A_vec)
    S1 = mp.fdiv(integral_S1, Zeta)     

    if error is False :       
        shannon_estimate = np.array(S1, dtype=np.float) 
    else :
        S2 = mp.fdiv(integral_with_mu_(mu_a, all_S2_a, A_vec), Zeta)
        S_devStd = np.sqrt(S2 - np.power(S1, 2))
        shannon_estimate = np.array([S1, S_devStd], dtype=np.float)   
        
    return shannon_estimate