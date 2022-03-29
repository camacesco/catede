#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) January 2022 Francesco Camaglia, LPENS 
'''

from typing import IO
import warnings
import numpy as np
from mpmath import mp
import multiprocessing
import itertools
import tqdm
from ._aux_definitions import *

def Kullback_Leibler_CMW( compACTdiv, n_bins=5e2, cutoff_ratio=4, error=False, CPU_Count=None, verbose=False, equal_diversity=False ):
    '''Kullback-Leibler divergence estimation with Camaglia Mora Walczak method.
    '''
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

    if equal_diversity is True : # alpha = beta
        from ._cmw_KL_eqdiv_divergence import Kullback_Leibler_CMW_eqdiv
        return Kullback_Leibler_CMW_eqdiv( compACTdiv, n_bins=n_bins, cutoff_ratio=cutoff_ratio, error=error, CPU_Count=CPU_Count, verbose=verbose )

    else : # standard alpha != beta

        K = compACTdiv.K

        disable = not verbose 
            
        # >>>>>>>>>>>>>>>>>>>>>>>>>
        #  Compute Alpha and Beta #
        # >>>>>>>>>>>>>>>>>>>>>>>>>
        
        # multiprocessing (WARNING:)
        POOL = multiprocessing.Pool( CPU_Count )  
        
        # FIXME : logspace instead of linspace 
        '''
        A_vec = np.linspace(0, np.log(K), n_bins+2)[1:-1]
        B_cutoff = ( cutoff_ratio + 1 ) * np.log(K)
        B_vec = np.linspace(np.log(K), B_cutoff, n_bins+2)[1:-1]
        '''

        A_vec = np.logspace( 0, np.log10( np.log(K) + 1 ), n_bins+2 ) - 1 
        A_vec = A_vec[1:-1]
        #B_cutoff = ( cutoff_ratio + 1 ) * np.log(K)
        B_cutoff = np.max([10, ( cutoff_ratio + 1 )]) * np.log(K) 
        B_vec = np.logspace( np.log10( np.log(K) ), np.log10(B_cutoff) , n_bins+2 )
        B_vec = B_vec[1:-1]

        args = [ (implicit_entropy_vs_alpha, A, 0, 1e20, K) for A in A_vec ]
        args = args + [ (implicit_crossentropy_vs_beta, B, 1.e-20, 1e20, K) for B in B_vec ]
        
        params = POOL.starmap( get_from_implicit, tqdm.tqdm(args, total=len(args), desc='Pre-computations 1/3', disable=disable) )
        params = np.asarray( params )
        
        alpha_vec = params[:n_bins]
        beta_vec = params[n_bins:]
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  Compute MeasureMu alpha and beta #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        args = [ (alpha, compACTdiv.compact_1 ) for alpha in alpha_vec ]
        args = args + [ (beta, compACTdiv.compact_2 ) for beta in beta_vec ]
        
        measures = POOL.starmap( measureMu, tqdm.tqdm(args, total=len(args), desc='Pre-computations 2/3', disable=disable) )
        measures = np.asarray( measures )  
        
        mu_alpha = measures[:n_bins]
        mu_beta = measures[n_bins:]
        # regualarization 
        mu_alpha /= np.max( mu_alpha )
        mu_beta /= np.max( mu_beta )
                    
        # >>>>>>>>>>>>>>>>
        #  Compute Phi   #
        # >>>>>>>>>>>>>>>>
            
        args = [ x + (compACTdiv.K, cutoff_ratio,) for x in itertools.product(A_vec, B_vec)]
                
        all_phi = POOL.starmap( Phi, tqdm.tqdm(args, total=len(args), 
                                                desc='Pre-computations 3/3', disable=disable) )
        all_phi = np.asarray( all_phi ).reshape(n_bins, n_bins)
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        # FIXME : choose args here only where Phi > 0
        args = [ x + (compACTdiv,) for x in itertools.product(alpha_vec, beta_vec)]
                
        all_DKL_ab = POOL.starmap( estimate_DKL_at_alpha_beta, tqdm.tqdm(args, total=len(args), 
                                                                        desc='Evaluations', disable=disable) )
        all_DKL_ab = np.asarray( all_DKL_ab ).reshape(n_bins, n_bins)
        all_DKL_ab_times_phi = np.multiply( all_phi, all_DKL_ab )
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  DKL2 estimator vs alpha,beta  #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
        if error is True :

            # FIXME : choose args here only where all_DKL_ab > a certain threshold
            all_DKL2_ab = POOL.starmap( estimate_DKL2_at_alpha_beta, tqdm.tqdm(args, total=len(args), 
                                                                            desc='Error evaluations', disable=disable) )
            all_DKL2_ab = np.asarray( all_DKL2_ab ).reshape(n_bins, n_bins)
            all_DKL2_ab_times_phi = np.multiply( all_phi, all_DKL2_ab )
        
        # >>>>>>>>>>>>>>>>>
        #   integrations  #
        # >>>>>>>>>>>>>>>>>
                    
        args = [ (mu_beta, x, B_vec) for x in all_phi ]
        args = args + [ (mu_beta, x, B_vec) for x in all_DKL_ab_times_phi ]
        if error is True : 
            args = args + [ (mu_beta, x, B_vec) for x in all_DKL2_ab_times_phi ]
            
        integrations_a = POOL.starmap( integral_with_mu, tqdm.tqdm(args, total=len(args), desc='Integration', disable=disable) )
        integrations_a = np.asarray(  integrations_a )
        
        # multiprocessing (WARNING:)    
        POOL.close()
        
        Zeta = integral_with_mu( mu_alpha, integrations_a[:n_bins], A_vec )
        DKL1 = integral_with_mu( mu_alpha, integrations_a[n_bins:2*n_bins], A_vec ) 
        DKL1 = mp.fdiv( DKL1, Zeta )  
        if error is False :  
            kullback_leibler_estimate = np.array(DKL1, dtype=np.float) 
        else :
            DKL2 = mp.fdiv( integral_with_mu(mu_alpha, integrations_a[2*n_bins:], A_vec ), Zeta )  
            DKL_devStd = np. sqrt(DKL2 - np.power(DKL1, 2))  
            kullback_leibler_estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
        
        return kullback_leibler_estimate

#########
#  PHI  #
#########

def Phi( A, B, K, cutoff_ratio ) :
    '''Mixture of Prior Kernel (?)'''

    D = B - A

    # choice of the prior
    rho_D = 1. # uniform

    # function by cases 
    if D >= cutoff_ratio * np.log(K) : # cutoff
        return 0.
    elif D >= np.log(K) : # uniform
        return rho_D / np.log(K)
    else :
        return rho_D / D 
    
#########################################
#  DKL estimation vs Dirichelet params  #
#########################################


def estimate_DKL_at_alpha_beta( a, b, compACTdiv ):
    '''Estimate of the divergence at fixed alpha and beta.'''
    
    # loading parameters from Divergence Compact        
    N_1, N_2 = compACTdiv.N_1, compACTdiv.N_2
    nn_1, nn_2, ff = compACTdiv.nn_1, compACTdiv.nn_2, compACTdiv.ff
    K = compACTdiv.K
    
    # DKL computation
    temp = ff.dot( (nn_1+a) * ( D_diGmm(N_2+K*b, nn_2+b) - D_diGmm(N_1+K*a+1, nn_1+a+1) ) ) 
    DKL_ab = mp.fdiv( temp, N_1+K*a )    
            
    return DKL_ab

def estimate_DKL2_at_alpha_beta( a, b, compACTdiv ) :
    ''' Estimate of the squared divergence at ficed alpha and beta.'''

    single_sum, double_sum = 0, 0
    
    # loading parameters from Divergence Compact        
    N_1, N_2, K = compACTdiv.N_1, compACTdiv.N_2, compACTdiv.K
    nn_1, nn_2, ff = compACTdiv.nn_1, compACTdiv.nn_2, compACTdiv.ff
    
    omega_diag = np.multiply( nn_1+a+1, nn_1+a )
    omega_mtx = np.outer( nn_1+a , nn_1+a )
    
    ''' term : q_i q_j ln(q_i) ln(q_j) '''
    # single sum term
    Ss_Term1 = np.power(D_diGmm(nn_1+a+2, N_1+K*a+2), 2) + D_triGmm(nn_1+a+2, N_1+K*a+2)
    single_sum = Ss_Term1
    # double sum term 
    Ds_Term1 = np.outer( D_diGmm(nn_1+a+1, N_1+K*a+2), D_diGmm(nn_1+a+1, N_1+K*a+2) ) - triGmm(N_1+K*a+2)
    double_sum = Ds_Term1

    ''' term : - q_i q_j ln(q_i) ln(t_j) - q_j q_i ln(q_j) ln(t_i) '''
    # single sum term
    Ss_Term2 = np.multiply( D_diGmm(nn_1+a+2, N_1+K*a+2), D_diGmm(nn_2+b, N_2+K*b) )
    single_sum -= 2 * Ss_Term2
    # double sum term 
    Ds_Term2 = np.outer( D_diGmm(nn_1+a+1, N_1+K*a+2), D_diGmm(nn_2+b, N_2+K*b) )
    double_sum -= 2 * Ds_Term2

    ''' term : q_i q_j ln(t_i) ln(t_j) '''
    # single sum term
    Ss_Term3 = np.power(D_diGmm(nn_2+b, N_2+K*b), 2) + D_triGmm(nn_2+b, N_2+K*b)
    single_sum += Ss_Term3
    # double sum term 
    Ds_Term3 = np.outer( D_diGmm(nn_2+b, N_2+K*b), D_diGmm(nn_2+b, N_2+K*b) ) - triGmm(N_2+K*b)
    double_sum += Ds_Term3
    
    ''' multiply times Omega matrix '''
    single_sum = np.multiply( single_sum, omega_diag )
    double_sum = np.multiply( double_sum, omega_mtx )
    
    ''' sum using frequencies '''
    DKL2_ab = ff.dot( single_sum - double_sum.diagonal() + double_sum.dot(ff) )
    DKL2_ab = mp.fdiv( DKL2_ab, mp.fmul(N_1+K*a+1, N_1+K*a) ) 
    
    return DKL2_ab