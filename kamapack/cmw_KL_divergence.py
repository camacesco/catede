#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) January 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
import multiprocessing
import itertools
import tqdm
from ._aux_definitions import *

def Kullback_Leibler_CMW( compACTdiv, bins=5e2, cutoff_ratio=5, error=False, CPU_Count=None, verbose=False ):
    '''Kullback-Leibler divergence estimation with Camaglia Mora Walczak method.
    '''

    K = compACTdiv.K
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>
                 
    try :
        n_bins = int(bins)
    except :
        raise TypeError("The parameter `bins` requires an integer value.")
        
    try :
        CPU_Count = int(CPU_Count)
        if CPU_Count < 1 :
            raise IOError("The parameter `CPU_Count` requires an integer value greater than 0.")     
    except :
        CPU_Count = multiprocessing.cpu_count()

    disable = not verbose 
        
    # >>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>
    
    # multiprocessing (WARNING:)
    POOL = multiprocessing.Pool( CPU_Count )  
    
    S_vec = np.linspace(0, np.log(K), n_bins+2)[1:-1]
    H_cutoff = cutoff_ratio * np.log(K)
    H_vec = np.linspace(np.log(K), H_cutoff, n_bins+2)[1:-1]
    
    args = [ (implicit_S_vs_Alpha, S, 0, 1e20, K) for S in S_vec ]
    args = args + [ (implicit_H_vs_Beta, H, 1.e-20, 1e20, K) for H in H_vec ]
     
    params = POOL.starmap( get_from_implicit, tqdm.tqdm(args, total=len(args), desc='Pre-computations 1/3', disable=disable) )
    params = np.asarray( params )
    
    Alpha_vec = params[:n_bins]
    Beta_vec = params[n_bins:]
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute MeasureMu Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    args = [ (a, compACTdiv.compact_A ) for a in Alpha_vec ]
    args = args + [ (b, compACTdiv.compact_B ) for b in Beta_vec ]
    
    measures = POOL.starmap( measureMu, tqdm.tqdm(args, total=len(args), desc='Pre-computations 2/3', disable=disable) )
    measures = np.asarray( measures )  
    
    mu_a = measures[:n_bins]
    mu_b = measures[n_bins:]
    # regualarization 
    mu_a /= np.max( mu_a )
    mu_b /= np.max( mu_b )
                
    # >>>>>>>>>>>>>>>>
    #  Compute Phi   #
    # >>>>>>>>>>>>>>>>
        
    args = [ x + (compACTdiv.K,) for x in itertools.product(S_vec, H_vec)]
            
    all_phi = POOL.starmap( Phi, tqdm.tqdm(args, total=len(args), 
                                               desc='Pre-computations 3/3', disable=disable) )
    all_phi = np.asarray( all_phi ).reshape(n_bins, n_bins)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  DKL estimator vs alpha,beta  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    args = [ x + (compACTdiv,) for x in itertools.product(Alpha_vec,Beta_vec)]
            
    all_DKL_ab = POOL.starmap( estimate_DKL_at_alpha_beta, tqdm.tqdm(args, total=len(args), 
                                                                     desc='Evaluations', disable=disable) )
    all_DKL_ab = np.asarray( all_DKL_ab ).reshape(n_bins, n_bins)
    all_DKL_ab_times_phi = np.multiply( all_phi, all_DKL_ab )
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  DKL2 estimator vs alpha,beta  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    if error is True :
        all_DKL2_ab = POOL.starmap( estimate_DKL2_at_alpha_beta, tqdm.tqdm(args, total=len(args), 
                                                                         desc='Error evaluations', disable=disable) )
        all_DKL2_ab = np.asarray( all_DKL2_ab ).reshape(n_bins, n_bins)
        all_DKL2_ab_times_phi = np.multiply( all_phi, all_DKL2_ab )
    
    # >>>>>>>>>>>>>>>>>
    #   integrations  #
    # >>>>>>>>>>>>>>>>>
                
    args = [ (mu_b, x, H_vec) for x in all_phi ]
    args = args + [ (mu_b, x, H_vec) for x in all_DKL_ab_times_phi ]
    if error is True : 
        args = args + [ (mu_b, x, H_vec) for x in all_DKL2_ab_times_phi ]
        
    integrations_a = POOL.starmap( integral_with_mu, tqdm.tqdm(args, total=len(args), desc='Integration', disable=disable) )
    integrations_a = np.asarray(  integrations_a )
    
    # multiprocessing (WARNING:)    
    POOL.close()
    
    Zeta = integral_with_mu( mu_a, integrations_a[:n_bins], S_vec )
    DKL1 = integral_with_mu( mu_a, integrations_a[n_bins:2*n_bins], S_vec ) 
    DKL1 = mp.fdiv( DKL1, Zeta )  
    if error is False :  
        kullback_leibler_estimate = np.array(DKL1, dtype=np.float) 
    else :
        DKL2 = mp.fdiv( integral_with_mu(mu_a, integrations_a[2*n_bins:], S_vec ), Zeta )  
        DKL_devStd = np.sqrt(DKL2 - np.power(DKL1, 2))  
        kullback_leibler_estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
    
    return kullback_leibler_estimate

#########
#  PHI  #
#########

def Phi( S, H, K ) :
    '''Uniformizing function (?)'''
    z = H - S
    if z >= np.log(K) :
        return 1. / np.log(K)
    else :
        return 1. / z 
    
#########################################
#  DKL estimation vs Dirichelet params  #
#########################################


def estimate_DKL_at_alpha_beta( a, b, compACTdiv ):
    '''Estimate of the divergence at fixed alpha and beta.'''
    
    # loading parameters from Divergence Compact        
    N_A, N_B = compACTdiv.N_A, compACTdiv.N_B
    nn_A, nn_B, ff = compACTdiv.nn_A, compACTdiv.nn_B, compACTdiv.ff
    K = compACTdiv.K
    
    # DKL computation
    temp = ff.dot( (nn_A+a) * ( D_diGmm(N_B+K*b, nn_B+b) - D_diGmm(N_A+K*a+1, nn_A+a+1) ) ) 
    DKL_ab = mp.fdiv( temp, N_A+K*a )    
            
    return DKL_ab

def estimate_DKL2_at_alpha_beta( a, b, compACTdiv ) :
    ''' Estimate of the squared divergence at ficed alpha and beta.'''

    single_sum, double_sum = 0, 0
    
    # loading parameters from Divergence Compact        
    N_A, N_B, K = compACTdiv.N_A, compACTdiv.N_B, compACTdiv.K
    nn_A, nn_B, ff = compACTdiv.nn_A, compACTdiv.nn_B, compACTdiv.ff
    
    omega_diag = np.multiply( nn_A+a+1, nn_A+a )
    omega_mtx = np.outer( nn_A+a , nn_A+a )
    
    ''' term : q_i q_j ln(q_i) ln(q_j) '''
    # single sum term
    Ss_Term1 = np.power(D_diGmm(nn_A+a+2, N_A+K*a+2), 2) + D_triGmm(nn_A+a+2, N_A+K*a+2)
    single_sum = Ss_Term1
    # double sum term 
    Ds_Term1 = np.outer( D_diGmm(nn_A+a+1, N_A+K*a+2), D_diGmm(nn_A+a+1, N_A+K*a+2) ) - triGmm(N_A+K*a+2)
    double_sum = Ds_Term1

    ''' term : - q_i q_j ln(q_i) ln(t_j) - q_j q_i ln(q_j) ln(t_i) '''
    # single sum term
    Ss_Term2 = np.multiply( D_diGmm(nn_A+a+2, N_A+K*a+2), D_diGmm(nn_B+b, N_B+K*b) )
    single_sum -= 2 * Ss_Term2
    # double sum term 
    Ds_Term2 = np.outer( D_diGmm(nn_A+a+1, N_A+K*a+2), D_diGmm(nn_B+b, N_B+K*b) )
    double_sum -= 2 * Ds_Term2

    ''' term : q_i q_j ln(t_i) ln(t_j) '''
    # single sum term
    Ss_Term3 = np.power(D_diGmm(nn_B+b, N_B+K*b), 2) + D_triGmm(nn_B+b, N_B+K*b)
    single_sum += Ss_Term3
    # double sum term 
    Ds_Term3 = np.outer( D_diGmm(nn_B+b, N_B+K*b), D_diGmm(nn_B+b, N_B+K*b) ) - triGmm(N_B+K*b)
    double_sum += Ds_Term3
    
    ''' multiply times Omega matrix '''
    single_sum = np.multiply( single_sum, omega_diag )
    double_sum = np.multiply( double_sum, omega_mtx )
    
    ''' sum using frequencies '''
    DKL2_ab = ff.dot( single_sum - double_sum.diagonal() + double_sum.dot(ff) )
    DKL2_ab = mp.fdiv( DKL2_ab, mp.fmul(N_A+K*a+1, N_A+K*a) ) 
    
    return DKL2_ab