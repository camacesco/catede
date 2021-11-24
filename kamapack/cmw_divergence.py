#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Divergence Estimator (in development)
    Copyright (C) November 2021 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
import multiprocessing
import itertools
import tqdm

from ._nsb_aux_definitions import *

def CamagliaMoraWalczak( compACTdiv, bins=5e2, cutoff_ratio=5, error=False, CPU_Count=None, progressbar=False ):
    '''
    CMW Kullback-Leibler divergence estimator description:
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

    disable = not progressbar 
        
    # >>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>
    
    
    # multiprocessing (WARNING:)
    POOL = multiprocessing.Pool( CPU_Count )  
    
    S_vec = np.linspace(0, np.log(K), n_bins)[1:-1]
    H_cutoff = cutoff_ratio * np.log(K)
    H_vec = np.linspace(np.log(K), H_cutoff, n_bins)[1:-1]
    
    args = [ (implicit_S_vs_Alpha, S, 0, 1e20, K) for S in S_vec ]
    args = args + [ (implicit_H_vs_Beta, H, 1.e-20, 1e20, K) for H in H_vec ]
     
    params = POOL.starmap( get_from_implicit, tqdm.tqdm(args, total=len(args), desc='Pre-computations 1/3', disable=disable) )
    params = np.asarray( params )
    
    Alpha_vec = params[:len(S_vec)]
    Beta_vec = params[len(S_vec):]
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  Compute MeasureMu Alpha and Beta #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    args = [ (a, compACTdiv.compact_A ) for a in Alpha_vec ]
    args = args + [ (b, compACTdiv.compact_B ) for b in Beta_vec ]
    
    measures = POOL.starmap( measureMu, tqdm.tqdm(args, total=len(args), desc='Pre-computations 2/3', disable=disable) )
    measures = np.asarray( measures )  
    
    mu_a = measures[:len(Alpha_vec)]
    mu_b = measures[len(Alpha_vec):]
                
    # >>>>>>>>>>>>>>>>>>
    #  Compute Sigma   #
    # >>>>>>>>>>>>>>>>>>
        
    # WARNING!: this itertools operation is a bottleneck!
    args = [ x + (compACTdiv.K,) for x in itertools.product(S_vec, H_vec)]
            
    all_sigma = POOL.starmap( Sigma, tqdm.tqdm(args, total=len(args), 
                                               desc='Pre-computations 3/3', disable=disable) )
    all_sigma = np.asarray( all_sigma ).reshape(len(S_vec), len(H_vec))
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  DKL estimator vs alpha,beta  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    # WARNING!: this itertools operation is a bottleneck!
    args = [ x[0] + x[1] + (compACTdiv,) for x in itertools.product(zip(Alpha_vec,S_vec),zip(Beta_vec,H_vec))]
            
    all_DKL_ab = POOL.starmap( estimate_DKL_at_alpha_beta, tqdm.tqdm(args, total=len(args), 
                                                                     desc='Evaluations', disable=disable) )
    all_DKL_ab = np.asarray( all_DKL_ab ).reshape(len(S_vec), len(H_vec))
    all_DKL_ab_times_sigma = np.multiply( all_sigma, all_DKL_ab )
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  DKL2 estimator vs alpha,beta  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    if error is True :
        all_DKL2_ab = POOL.starmap( estimate_DKL2_at_alpha_beta, tqdm.tqdm(args, total=len(args), 
                                                                         desc='Error evaluations', disable=disable) )
        all_DKL2_ab = np.asarray( all_DKL2_ab ).reshape(len(S_vec), len(H_vec))
        all_DKL2_ab_times_sigma = np.multiply( all_sigma, all_DKL2_ab )
    
    # >>>>>>>>>>>>>>>>>
    #   integrations  #
    # >>>>>>>>>>>>>>>>>
                
    args = [ (mu_b, x, H_vec) for x in all_sigma ]
    args = args + [ (mu_b, x, H_vec) for x in all_DKL_ab_times_sigma ]
    
    if error is True :    
        args = args + [ (mu_b, x, H_vec) for x in all_DKL2_ab_times_sigma ]
        
    integrations_a = POOL.starmap( integral_with_mu, tqdm.tqdm(args, total=len(args), desc='Integration', disable=disable) )
    integrations_a = np.asarray(  integrations_a )
    
    # multiprocessing (WARNING:)    
    POOL.close()
    
    Zeta = integral_with_mu( mu_a, integrations_a[:len(Alpha_vec)], S_vec )
    DKL1 = mp.fdiv( integral_with_mu( mu_a, integrations_a[len(Alpha_vec):2*len(Alpha_vec)], S_vec ), Zeta )  
    
    if error is False :       
        kullback_leibler_estimate = np.array(DKL1, dtype=np.float) 
        
    else :
        DKL2 = mp.fdiv( integral_with_mu(mu_a, integrations_a[2*len(Alpha_vec):], S_vec ), Zeta )  
        DKL_devStd = np.sqrt(DKL2 - np.power(DKL1, 2))  
        kullback_leibler_estimate = np.array([DKL1, DKL_devStd], dtype=np.float)   
    
    return kullback_leibler_estimate


###########
#  SIGMA  #
###########

def Sigma( S, H, K ) :
    z = H - S
    if z >= np.log(K) :
        return 1. / np.log(K)
    else :
        return 1. / z 
    
#########################################
#  DKL estimation vs Dirichelet params  #
#########################################


def estimate_DKL_at_alpha_beta( a, S, b, H, compACTdiv ):
    '''
    It returns D_KL.
    '''
    
    # loading parameters from Divergence Compact        
    N_A, N_B = compACTdiv.N_A, compACTdiv.N_B
    nn_A, nn_B, ff = compACTdiv.nn_A, compACTdiv.nn_B, compACTdiv.ff
    K = compACTdiv.K
    
    # DKL computation
    temp = ff.dot( (nn_A+a) * ( D_polyGmm(0, N_B+K*b, nn_B+b) - D_polyGmm(0, N_A+K*a+1, nn_A+a+1) ) ) 
    DKL_ab_times_sigma = mp.fdiv( temp, N_A+K*a )    
            
    return DKL_ab_times_sigma

def estimate_DKL2_at_alpha_beta( a, S, b, H, compACTdiv ) :
    '''
    It returns D_KL2.
    '''
    
    # loading parameters from Divergence Compact        
    N_A, N_B, K = compACTdiv.N_A, compACTdiv.N_B, compACTdiv.K
    nn_A, nn_B, ff = compACTdiv.nn_A, compACTdiv.nn_B, compACTdiv.ff
    
    omega_diag = (nn_A+a+1) * (nn_A+a)
    omega_mtx = ( (nn_A+a)[:,None] * (nn_A+a) )
    
    '''
    squared term : q_i q_j ln(q_i) ln(q_j)
    '''
    
    # single sum term
    Ss_Term1 = np.power(D_polyGmm(0, nn_A+a+2, N_A+K*a+2), 2) + D_polyGmm(1, nn_A+a+2, N_A+K*a+2)
    single_sum = Ss_Term1
    
    # double sum term 
    Ds_Term1 = D_polyGmm(0, nn_A+a+1, N_A+K*a+2)[:,None] * D_polyGmm(0, nn_A+a+1, N_A+K*a+2) - polygamma(1, N_A+K*a+2)
    double_sum = Ds_Term1

    '''
    squared term : q_i q_j ln(q_i) ln(t_j)
    '''

    # single sum term
    Ss_Term2 = D_polyGmm(0, nn_A+a+2, N_A+K*a+2) * D_polyGmm(0, nn_B+b, N_B+K*b)
    single_sum -= 2 * Ss_Term2
    
    # double sum term 
    Ds_Term2 = D_polyGmm(0, nn_A+a+1, N_A+K*a+2)[:,None] * D_polyGmm(0, nn_B+b, N_B+K*b)
    double_sum -= 2 * Ds_Term2

    '''
    squared term : q_i q_j ln(t_i) ln(t_j)
    '''
   
    # single sum term
    Ss_Term3 = np.power(D_polyGmm(0, nn_B+b, N_B+K*b), 2) + D_polyGmm(1, nn_B+b, N_B+K*b)
    single_sum += Ss_Term3
    
    # double sum term 
    Ds_Term3 = D_polyGmm(0, nn_B+b, N_B+K*b)[:,None] * D_polyGmm(0, nn_B+b, N_B+K*b) - polygamma(1, N_B+K*b)
    double_sum += Ds_Term3
    
    '''
    multiply omega matrix
    '''
    
    single_sum = np.multiply( single_sum, omega_diag )
    double_sum = np.multiply( double_sum, omega_mtx )
    
    '''
    sum using frequencies
    '''
    
    DKL2_ab = ff.dot( single_sum - double_sum.diagonal() + double_sum.dot(ff) )
    DKL2_ab = mp.fdiv( DKL2_ab, mp.fmul(N_A+K*a+1, N_A+K*a) ) 
    
    return DKL2_ab

    