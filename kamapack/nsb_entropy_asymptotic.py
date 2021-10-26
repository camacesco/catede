#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) October 2021 Francesco Camaglia, LPENS 
    WARNING!: to be checked 
'''

import numpy as np
from scipy.special import polygamma
from kamapack.nsb_entropy import Delta_polyGmm

###
def NSB_low_coinc_est(experiment):

    # loading parameters from experiment 
    N = experiment.tot_counts                           # total number of counts
    D = N - experiment.obs_n_categ
    
    # expansion for small numbers of coincidences
    C_gamma = - polygamma( 0, 1 ) 
    S = C_gamma - np.log(2) + 2 * np.log(N) - polygamma(0,D)
    S_devStd = np.sqrt( polygamma( 1, D ) )

    return np.array([S, S_devStd])
###



### 
def NSB_sad_pnt_est( experiment ):

    # loading parameters from experiment 
    N = experiment.tot_counts                           # total number of counts
    K = experiment.usr_n_categ                          # number of given categories
    Kobs = experiment.obs_n_categ                       # number of observed categories
    nn = temp.index.values                              # counts
    ff = temp.values                                    # recurrency of counts
    
    # saddle-point estimation of the beta
    dd = 1 - Kobs / N
    b_m1 = 0.5 * ( N - 1 ) / N
    b_0 = ( 1 - 2*N ) / ( 3*N )
    b_1 = ( N - 1 - 2 / N ) / ( 9 * ( N - 1 ) )
    ki_0 = N * ( b_m1 / dd + b_0 + b_1 * dd )

    # evaluation of the integral at the saddle
    obsMask = nn > 1
    ki_Z = (Kobs/ki_0**2+Delta_polyGmm(1,ki_0+N,ki_0))
    ki_1 = np.dot(ff[obsMask],(Delta_polyGmm(0,nn[obsMask],1)))/ki_Z
    ki_2 = ((Kobs/ki_0**3+0.5*Delta_polyGmm(2,ki_0,ki_0+N))*ki_1**2+ki_0*np.dot(ff[obsMask],(Delta_polyGmm(1,nn[obsMask],1))))/ki_Z
    beta_sad_pnt = ki_0/K+ki_1/K**2+ki_2/K**3
        
    trash, S, S2 = nsb._estimator_beta( beta_sad_pnt, experiment )    
    S_devStd = np.sqrt( S2 - np.power( S ,2 ) )
    
    return np.array([S, S_devStd])
###
