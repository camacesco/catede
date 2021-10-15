#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) October 2021 Francesco Camaglia, LPENS 
    Following the architecture of J. Hausser and K. Strimmer : https://strimmerlab.github.io/software/entropy/  
'''

import numpy as np
from scipy.special import comb

from kamapack.estimator import nsb_entropy

# loagirthm unit
_unit_Dict_ = { "log2": 1. / np.log(2), "ln": 1., "log10": 1. / np.log(10) }


########################
#  ENTROPY ESTIMATION  #
########################

def _entropy( experiment, method, unit=None, **kwargs ):

    # loading units
    if unit in _unit_Dict_.keys( ) :
        unit_conv = _unit_Dict_[ unit ]
    else:
        raise IOError("Unknown unit, please choose amongst ", _unit_Dict_.keys( ) )

    # choosing entropy estimation method
    if method == "ML":                          # Maximum Likelihood
        shannon_estimate = maxlike( experiment )
    elif method == "MM":                        # Miller Madow
        shannon_estimate = MillerMadow( experiment )
    elif method == "CS":                        # Chao Shen       
        shannon_estimate = ChaoShen( experiment )
    elif method == "Jeffreys":                  # Jeffreys
        a = 0.5
        shannon_estimate = Dirichlet( experiment, a )
    elif method == "Laplace":                   # Laplace
        a = 1.
        shannon_estimate = Dirichlet( experiment, a )
    elif method == "SG":                        # Schurmann-Grassberger
        a = 1. / experiment.obs_categ
        shannon_estimate = Dirichlet( experiment, a )
    elif method == "minimax":                   # minimax
        a = np.sqrt( experiment.N ) / experiment.obs_categ
        shannon_estimate = Dirichlet( experiment, a )
    elif method == "NSB":                       # Nemenman Shafee Bialek
        shannon_estimate = nsb.NemenmanShafeeBialek( experiment, **kwargs )
    else:
        raise IOError("The chosen method is unknown.")

    return unit_conv * shannon_estimate
###



##################################
#  MAXIMUM LIKELIHOOD ESTIMATOR  #
##################################

def maxlike( experiment ):
    '''
    Maximum likelihood estimator.
    '''

    # loading parameters from experiment 
    N = experiment.tot_counts                           # total number of counts
    temp = experiment.counts_hist.copy()
    if 0 in temp : del temp[ 0 ]                        # delete locally 0 counts
    nn = temp.index.values                              # counts
    ff = temp.values                                    # recurrency of counts
    
    shannon_estimate = np.array( np.log(N) - np.dot( ff , np.multiply( nn, np.log(nn) ) ) / N )
    return shannon_estimate
###



############################
#  MILLER MADOW ESTIMATOR  #
############################

def MillerMadow( experiment ): 
    '''
    Miller-Madow estimator.
    '''
    
    # loading parameters from experiment 
    N = experiment.tot_counts           # total number of counts
    Kobs = experiment.obs_n_categ       # number of bins with non-zero counts: obs_categ

    shannon_estimate = np.array( maxlike( experiment ) + 0.5 * ( Kobs - 1 ) / N )
    return shannon_estimate 
###



#########################
#  CHAO SHEN ESTIMATOR  #
#########################

def ChaoShen( experiment ):
    '''
    Compute Chao-Shen (2003) entropy estimator 
    WARNING!: TO BE CHECKED
    '''

    def __coverage( nn, ff ) :
        '''
        Good-Turing frequency estimation with Zhang-Huang formulation
        '''
        N = np.dot( nn, ff )
        # Check for the pathological case of all singletons (to avoid coverage = 0)
        # i.e. nn = [1], which means ff = [N]
        if ff[ np.where( nn == 1 )[0] ] == N :  
            # this correpsonds to the correction ff_1=N |==> ff_1=N-1
            GoodTuring = ( N - 1 ) / N                                  
        else :
            sign = np.power( -1, nn + 1 )
            binom = list( map( lambda k : 1. / comb(N,k), nn ) )
            GoodTuring = np.sum( sign * binom * ff )
            
        return 1. - GoodTuring
    ###
    
    # loading parameters from experiment
    N = experiment.tot_counts                           # total number of counts
    temp = experiment.counts_hist.copy()
    if 0 in temp : del temp[ 0 ]                        # delete locally 0 counts
    nn = temp.index.values                              # counts
    ff = temp.values                                    # recurrency of counts

    C = __coverage( nn, ff )                            
    p_vec = C * nn / N                                # coverage adjusted empirical frequencies
    lambda_vec = 1. - np.power( 1. - p_vec, N )         # probability to see a bin (specie) in the sample

    shannon_estimate = np.array( - np.dot( ff , p_vec * np.log( p_vec ) / lambda_vec ) )
    return shannon_estimate 
###



##########################
#  DIRICHELET ESTIMATOR  #
##########################

def Dirichlet( experiment, a ):
    '''
    Estimate entropy based on Dirichlet-multinomial pseudocount model.
    a:  pseudocount per bin
    a=0          :   empirical estimate
    a=1          :   Laplace
    a=1/2        :   Jeffreys
    a=1/M        :   Schurmann-Grassberger  (M: number of bins)
    a=sqrt(N)/M  :   minimax
    WARNING!: TO BE CHECKED
    '''

    # loading parameters from experiment 
    N = experiment.tot_counts                           # total number of counts
    nn = temp.index.values                              # counts
    ff = temp.values                                    # recurrency of counts

    nn_a = nn + a                                       # counts plus pseudocounts
    N_a = N + a * np.sum( ff )                          # total number of counts plus pseudocounts

    shannon_estimate  = np.array(  np.log( N_a ) - np.dot( ff , nn_a * np.log( nn_a ) ) / N_a )
    return shannon_estimate  
###

