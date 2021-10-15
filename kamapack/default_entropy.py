#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) October 2021 Francesco Camaglia, LPENS 
    Following the architecture of J. Hausser and K. Strimmer : https://strimmerlab.github.io/software/entropy/  
'''

import numpy as np
from scipy.special import comb

from kamapack import nsb_entropy

# loagirthm unit
_unit_Dict_ = { "log2": 1. / np.log(2), "ln": 1., "log10": 1. / np.log(10) }


#################
#  SWITCHBOARD  #
#################

def switchboard( compACT, method, unit=None, **kwargs ):

    # loading units
    if unit in _unit_Dict_.keys( ) :
        unit_conv = _unit_Dict_[ unit ]
    else:
        raise IOError("Unknown unit, please choose amongst ", _unit_Dict_.keys( ) )

    # choosing entropy estimation method
    if method == "ML":                          # Maximum Likelihood
        shannon_estimate = maxlike( compACT )
    elif method == "MM":                        # Miller Madow
        shannon_estimate = MillerMadow( compACT )
    elif method == "CS":                        # Chao Shen       
        shannon_estimate = ChaoShen( compACT )
    elif method == "Jeffreys":                  # Jeffreys
        a = 0.5
        shannon_estimate = Dirichlet( compACT, a )
    elif method == "Laplace":                   # Laplace
        a = 1.
        shannon_estimate = Dirichlet( compACT, a )
    elif method == "SG":                        # Schurmann-Grassberger
        a = 1. / compACT.obs_categ
        shannon_estimate = Dirichlet( compACT, a )
    elif method == "minimax":                   # minimax
        a = np.sqrt( compACT.N ) / compACT.obs_categ
        shannon_estimate = Dirichlet( compACT, a )
    elif method == "NSB":                       # Nemenman Shafee Bialek
        shannon_estimate = nsb.NemenmanShafeeBialek( compACT, **kwargs )
    else:
        raise IOError("The chosen method is unknown.")

    return unit_conv * shannon_estimate
###



##################################
#  MAXIMUM LIKELIHOOD ESTIMATOR  #
##################################

def maxlike( compACT ):
    '''
    Maximum likelihood estimator.
    '''

    # loading parameters from compACT 
    N, nn, ff = compACT.N, compACT.nn, compACT.ff
    # delete 0 counts (if present they are at position 0)
    if 0 in nn : del nn[ 0 ], del ff[ 0 ]                      
    
    shannon_estimate = np.array( np.log(N) - np.dot( ff , np.multiply( nn, np.log(nn) ) ) / N )
    return shannon_estimate
###



############################
#  MILLER MADOW ESTIMATOR  #
############################

def MillerMadow( compACT ): 
    '''
    Miller-Madow estimator.
    '''
    
    # loading parameters from compACT 
    N, Kobs = compACT.N, compACT.Kobs

    shannon_estimate = np.array( maxlike( compACT ) + 0.5 * ( Kobs - 1 ) / N )
    return shannon_estimate 
###



#########################
#  CHAO SHEN ESTIMATOR  #
#########################

def ChaoShen( compACT ):
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
    
    # loading parameters from compACT 
    N, nn, ff = compACT.N, compACT.nn, compACT.ff
    # delete 0 counts (if present they are at position 0)
    if 0 in nn : del nn[ 0 ], del ff[ 0 ]     

    C = __coverage( nn, ff )                            
    p_vec = C * nn / N                                # coverage adjusted empirical frequencies
    lambda_vec = 1. - np.power( 1. - p_vec, N )         # probability to see a bin (specie) in the sample

    shannon_estimate = np.array( - np.dot( ff , p_vec * np.log( p_vec ) / lambda_vec ) )
    return shannon_estimate 
###



##########################
#  DIRICHELET ESTIMATOR  #
##########################

def Dirichlet( compACT, a ):
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

    # loading parameters from compACT 
    N, nn, ff = compACT.N, compACT.nn, compACT.ff

    nn_a = nn + a                                       # counts plus pseudocounts
    N_a = N + a * np.sum( ff )                          # total number of counts plus pseudocounts

    shannon_estimate  = np.array(  np.log( N_a ) - np.dot( ff , nn_a * np.log( nn_a ) ) / N_a )
    return shannon_estimate  
###

