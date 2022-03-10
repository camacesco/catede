#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) January 2022 Francesco Camaglia, LPENS 
    Following the architecture of J. Hausser and K. Strimmer : 
    https://strimmerlab.github.io/software/entropy/  
'''

import numpy as np
from scipy.special import comb
from . import nsb_entropy

# loagirthm unit
_unit_Dict_ = { "ln": 1., "log2": 1./np.log(2), "log10": 1./np.log(10) }

#################
#  SWITCHBOARD  #
#################

def switchboard( compACT, method="naive", unit=None, **kwargs ):

    # loading units
    if unit in _unit_Dict_.keys( ) :
        unit_conv = _unit_Dict_[ unit ]
    else:
        raise IOError("Unknown unit, please choose amongst ", _unit_Dict_.keys( ) )

    # choosing entropy estimation method
    if method in ["naive", "maximum-likelihood"] :    
        shannon_estimate = Naive( compACT )
        
    elif method in ["NSB", "Nemenman-Shafee-Bialek"]:   
        shannon_estimate = nsb_entropy.NemenmanShafeeBialek( compACT, **kwargs )
        
    elif method in ["MM", "Miller-Madow"]:  
        shannon_estimate = MillerMadow( compACT )
        
    elif method in ["CS", "Chao-Shen"] :        
        shannon_estimate = ChaoShen( compACT )
 
    elif method in ["D", "Dirichlet"] :
        if "a" not in kwargs :
            raise IOError("The Dirichlet parameter `a` must be specified.")
        a = kwargs['a']
        shannon_estimate = Dirichlet( compACT, a )       
        
    elif method in ["L", "Laplace", "Bayesian-Laplace"] :
        a = 1.
        shannon_estimate = Dirichlet( compACT, a )

    elif method in ["Jeffreys", "Krichevsky-Trofimov"] :
        a = 0.5
        shannon_estimate = Dirichlet( compACT, a )

    elif method in ["SG", "Schurmann-Grassberger"]:
        a = 1. / compACT.Kobs
        shannon_estimate = Dirichlet( compACT, a )
        
    elif method in ["minimax", "Trybula"]:
        # FIXME: warning
        a = np.sqrt( compACT.N ) / compACT.K
        shannon_estimate = Dirichlet( compACT, a )

    else:
        raise IOError("The chosen method is unknown.")

    return unit_conv * shannon_estimate
###

#####################
#  NAIVE ESTIMATOR  #
#####################

def Naive( compACT ):
    '''Entropy estimation (naive).'''

    # loading parameters from compACT 
    N, nn, ff = compACT.N, compACT.nn, compACT.ff
    # delete 0 counts (if present they are at position 0)
    if 0 in nn : nn, ff = nn[1:], ff[1:]                      
    
    output = np.log(N) - np.dot( ff , np.multiply( nn, np.log(nn) ) ) / N
    return np.array( output )
###

############################
#  MILLER MADOW ESTIMATOR  #
############################

def MillerMadow( compACT ): 
    '''Entropy estimation with Miller-Madow pseudocount model.'''
    
    # loading parameters from compACT 
    N, Kobs = compACT.N, compACT.Kobs

    output = Naive( compACT ) + 0.5 * ( Kobs - 1 ) / N
    return np.array( output )
###

#########################
#  CHAO SHEN ESTIMATOR  #
#########################

def ChaoShen( compACT ):
    '''Entropy estimation with Chao-Shen pseudocount model.'''

    def __coverage( nn, ff ) :
        '''Good-Turing frequency estimation with Zhang-Huang formulation.'''

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
    if 0 in nn : nn, ff = nn[1:], ff[1:]        

    C = __coverage( nn, ff )                            
    p_vec = C * nn / N                                  # coverage adjusted empirical frequencies
    lambda_vec = 1. - np.power( 1. - p_vec, N )         # probability to see a bin (specie) in the sample

    output = - np.dot( ff , p_vec * np.log( p_vec ) / lambda_vec )
    return np.array( output )
###

##########################
#  DIRICHELET ESTIMATOR  #
##########################

def Dirichlet( compACT, a ):
    '''Entropy estimation with Dirichlet-multinomial pseudocount model.

    Parameters
    ----------  

    a: float
        Dirichlet parameter
    '''

    # loading parameters from compACT 
    N, K = compACT.N, compACT.K
    nn, ff = compACT.nn, compACT.ff
    
    # frequencies with pseudocounts
    hh_a = (nn + a) / (N + K * a)      
    
    output = - np.dot( ff , hh_a * np.log( hh_a ) )
    return np.array( output )
###