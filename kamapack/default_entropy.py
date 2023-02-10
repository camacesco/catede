#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) October 2022 Francesco Camaglia, LPENS 
    Following the architecture of J. Hausser and K. Strimmer : 
    https://strimmerlab.github.io/software/entropy/  
'''

import numpy as np
from scipy.special import comb, entr
from .new_calculus import optimal_dirichlet_param
from .nsb_shannon_entropy import main as _Shannon_est
#from .nsb_Simpson import Simpson_NSB # FLAG 
import warnings 

_method_List_ = [
    "naive", "maximum-likelihood",
    "NSB", "Nemenmann-Shafee-Bialek",
    "CS", "Chao-Shen",
    "Di", "Dirichlet", 
    "Je", "Jeffreys", "Krichevsky-Trofimov", 
    "MM", "Miller-Madow", 
    "La", "Laplace", 
    "Tr", "minimax", "Trybula", 
    "Pe", "Schurmann-Grassberger", "Perks",
]

_which_List_ = ["Shannon"] # ["Shannon", "Simpson"] # FLAG

# unit of the logarithm
_unit_Dict_ = {
    "n": 1., "ln": 1., "default": 1.,
    "2": 1./np.log(2), "log2": 1./np.log(2),
    "10": 1./np.log(10), "log10": 1./np.log(10),
}

#############
#  ALIASES  #
#############

def Shannon_oper( x, y=None ) :
    ''' - x * log( x ) '''
    return entr(x)

def Simpson_oper( x ) :
    ''' x^2 '''
    return np.power(x,2)

#################
#  SWITCHBOARD  #
#################

def switchboard( compACT, method="naive", which="Shannon", unit="default", **kwargs ):

    # check which 
    if which not in _which_List_ :
        raise IOError("Unkown divergence. Please choose `which` amongst :", _which_List_ )

    # loading units
    if which in ["Shannon"] :
        if unit not in _unit_Dict_.keys( ) :
            warnings.warn( "Please choose `unit` amongst :", _unit_Dict_.keys( ), ". Falling back to default." )
        unit_conv = _unit_Dict_.get( unit, _unit_Dict_["default"] )
    else :
        unit_conv = 1

    # choosing entropy estimation method
    if method in ["naive", "maximum-likelihood"] :    
        estimate = Naive( compACT, which=which, **kwargs )
        
    elif method in ["NSB", "Nemenman-Shafee-Bialek"]:   
        if which == "Shannon" :
            estimate = _Shannon_est( compACT, **kwargs )
        #elif which == "Simpson" :
        #    estimate = Simpson_NSB( compACT, **kwargs )
        else :
            raise IOError("FIXME: place holder.")
        
    elif method in ["MM", "Miller-Madow"]:  
        estimate = MillerMadow( compACT, which=which, **kwargs )
        
    elif method in ["CS", "Chao-Shen"] :        
        estimate = ChaoShen( compACT, which=which, **kwargs )
 
    elif method in ["Di", "Dirichlet"] :
        if "a" not in kwargs :
            a = "optimal"
            #warnings.warn("Dirichlet parameter `a` set to optimal.")
        else :
            a = kwargs['a']
        estimate = Dirichlet( compACT, a, which=which, **kwargs )       
        
    elif method in ["La", "Laplace", "Bayesian-Laplace"] :
        a = 1.
        estimate = Dirichlet( compACT, a, which=which, **kwargs )

    elif method in ["Je", "Jeffreys", "Krichevsky-Trofimov"] :
        a = 0.5
        estimate = Dirichlet( compACT, a, which=which, **kwargs )

    elif method in ["Pe", "Perks", "SG", "Schurmann-Grassberger"]:
        a = 1. / compACT.Kobs
        estimate = Dirichlet( compACT, a, which=which, **kwargs )
        
    elif method in ["Tr", "Trybula", "mm", "minimax"]:
        a = np.sqrt( compACT.N ) / compACT.K
        estimate = Dirichlet( compACT, a, which=which, **kwargs )

    else:
        raise IOError("Unkown method. Please choose `method` amongst :", _method_List_ )

    return unit_conv * estimate
###

#####################
#  NAIVE ESTIMATOR  #
#####################

def Naive( compACT, which ):
    '''Entropy estimation (naive).'''

    # loading parameters from compACT 
    N, nn, ff = compACT.N, compACT.nn, compACT.ff
    # delete 0 counts (if present they are at position 0)
    if 0 in nn : nn, ff = nn[1:], ff[1:]         

    hh = nn / N

    if which == "Shannon" :
        output = np.dot( ff , Shannon_oper( hh ) )

    #elif which == "Simpson" :
    #    output = np.dot( ff , Simpson_oper( hh ) )

    return np.array( output )
###

############################
#  MILLER MADOW ESTIMATOR  #
############################

def MillerMadow( compACT, which, ): 
    '''Entropy estimation with Miller-Madow pseudocount model.'''
    
    # loading parameters from compACT 
    N, Kobs = compACT.N, compACT.Kobs

    if which == "Shannon" :
        output = Naive( compACT, which="Shannon" ) + 0.5 * ( Kobs - 1 ) / N
    else :
        raise IOError("FIXME: place holder.")

    return np.array( output )
###

#########################
#  CHAO SHEN ESTIMATOR  #
#########################

def ChaoShen( compACT, which="Shannon" ):
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
    
    if which == "Shannon" :
        output = np.dot( ff, Shannon_oper( p_vec ) / lambda_vec )
    else :
        raise IOError("FIXME: place holder.")
            
    return np.array( output )
###

##########################
#  DIRICHELET ESTIMATOR  #
##########################

def Dirichlet( compACT, a, which="Shannon", ):
    '''Entropy estimation with Dirichlet-multinomial pseudocount model.

    Parameters
    ----------  

    a: float
        concentration parameter
    '''

    # loading parameters from compACT 
    N, K = compACT.N, compACT.K
    nn, ff = compACT.nn, compACT.ff

    if a == "optimal" :
        a = optimal_dirichlet_param(compACT)
    else :
        try:
            a = np.float64(a)
        except :
            raise IOError('The concentration parameter `a` must be a scalar.')
        if a < 0 :
            raise IOError('The concentration parameter `a` must greater than 0.')

    # frequencies with pseudocounts
    hh_a = (nn + a) / (N + K * a)      
    
    if which == "Shannon" :
        output = np.dot( ff, Shannon_oper( hh_a ) )

    #elif which == "Simpson" :  
    #    output = np.dot( ff, Simpson_oper( hh_a ) )

    else :
        raise IOError("Unknown method `Dirichlet` for the chosen quantity.")

    return np.array( output )
###