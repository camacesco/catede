#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) April 2023 Francesco Camaglia, LPENS 

    Following the architecture of J. Hausser and K. Strimmer : 
    https://strimmerlab.github.io/software/entropy/  
'''

import numpy as np
from scipy.special import comb, entr
from .bayesian_calculus import optimal_polya_param
from .nsb.shannon import main as _shannon_nsb_est
from .nsb.simpson import main as _simpson_nsb_est
from .dirichlet_multinomial import D_diGmm
import warnings 

_method_List_ = [
    "naive",
    "cat", "categorical",
    "max_evidence",
    "NSB", "Nemenmann-Shafee-Bialek",
    "CS", "Chao-Shen", 
    "Je", "Jeffreys",
    "MM", "Miller-Madow", 
    "La", "Laplace", 
    "Tr", "minimax", "Trybula", 
    "Pe", "Perks",
    "SG", "Schurmann-Grassberger", 
]

_which_List_ = ["Shannon", "Simpson"]

# unit of the logarithm
_unit_Dict_ = {
    "n": 1., "ln": 1., "default": 1.,
    "2": 1./np.log(2), "log2": 1./np.log(2),
    "10": 1./np.log(10), "log10": 1./np.log(10),
}

#############
#  ALIASES  #
#############

def shannon_operator( x, y=None ) :
    ''' - x * log( x ) '''
    return entr(x)

def simpson_operator( x ) :
    ''' x^2 '''
    return np.power(x,2)

#################
#  SWITCHBOARD  #
#################

def switchboard( compExp, method="naive", which="Shannon", unit="default", **kwargs ):
    '''.'''

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
    if method in ["naive"] :    
        estimate = naive(compExp, which=which)
        
    elif method in ["NSB", "Nemenman-Shafee-Bialek"]:   
        if which == "Shannon" :
            estimate = _shannon_nsb_est(compExp, **kwargs)
        elif which == "Simpson" :
            estimate = _simpson_nsb_est(compExp, **kwargs )
        else :
            raise IOError("FIXME: place holder.")
        
    elif method in ["MM", "Miller-Madow"]:  
        estimate = miller_madow(compExp, which=which)
        
    elif method in ["CS", "Chao-Shen"] :        
        estimate = chao_shen(compExp, which=which)

    elif method in ["SG", "Schurmann-Grassberger"] :
        estimate = schurmann_grassberger(compExp, which=which)

    elif method in ["max_evidence"] :
        estimate = dirichlet_multinomial_expected_value(compExp, which=which, **kwargs)   

    elif method in ["categorical"] :
        estimate = dirichlet_multinomial_expected_value(compExp, param=1, which=which, **kwargs)   

    elif method in ["La", "Laplace"] :
        a = 1.
        estimate = dirichlet_multinomial_pseudo_count(compExp, a, which=which)

    elif method in ["Je", "Jeffreys"] :
        a = 0.5
        estimate = dirichlet_multinomial_pseudo_count(compExp, a, which=which)

    elif method in ["Pe", "Perks"]:
        a = 1. / compExp.Kobs
        estimate = dirichlet_multinomial_pseudo_count(compExp, a, which=which)
        
    elif method in ["Tr", "Trybula", "minimax"]:
        a = np.sqrt( compExp.N ) / compExp.K
        estimate = dirichlet_multinomial_pseudo_count(compExp, a, which=which)

    else:
        raise IOError("Unkown method. Please choose `method` amongst :", _method_List_)
    
    '''
    elif method in ["Di", "Dirichlet"] :
        if "a" not in kwargs :
            a = None
            #warnings.warn("Dirichlet parameter `a` set to optimal.")
        else :
            a = kwargs['a']
        estimate = dirichlet_multinomial_pseudo_count( compExp, a=None, which=which, **kwargs ) 
    '''

    return unit_conv * estimate
###

#####################
#  NAIVE ESTIMATOR  #
#####################

def naive( compExp, which="Shannon" ):
    '''Entropy estimation (naive).'''

    # loading parameters from compExp 
    N = compExp.N
    # delete 0 counts 
    gtr0mask = compExp.nn > 0
    nn, ff = compExp.nn[gtr0mask], compExp.ff[gtr0mask]

    hh = nn / N

    if which == "Shannon" :
        output = ff.dot(shannon_operator(hh))

    elif which == "Simpson" :
        output = ff.dot(simpson_operator(hh))

    else :
        raise IOError("FIXME: place holder.")

    return np.array( output )
###

############################
#  MILLER MADOW ESTIMATOR  #
############################

def miller_madow( compExp, which="Shannon", ): 
    '''Entropy estimation with Miller-Madow bias correction.
    
    ref:
    Miller, G. Note on the bias of information estimates. 
    Information Theory in Psychology: Problems and Methods, 95-100 (1955). '''

    
    # loading parameters from compExp 
    N, K = compExp.N, compExp.K

    if which == "Shannon" :
        output = naive(compExp, which="Shannon") + 0.5 * (K - 1) / N

    else :
        raise IOError("FIXME: place holder.")

    return np.array( output )
###


#####################################
#  SCHURMANN-GRASSBERGER ESTIMATOR  #
#####################################

def schurmann_grassberger(compExp, which="Shannon",): 
    '''Entropy estimation with Schurmann-Grassberger method.
    
    ref:
    Schürmann, T. Bias analysis in entropy estimation. 
    J. Phys. A: Math. Gen. 37, L295 (2004).
    '''
    
    # loading parameters from compExp 
    N = compExp.N
    nn, ff = compExp.nn, compExp.ff
    # delete 0 counts (if present they are at position 0)
    if 0 in nn : nn, ff = nn[1:], ff[1:]    

    if which == "Shannon" :
        output = ff.dot(nn * D_diGmm(N, nn)) / N

    else :
        raise IOError("FIXME: place holder.")

    return np.array(output)
###

#########################
#  CHAO SHEN ESTIMATOR  #
#########################

def _good_turing_coverage(nn, ff) :
    '''Good-Turing frequency estimation with Zhang-Huang formulation.
    
    ref:
    Zhang, Z. & Huang, H. Turing's formula revisited*.
    Journal of Quantitative Linguistics 14, 222-241 (2007).
    '''

    N = ff.dot(nn)
    # Check for the pathological case of all singletons (to avoid coverage = 0)
    # i.e. nn = [1], which means ff = [N]
    if ff[np.where(nn == 1)[0]] == N :  
        # this correpsonds to the correction ff_1=N |==> ff_1=N-1
        GoodTuring = (N - 1) / N                                  
    else :
        sign = np.power(-1, nn + 1)
        binom = 1. / comb(N, nn)
        GoodTuring = ff.dot(sign * binom)
        
    return 1. - GoodTuring

def chao_shen(compExp, which="Shannon"):
    '''Entropy estimation with Chao-Shen model (coverage adjusted estimator).

    ref: 
    Chao, A. & Shen, T.-J. Nonparametric estimation of Shannon's index of diversity when there are unseen species in sample. 
    Environmental and Ecological Statistics 10, 429-443 (2003).
    '''

    # loading parameters from compExp 
    N, nn, ff = compExp.N, compExp.nn, compExp.ff
    # delete unseen categories information, i.e. 0 counts 
    mask = np.where(nn > 0)[0]
    nn, ff = nn[mask], ff[mask]        
   
    # coverage adjusted empirical frequencies  
    C = _good_turing_coverage(nn, ff)                       
    p_vec = C * nn / N                         
    # probability to see a bin (specie) in the sample         
    lambda_vec = 1. - np.power(1. - p_vec, N)         
    
    if which == "Shannon" :
        output = ff.dot(shannon_operator(p_vec) / lambda_vec)

    elif which == "Simpson" :
        output = ff.dot(simpson_operator(p_vec) / lambda_vec)

    else :
        raise IOError("FIXME: place holder.")
            
    return np.array( output )
###

##########################
#  DIRICHELET ESTIMATOR  #
##########################

def dirichlet_multinomial_pseudo_count(compExp, a=None, which="Shannon"):
    '''Entropy estimation with Dirichlet-multinomial pseudocount model.

    Parameters
    ----------  

    a: float
        concentration parameter
    '''

    # loading parameters from compExp 
    N, K = compExp.N, compExp.K
    nn, ff = compExp.nn, compExp.ff

    if a == None :
        a = optimal_polya_param(compExp)
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
        output = ff.dot(shannon_operator(hh_a))

    elif which == "Simpson" :  
        output = ff.dot(simpson_operator(hh_a))

    else :
        raise IOError("Unknown method for the chosen quantity.")

    return np.array( output )
###

##################
#  MAX EVIDENCE  #
##################

def dirichlet_multinomial_expected_value(compExp, param=None, which="Shannon", error=False):
    '''Expected entropy with Dirichlet-multinomial at maximum evidence.

    '''
    if param == None :
        a_star = optimal_polya_param(compExp)
    else :
        a_star = np.float64(param)
        if a_star <= 0 :
            raise IOError('The quantity `param` should be a scalar >0.')

    if which == "Shannon" :
        output = compExp.shannon( a_star )
        if error == True :
            tmp = compExp.squared_shannon( a_star )
            output = [ output, np.sqrt(tmp - output**2) ]

    elif which == "Simpson" :  
        output = compExp.simpson( a_star )
        if error == True :
            tmp = compExp.squared_simpson( a_star )
            output = [ output, np.sqrt(tmp - output**2) ]
    else :
        raise IOError("Unknown method for the chosen quantity.")

    return np.array( output )
###