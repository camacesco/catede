#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    *** Default Divergence ***
    Copyright (C) January 2024 Francesco Camaglia, LPENS 
'''

import warnings
import numpy as np 
from copy import deepcopy
from .dpm.kullback_leibler import main as _DKL_dpm_estimator
from .dpm.squared_hellinger import main as _DH2_dpm_estimator
from .dpm.symmetrized_KL import main as _symmDKL_dpm_estimator
from .default_entropy import _unit_Dict_
from .bayesian_calculus import optimal_polya_param
from scipy.special import rel_entr
from .dirichlet_multinomial import D_diGmm

_method_List_ = [
    "naive", 
    "DC", "categorical",
    "DP", "max_evidence",
    "DPM", 
    "Zh", "Zhang-Grabchak",
    "Je", "Jeffreys", 
    "La", "Laplace", 
    "Tr", "Trybula", "minimax", 
    "Pe", "Perks",
]

_which_List_ = ["squared-Hellinger", "Jensen-Shannon", "Kullback-Leibler", "symmetrized-KL"]

#############
#  ALIASES  #
#############

def KullbackLeibler_oper(x, y) :
    ''' x * log(x/y) '''
    return rel_entr(x, y)

def symmetrized_KL_oper(x, y) :
    ''' 0.5 * ( x * log(x/y) + y * log(y/x)) '''
    return 0.5 * (rel_entr(x,y) + rel_entr(y,x))

def JensenShannon_oper(x, y) :
    '''   '''
    mm = 0.5 * (x + y)
    return 0.5 * (KullbackLeibler_oper(x, mm) + KullbackLeibler_oper(y, mm))

def Bhattacharyya_oper(x, y) :
    ''' sqrt( x * y ) '''
    return np.sqrt(np.multiply(x, y)) 

#################
#  SWITCHBOARD  #
#################

def switchboard(cpct_div, method="naive", which="Kullback-Leibler", unit="default", **kwargs):

    # check which 
    if which not in _which_List_ :
        raise IOError("Unkown divergence. Please choose `which` amongst :", _which_List_ )
    
    # loading units
    if which in ["Jensen-Shannon", "Kullback-Leibler", "symmetrized-KL"] :
        if unit not in _unit_Dict_.keys( ) :
            warnings.warn( "Please choose `unit` amongst :", _unit_Dict_.keys( ), ". Falling back to default." )
        unit_conv = _unit_Dict_.get( unit, _unit_Dict_["default"] )
    else :
        unit_conv = 1
        
    # choosing entropy estimation method
    if method in ["naive"] :  
        divergence_estimate = naive(cpct_div, which=which)
    
    elif method in ["DPM"] :       
        if which == "Jensen-Shannon" :
            raise IOError(f"Unknown method `{method}` for {which}.")
        elif which == "Kullback-Leibler" :
            divergence_estimate = _DKL_dpm_estimator(cpct_div, **kwargs)
        elif which == "squared-Hellinger" :
            divergence_estimate = _DH2_dpm_estimator(cpct_div, **kwargs)
        elif which == "symmetrized-KL" :
            divergence_estimate = _symmDKL_dpm_estimator(cpct_div, **kwargs)
    
    elif method in ["Zh", "Zhang-Grabchak"] :
        if which in ["Kullback-Leibler", "symmetrized-KL"] :
            divergence_estimate = zhang(cpct_div, which=which)
        else :
            raise IOError(f"Unknown method `{method}` for {which}.")
        
    elif method in ["DP", "max_evidence"] :
        divergence_estimate = dirichlet_multinomial_expected_value(cpct_div, which=which, **kwargs)

    elif method in ["DC", "categorical"] :
        divergence_estimate = dirichlet_multinomial_expected_value(cpct_div, params=[1,1], which=which, **kwargs)

    elif method in ["Je", "Jeffreys"] :
        params = np.array([0.5, 0.5])
        divergence_estimate = dirichlet_multinomial_pseudo_count(cpct_div, params, which=which)
    
    elif method in ["La", "Laplace"] :
        params = np.array([1., 1.])
        divergence_estimate = dirichlet_multinomial_pseudo_count(cpct_div, params, which=which)
        
    elif method in ["Tr", "Trybula", "minimax"]:  
        a = np.sqrt( cpct_div.N_1 ) / cpct_div.compact_1.K
        b = np.sqrt( cpct_div.N_2 ) / cpct_div.compact_2.K
        params = np.array([a, b])
        divergence_estimate = dirichlet_multinomial_pseudo_count(cpct_div, params, which=which)
     
    elif method in ["Pe", "Perks"]:
        a = 1. / cpct_div.compact_1.Kobs
        b = 1. / cpct_div.compact_2.Kobs
        params = np.array([a, b])
        divergence_estimate = dirichlet_multinomial_pseudo_count(cpct_div, params, which=which)

    else:
        raise IOError(f"Unkown method `{method}`.\n Please choose amongst :\n", _method_List_)
    
    return unit_conv * divergence_estimate
###

###########
#  NAIVE  #
###########

def naive(cpct_div, which="Kullback-Leibler") :
    '''Estimation of divergence with frequencies of observed categories.
    All counts equal to 0 are not considered.
    '''
    
    # loading parameters from cpct_div 
    N_1, N_2 = cpct_div.N_1, cpct_div.N_2
    # delete 0 counts
    gtr0mask = np.logical_and( cpct_div.nn_1 > 0, cpct_div.nn_2 > 0 )
    nn_1, nn_2, ff = cpct_div.nn_1[gtr0mask], cpct_div.nn_2[gtr0mask], cpct_div.ff[gtr0mask]
    
    hh_1 = nn_1 / N_1                  # frequencies
    hh_2 = nn_2 / N_2                  # frequencies
    
    if which == "Jensen-Shannon" :
        output = ff.dot(JensenShannon_oper(hh_1, hh_2))

    elif which == "Kullback-Leibler" :                       
        output = ff.dot(KullbackLeibler_oper(hh_1, hh_2))

    elif which == "symmetrized-KL" :                       
        output = ff.dot(symmetrized_KL_oper(hh_1, hh_2))

    elif which == "squared-Hellinger" :  
        output = 1 - ff.dot(Bhattacharyya_oper(hh_1, hh_2))

    else :
        raise IOError("Unknown method for the chosen divergence.")

    return np.array( output )


#####################
#  ZHANG ESTIMATOR  #
#####################

def zhang( cpct_div, which="Kullback-Leibler", CPU_Count=None) :
    ''' Z estimator for the DKL.

    Zhang, Z. & Grabchak, M. Nonparametric Estimation of Küllback-Leibler Divergence.
    Neural Computation 26, 2570-2593 (2014).

    resummed using :
    Schürmann T. A Note on Entropy Estimation. 
    Neural computation, 27(10), 2097-2106 (2015).
    '''

    # delete 0 counts in system 1 since those term contribution is 0
    mask = cpct_div.nn_1 > 0
    nn_1, nn_2, ff = cpct_div.nn_1[mask], cpct_div.nn_2[mask], cpct_div.ff[mask]
    N_1, N_2 = cpct_div.N_1, cpct_div.N_2

    if which == "Kullback-Leibler" : 
        output = ff.dot(nn_1 * (D_diGmm(N_2+1, nn_2+1) - D_diGmm(N_1, nn_1))) / N_1

    elif which == "symmetrized-KL" :  
        output = 0.5 * ff.dot(nn_1 * (D_diGmm(N_2+1, nn_2+1) - D_diGmm(N_1, nn_1))) / N_1
        mask = cpct_div.nn_2 > 0
        nn_1, nn_2, ff = cpct_div.nn_1[mask], cpct_div.nn_2[mask], cpct_div.ff[mask]
        output += 0.5 * ff.dot(nn_2 * (D_diGmm(N_1+1, nn_1+1) - D_diGmm(N_2, nn_2))) / N_2

    else :
        raise IOError("Unknown method for the chosen divergence.")

    return np.array( output )

##########################
#  DIRICHLET ESTIMATOR  #
##########################

def dirichlet_multinomial_pseudo_count( cpct_div, params=None, which="Kullback-Leibler"):
    '''Estimation of divergence with Dirichlet-multinomial pseudocount model.'''
    # check options
    if np.any(np.array(params) == None) :
        a = optimal_polya_param(cpct_div.compact_1)
        b = optimal_polya_param(cpct_div.compact_2)
    else :
        try:
            a = np.float64(params[0])
        except :
            raise IOError('The concentration parameter `a` must be a scalar.')
        if a < 0 :
            raise IOError('The concentration parameter `a` must greater than 0.')
        try:
            b = np.float64(params[1])
        except :
            raise IOError('The concentration parameter `b` must be a scalar.')
        if b < 0 :
            raise IOError('The concentration parameter `b` must greater than 0.')

    # loading parameters from cpct_div 
    N_1, N_2, K = cpct_div.N_1, cpct_div.N_2, cpct_div.K
    nn_1, nn_2, ff = cpct_div.nn_1, cpct_div.nn_2, cpct_div.ff

    hh_1_a = (nn_1 + a) / (N_1 + K*a)     # frequencies with pseudocounts
    hh_2_b = (nn_2 + b) / (N_2 + K*b)     # frequencies with pseudocounts

    if which == "Jensen-Shannon" :
        output = ff.dot(JensenShannon_oper(hh_1_a, hh_2_b))

    elif which == "Kullback-Leibler" :                               
        output = ff.dot(KullbackLeibler_oper(hh_1_a, hh_2_b))

    elif which == "symmetrized-KL" :                       
        output = ff.dot(symmetrized_KL_oper(hh_1_a, hh_2_b))

    elif which == "squared-Hellinger" :  
        output = 1 - ff.dot(Bhattacharyya_oper(hh_1_a, hh_2_b))

    else :
        raise IOError("Unknown method for the chosen quantity.")

    return np.array( output )
###


##########################
#  MAX_EVIDENCE  #
##########################

def dirichlet_multinomial_expected_value(cpct_div, params=None, which="Kullback-Leibler", error=False,):
    '''Expected value of the divergence under Dirichlet-multinomial.'''

    if np.any(np.array(params) == None) :
        a = optimal_polya_param(cpct_div.compact_1)
        b = optimal_polya_param(cpct_div.compact_2)
    else :
        try:
            a = np.float64(params[0])
        except :
            raise IOError('The concentration parameter `a` must be a scalar.')
        if a < 0 :
            raise IOError('The concentration parameter `a` must greater than 0.')
        try:
            b = np.float64(params[1])
        except :
            raise IOError('The concentration parameter `b` must be a scalar.')
        if b < 0 :
            raise IOError('The concentration parameter `b` must greater than 0.')
        
    if which == "Kullback-Leibler" :                               
        output = cpct_div.kullback_leibler(a, b)
        if error == True :
            tmp = cpct_div.squared_kullback_leibler(a, b)
            output = [output, np.sqrt(tmp - output**2)]

    elif which == "symmetrized-KL" :  
        cpct_div_rev = deepcopy(cpct_div)         
        output = 0.5 * ( cpct_div.kullback_leibler(a, b) + cpct_div_rev.kullback_leibler(b, a))
        if error == True :            
            raise SystemError(f"Error of {which} to be coded...")

    elif which == "squared-Hellinger" :  
        tmp1 = cpct_div.bhattacharyya(a, b)
        output = 1 - tmp1
        if error == True :
            tmp2 = cpct_div.squared_bhattacharyya(a, b)
            output = [output, np.sqrt(tmp2 - tmp1**2)]

    else :
        raise IOError("Unknown method for the chosen quantity.")

    return np.array(output)
###