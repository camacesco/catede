#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    *** Default Divergence ***
    Copyright (C) January 2022 Francesco Camaglia, LPENS 
    Following the architecture of J. Hausser and K. Strimmer : https://strimmerlab.github.io/software/entropy/
'''

import numpy as np 
from .cmw_KL_divergence import Kullback_Leibler_CMW
from ._aux_definitions import optimal_dirichlet_param
from ._aux_shannon import _unit_Dict_

_method_List_ = ["naive", "CMW", "Jeffreys", "Laplace", "minimax", "SG"]
_which_List_ = ["Jensen-Shannon", "Kullback-Leibler"]

#################
#  SWITCHBOARD  #
#################

def switchboard( compACT, method="naive", unit=None, which="Kullback-Leibler", **kwargs ):
    
    # loading units
    if unit in _unit_Dict_.keys( ) :
        unit_conv = _unit_Dict_[ unit ]
    else:
        raise IOError("Unknown unit. Please choose `unit` amongst :", _unit_Dict_.keys( ) )

    # check which 
    if which not in _which_List_ :
        raise IOError("Unkown divergence. Please choose `which` amongst :", _which_List_ )

    # choosing entropy estimation method
    if method in ["naive", "maximum-likelihood"] :  
        dkl_estimate = Naive( compACT, which=which, **kwargs )
    
    elif method == "CMW" :                       # Camaglia Mora Walczak
        if which in ["Jensen-Shannon"] :
            raise IOError("Unknown method `CMW` for the chosen divergence.")
        elif which == "Kullback-Leibler" :
            dkl_estimate = Kullback_Leibler_CMW( compACT, **kwargs )
        else :
            pass

    elif method in ["D", "Dirichlet"] :
        if "a" not in kwargs :
            raise IOError("The Dirichlet parameter `a` must be specified.")
        a = kwargs["a"]
        if "b" in kwargs :
            b = kwargs["b"]
        else :
            b = a
        dkl_estimate = Dirichlet( compACT, a, b, which=which )       
    
    elif method in ["Jeffreys", "Krichevsky-Trofimov"] :
        a = 0.5
        b = 0.5
        dkl_estimate = Dirichlet( compACT, a, b, which=which )
    
    elif method in ["L", "Laplace", "Bayesian-Laplace"] :
        a = 1.
        b = 1.
        dkl_estimate = Dirichlet( compACT, a, b, which=which )
        
    elif method in ["minimax", "Trybula"]:  
        a = np.sqrt( compACT.N_1 ) / compACT.compact_1.K
        b = np.sqrt( compACT.N_2 ) / compACT.compact_2.K
        dkl_estimate = Dirichlet( compACT, a, b, which=which )
     
    elif method in ["SG", "Schurmann-Grassberger"]:
        a = 1. / compACT.compact_1.Kobs
        b = 1. / compACT.compact_2.Kobs
        dkl_estimate = Dirichlet( compACT, a, b, which=which )

    else:
        raise IOError("Unkown method. Please choose `method` amongst :", _method_List_ )

    return unit_conv * dkl_estimate
###

###########
#  NAIVE  #
###########

def Naive( compACT, which="Kullback-Leibler", ) :
    '''Estimation of divergence with frequencies of observed categories.'''
    
    # loading parameters from compACT 
    N_1, N_2 = compACT.N_1, compACT.N_2
    # delete 0 counts
    gtr0mask = np.logical_and( compACT.nn_1 > 0, compACT.nn_2 > 0 )
    nn_1, nn_2, ff = compACT.nn_1[gtr0mask], compACT.nn_2[gtr0mask], compACT.ff[gtr0mask]
    
    hh_1 = nn_1 / N_1                  # frequencies
    hh_2 = nn_2 / N_2                  # frequencies
    
    if which == "Jensen-Shannon" :
        mm_1p2 = 0.5 * ( hh_1 + hh_2 )
        output = 0.5 * np.dot( ff, hh_1 * np.log( hh_1 / mm_1p2 ) + hh_2 * np.log( hh_2 / mm_1p2 ) )

    elif which == "Kullback-Leibler" :                       
        output = np.dot( ff, hh_1 * np.log( hh_1 / hh_2 ) )

    else :
        raise IOError("Unknown method `Naive` for the chosen divergence.")

    return np.array( output )

##########################
#  DIRICHELET ESTIMATOR  #
##########################

def Dirichlet( compACT, a, b, which="Kullback-Leibler", ):
    '''Estimation of divergence with Dirichlet-multinomial pseudocount model.
    '''

    # loading parameters from compACT 
    N_1, N_2, K = compACT.N_1, compACT.N_2, compACT.K
    # delete 0 counts
    nn_1, nn_2, ff = compACT.nn_1, compACT.nn_2, compACT.ff

    if a == "optimal" :
        a = optimal_dirichlet_param(compACT.compact_1)
    else :
        try:
            a = np.float64(a)
        except :
            raise IOError('The Dirichlet parameter must be a scalar.')
        if a < 0 :
            raise IOError('The Dirichlet parameter must greater than 0.')

    if b == "optimal" :
        b = optimal_dirichlet_param(compACT.compact_2)
    else :
        try:
            b = np.float64(b)
        except :
            raise IOError('The Dirichlet parameter must be a scalar.')
        if b < 0 :
            raise IOError('The Dirichlet parameter must greater than 0.')

    hh_1_a = ( nn_1 + a ) / ( N_1 + K*a )     # frequencies with pseudocounts
    hh_2_b = ( nn_2 + b ) / ( N_2 + K*b )     # frequencies with pseudocounts

    if which == "Jensen-Shannon" :
        mm_1p2_ab = 0.5 * ( hh_1_a + hh_2_b )
        output = 0.5 * np.dot( ff, hh_1_a * np.log( hh_1_a / mm_1p2_ab ) + hh_2_b * np.log( hh_2_b / mm_1p2_ab ) )

    elif which == "Kullback-Leibler" :                               
        output = np.dot( ff, hh_1_a * np.log( hh_1_a / hh_2_b ) )

    else :
        raise IOError("Unknown method `Dirichlet` for the chosen divergence.")

    return np.array( output )
###
