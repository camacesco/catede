#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    *** Default Divergence ***
    Copyright (C) January 2022 Francesco Camaglia, LPENS 
    Following the architecture of J. Hausser and K. Strimmer : https://strimmerlab.github.io/software/entropy/
'''

import numpy as np 
from .cmw_KL_divergence import Kullback_Leibler_CMW

# loagirthm unit
_unit_Dict_ = { "ln": 1., "log2": 1./np.log(2), "log10": 1./np.log(10) }
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
    if method == "naive" :                       # Naive
        dkl_estimate = Naive( compACT, which=which, **kwargs )
    
    elif method == "CMW" :                       # Camaglia Mora Walczak
        if which in ["Jensen-Shannon"] :
            raise IOError("Unknown method `CMW` for the chosen divergence.")
        else :
            dkl_estimate = Kullback_Leibler_CMW( compACT, **kwargs )
    
    elif method == "Jeffreys" :                  # Jeffreys
        a = 0.5
        b = 0.5
        dkl_estimate = Dirichlet( compACT, a, b, which=which, **kwargs )
    
    elif method == "Laplace" :                   # Laplace
        a = 1.
        b = 1.
        dkl_estimate = Dirichlet( compACT, a, b, which=which, **kwargs )
        
    elif method == "minimax" :                   # minimax
        a = np.sqrt( compACT.N_A ) / compACT.compact_A.Kobs
        b = np.sqrt( compACT.N_B ) / compACT.compact_B.Kobs
        dkl_estimate = Dirichlet( compACT, a, b, which=which, **kwargs )
     
    elif method == "SG" :                        # Schurmann-Grassberger
        a = 1. / compACT.compact_A.Kobs
        b = 1. / compACT.compact_B.Kobs
        dkl_estimate = Dirichlet( compACT, a, b, which=which, **kwargs )

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
    N_A, N_B = compACT.N_A, compACT.N_B
    # delete 0 counts
    gtr0mask = np.logical_and( compACT.nn_A > 0, compACT.nn_B > 0 )
    nn_A, nn_B, ff = compACT.nn_A[gtr0mask], compACT.nn_B[gtr0mask], compACT.ff[gtr0mask]
    
    hh_A = nn_A / N_A                  # frequencies
    hh_B = nn_B / N_B                  # frequencies
    
    if which == "Jensen-Shannon" :
        mm_AB = 0.5 * ( hh_A + hh_B )
        output = 0.5 * np.dot( ff, hh_A * np.log( hh_A / mm_AB ) + hh_B * np.log( hh_B / mm_AB ) )
    elif which == "Kullback-Leibler" :                       
        output = np.dot( ff, hh_A * np.log( hh_A / hh_B ) )
    else :
        raise IOError("Unknown method `Naive` for the chosen divergence.")

    return np.array( output )

##########################
#  DIRICHELET ESTIMATOR  #
##########################

def Dirichlet( compACT, a, b, which="Kullback-Leibler", ):
    '''Estimation of divergence with Dirichlet-multinomial pseudocount model.

    Pseudocount per bin (Dirichlet parameter)
    a=1 , b=1                            :   Laplace
    a=1/2 , b=1/2                        :   Jeffreys
    a=1/Kobs , b=1/Kobs                  :   Schurmann-Grassberger  (Kobs: number of bins)
    a=sqrt(N_A)/Kobs , b=sqrt(N_B)/Kobs  :   minimax
    
    Parameters
    ----------  
    a : float
        Dirichlet parameter of experiment A
    b : float
        Dirichlet parameter of experiment B
    which :    
    '''

    # loading parameters from compACT 
    N_A, N_B = compACT.N_A, compACT.N_B
    # delete 0 counts
    nn_A, nn_B, ff = compACT.nn_A, compACT.nn_B, compACT.ff

    nn_A_a = nn_A + a                   # counts plus pseudocounts
    nn_B_b = nn_B + b                   # counts plus pseudocounts                                   
    N_A_a = N_A + a * np.sum( ff )      # total number of counts plus pseudocounts               
    N_B_b = N_B + b * np.sum( ff )      # total number of counts plus pseudocounts
    
    hh_A_a = nn_A_a / N_A_a             # frequencies
    hh_B_b = nn_B_b / N_B_b             # frequencies
    if which == "Jensen-Shannon" :
        mm_AB_ab = 0.5 * ( hh_A_a + hh_B_b )
        output = 0.5 * np.dot( ff, hh_A_a * np.log( hh_A_a / mm_AB_ab ) + hh_B_b * np.log( hh_B_b / mm_AB_ab ) )
    elif which == "Kullback-Leibler" :                               
        output = np.dot( ff, hh_A_a * np.log( hh_A_a / hh_B_b ) )
    else :
        raise IOError("Unknown method `Dirichlet` for the chosen divergence.")

    return np.array( output )
###
