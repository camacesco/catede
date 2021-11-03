#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Default Divergence (in development)
    Copyright (C) November 2021 Francesco Camaglia, LPENS 
'''

import numpy as np 

from kamapack import nsb_divergence

# loagirthm unit
_unit_Dict_ = { "ln": 1., "log2": 1./np.log(2), "log10": 1./np.log(10) }


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
        dkl_estimate = MaximumLikelihood( compACT )
    elif method == "NSB":                       # Nemenman Shafee Bialek (?)
        dkl_estimate = nsb_divergence.NemenmanShafeeBialek( compACT, **kwargs )
    else:
        raise IOError("The chosen method is unknown.")

    return unit_conv * dkl_estimate
###



##################################
#  MAXIMUM LIKELIHOOD ESTIMATOR  #
##################################

def MaximumLikelihood( compACT, only_cross=False ) :
    '''
    Replacing probabilities with frequencies without considering categories not seen in one of the two.
    '''
    
    # loading parameters from compACT 
    N_A, N_B = compACT.N_A, compACT.N_B
    # delete 0 counts
    gtr0mask = np.logical_and( compACT.nn_A > 0, compACT.nn_B > 0 )
    nn_A, nn_B, ff = compACT.nn_A[gtr0mask], compACT.nn_B[gtr0mask], compACT.ff[gtr0mask]
    
    hh_A = nn_A / N_A
    hh_B = nn_B / N_B
    
    if only_cross is True :
        output = np.dot ( ff, - hh_A * np.log( hh_B ))
    else :                                   
        output = np.dot ( ff, hh_A * np.log( hh_A / hh_B ) )
        
    return output
