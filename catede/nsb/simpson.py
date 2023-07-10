#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Simpson Index NSB Estimator
    Copyright (C) June 2023 Francesco Camaglia, LPENS 
'''

import numpy as np
from .entropy import nsb_estimator, nsb_wrapper
from ..bayesian_calculus import *

#############################
#  SIMPSON INDEX METAPRIOR  #
#############################

class DirSimpson( one_dim_metapr ) :
    def __init__( self, K ) :
        '''Class for `a priori` expected Simpson index under symmetric Dirichlet prior.'''
        self.K = K
        self._sign = 1
    def apriori( self, a ) :
        '''`a priori` expected Simpson index.'''
        return (self.K - 1) * np.power( self.K * a + 1, -2 )
    def drv_1( self, a ) :
        '''1st derivative of the `a priori` expected Simpson index.'''
        return 2 * self.K * (self.K - 1) * np.power( self.K * a + 1, -3 )
    def drv_2( self, a ) :
        '''2nd derivative of the `a priori` expected Simpson index.'''
        return - 6 * np.power(self.K, 2) * (self.K - 1) * np.power(self.K * a + 1, -4)
    def drv_3( self, a ) :
        '''3rd derivative of the `a priori` expected Simpson index.'''
        return 24 * np.power(self.K, 3) * (self.K -1) * np.power(self.K * a + 1, -5)

#####################
#  ENTROPY WRAPPER  #
#####################

class nsb_simpson(nsb_wrapper) :
    ''' Auxiliary class to compute Shannon entropy NSB estimator.'''
    def __init__( self, cpct_exp ) :
        self.cpct_exp = cpct_exp
        self.dir_meta_obj = DirSimpson(cpct_exp.K)
        self.meta_likelihood = one_dim_meta_likelihood(self.cpct_exp, self.dir_meta_obj)

    def entropy(self, a) :
        return self.cpct_exp.simpson(a)
    def squared_entropy(self, a) :
        return self.cpct_exp.squared_simpson(a)    
    def mean(self, S1) :
        return S1
    def std(self, S1, S2) :
        return np.sqrt(S2 - np.power(S1, 2))
    
################
#  EXECUTABLE  #
################

def main(cpct_exp, error=False, n_bins=None, 
         cpu_count=None, verbose=False ) :
    '''Kullback-Leibler divergence estimation with UDM method.'''

    nsb_wrap = nsb_simpson(cpct_exp)
    output = nsb_estimator(nsb_wrap, error=error, n_bins=n_bins, 
                           cpu_count=cpu_count, verbose=verbose)
    return output 