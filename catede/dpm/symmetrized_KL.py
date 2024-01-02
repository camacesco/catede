#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    symmetrized Kullback-Leibler Divergence DP(M) Estimator
    Copyright (C) January 2024 Francesco Camaglia, LPENS 
'''

import numpy as np
from .divergence import dpm_estimator, dpm_wrapper
from ..bayesian_calculus import *
from copy import deepcopy

############################
#  CROSSENTROPY METAPRIOR  #
############################

class _sDKL_auxfunc( one_dim_metapr ) :
    def __init__( self, K ) :
        '''Class for `a priori` expected symmetrized DKL under symmetric Dirichlet prior.'''
        self.K = K 
        self._sign = -1
        self.cnst = (K - 1.) / K
    def apriori( self, a ) :
        '''`a priori` expected <aux_name>.'''
        return self.cnst * np.power(a, -1)
    def drv_1( self, a ) :
        '''1st derivative of the `a priori` <aux_name>.'''
        return - self.cnst * np.power(a, -2)
    def drv_2( self, a ) :
        '''2nd derivative of the `a priori` <aux_name>.'''
        return 2 * self.cnst * np.power(a, -3)
    def drv_3( self, a ) :
        '''3rd derivative of the `a priori` <aux_name>.'''
        return - 6 * self.cnst * np.power(a, -4)

##################################
#  SYMM KL DIVERGENCE METAPRIOR  #
#################################

class DirSymmetrizedKLdiv( two_dim_metapr ) :
    def __init__(self, K, choice, **kwargs ) :
        two_dim_metapr.__init__(self, K, choice, **kwargs)
        self.A = _sDKL_auxfunc(self.K)
        self.B = _sDKL_auxfunc(self.K)
    
    '''
    A priori divergence < D | a, b >
    '''
        
    def diverg_apriori( self, a, b ):
        '''.'''
        return self.A.apriori(a) + self.B.apriori(b)
    def diverg_apriori_jac( self, a, b ):
        '''.'''
        dap = self.diverg_apriori( a, b )
        output = np.zeros( shape = (np.size(dap), 2,) )
        # FIXME : can I predict dap shape just from a and b ?
        output[:,0] += self.A.drv_1(a)
        output[:,1] += self.B.drv_1(b)
        return output
    def diverg_apriori_hess( self, a, b ):
        '''.'''
        dap = self.diverg_apriori( a, b )
        output = np.zeros( shape = (np.size(dap), 2, 2,) )
        # FIXME : can I predict dap shape just from a and b ?
        output[:,0,0] += self.A.drv_2(a)
        output[:,1,1] += self.B.drv_2(b)
        return output
        
    '''
    Marginalizing function phi.
    '''

    def marginaliz_phi( self, a, b ) :
        '''.'''
        dap = self.diverg_apriori( a, b )
        output = np.ones( shape = (np.size(dap),) ) / dap
        return output
    def log_marginaliz_phi( self, a, b ) :
        '''.'''
        dap = self.diverg_apriori( a, b )
        output = np.zeros( shape = (np.size(dap),) )
        output -= np.log( dap )
        return output
    def log_marginaliz_phi_jac( self, a, b ) :
        '''.'''
        dap = self.diverg_apriori( a, b )
        dap_jac = self.diverg_apriori_jac(a, b)
        output = np.zeros( shape = np.shape(dap_jac))
        output[:,0] -= (dap_jac[:,0] / dap)
        output[:,1] -= (dap_jac[:,1] / dap)
        return output
    def log_marginaliz_phi_hess( self, a, b ) :
        '''.'''
        dap = self.diverg_apriori( a, b )
        dap_jac = self.diverg_apriori_jac(a, b)
        dap_hess = self.diverg_apriori_hess(a, b)
        output = np.zeros( shape = np.shape(dap_hess) )
        output[:,0,0] -= (dap_hess[:,0,0] / dap - np.power(dap_jac[:,0] / dap, 2))
        output[:,0,1] -= (dap_jac[:,0] * dap_jac[:,1] * np.power(dap, -2))
        output[:,1,0] = output[:,0,1]
        output[:,1,1] -= (dap_hess[:,1,1] / dap - np.power(dap_jac[:,1] / dap, 2))
        return output

########################
#  DIVERGENCE WRAPPER  #
########################

class dpm_Symmetrized_KL( dpm_wrapper ) :
    ''' Auxiliary class to compute Kullback Leibler divergence UDM estimator.'''
    def __init__( self, cpct_div, choice="log-uniform", **kwargs ) :
        self.cpct_div = cpct_div
        self.cpct_div_rev = deepcopy(cpct_div)
        self.cpct_div_rev.reverse()
        self.dir_meta_obj = DirSymmetrizedKLdiv(cpct_div.K, choice=choice, **kwargs)
        self.meta_likelihood = two_dim_meta_likelihood(self.cpct_div, self.dir_meta_obj)

    def divergence( self, a, b ) :
        return 0.5 * ( self.cpct_div.kullback_leibler(a, b) + self.cpct_div_rev.kullback_leibler(b, a) )
    def mean(self, DIV1) :
        return DIV1
    '''
    # FIXME: to be implemented
    def squared_divergence( self, a, b ) :
        return self.cpct_div.squared_kullback_leibler( a, b )
    def std(self, DIV1, DIV2) :
        return np.sqrt(DIV2 - np.power(DIV1, 2))
    '''

################
#  EXECUTABLE  #
################

def main( cpct_div, error=False, n_bins=None, n_sigma=3,
    choice="log-uniform", scaling=1., cpu_count=None, verbose=False, logscaled=True ) :
    '''Kullback-Leibler divergence estimation with UDM method.'''

    # FIXME : developer 
    if error == True :
        raise warnings.warn( "The options `error` is not available yet."  )
      
    dpm_wrap = dpm_Symmetrized_KL(cpct_div, choice=choice, scaling=scaling)
    output = dpm_estimator(dpm_wrap, error=False, n_bins=n_bins, n_sigma=n_sigma,
                           cpu_count=cpu_count, verbose=verbose, logscaled=logscaled)
    return output 
 