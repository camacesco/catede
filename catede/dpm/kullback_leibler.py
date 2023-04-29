#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Kullback-Leibler Divergence Estimator
    Copyright (C) February 2023 Francesco Camaglia, LPENS 
'''

import numpy as np
from .divergence import dpm_estimator, dpm_wrapper
from ..bayesian_calculus import *
from ..nsb.shannon import DirShannon

############################
#  CROSSENTROPY METAPRIOR  #
############################

class DirCrossEntr( one_dim_metapr ) :
    def __init__( self, K ) :
        '''Class for `a priori` expected crossentropy under symmetric Dirichlet prior.'''
        self.K = K 
        self._sign = -1
    def apriori( self, b ) :
        '''`a priori` expected cross-entropy.'''
        return D_diGmm(self.K * b, b)
    def drv_1( self, b ) :
        '''1st derivative of the `a priori` expected crossentropy.'''
        return self.K * triGmm(self.K * b) - triGmm(b)
    def drv_2( self, b ) :
        '''2nd derivative of the `a priori` expected crossentropy.'''
        return np.power(self.K, 2) * quadriGmm(self.K * b) - quadriGmm(b)
    def drv_3( self, b ) :
        '''3rd derivative of the `a priori` expected crossentropy.'''
        return np.power(self.K, 3) * quintiGmm(self.K * b) - quintiGmm(b)

###################################
#  EQUAL KL DIVERGENCE METAPRIOR  #
###################################

class equalDirKLdiv( one_dim_metapr ) :
    def __init__( self, K ) :
        '''Class for `a priori` expected KL divergence under equal symmetric Dirichlet priors.'''
        self.K = K 
        self._cnst = (self.K - 1) / self.K 
        self._sign = 1
    def apriori( self, a ) :
        '''`a priori` expected (equal) KL divergence.'''
        return self._cnst * np.power(a, -1)
    def drv_1( self, a ) :
        '''1st derivative of the `a priori` expected (equal) KL divergence.'''
        return self._cnst * np.power(a, -2)
    def drv_2( self, a ) :
        '''2nd derivative of the `a priori` expected (equal) KL divergence.'''
        return - 2 * self._cnst * np.power(a, -3)
    def drv_3( self, a ) :
        '''3rd derivative of the `a priori` expected (equal) KL divergence.'''
        return 6 * self._cnst * np.power(a, -4)

#############################
#  KL DIVERGENCE METAPRIOR  #
#############################

class DirKLdiv( two_dim_metapr ) :
    def __init__(self, K, choice, **kwargs ) :
        two_dim_metapr.__init__(self, K, choice, **kwargs)
        self.A = DirShannon(self.K)
        self.B = DirCrossEntr(self.K)
    
    '''
    A priori divergence < D | a, b >
    '''
        
    def diverg_apriori( self, a, b ):
        '''.'''
        return self.B.apriori(b) - self.A.apriori(a)
    def diverg_apriori_jac( self, a, b ):
        '''.'''
        dap = self.diverg_apriori( a, b )
        output = np.zeros( shape = (np.size(dap), 2,) )
        # FIXME : can I predict dap shape just from a and b ?
        output[:,0] -= self.A.drv_1(a)
        output[:,1] += self.B.drv_1(b)
        return output
    def diverg_apriori_hess( self, a, b ):
        '''.'''
        dap = self.diverg_apriori( a, b )
        output = np.zeros( shape = (np.size(dap), 2, 2,) )
        # FIXME : can I predict dap shape just from a and b ?
        output[:,0,0] -= self.A.drv_2(a)
        output[:,1,1] += self.B.drv_2(b)
        return output
        
    '''
    Marginalizing function phi.
    '''

    def marginaliz_phi( self, a, b ) :
        '''.'''
        dap = self.diverg_apriori( a, b )
        output = np.ones( shape = (np.size(dap),) )
        mask = dap < np.log(self.K)
        output[ mask ] /= dap[ mask ]
        output[ ~mask ] /= np.log(self.K)
        return output
    def log_marginaliz_phi( self, a, b ) :
        '''.'''
        dap = self.diverg_apriori( a, b )
        output = np.zeros( shape = (np.size(dap),) )
        mask = dap < np.log(self.K) 
        output[ mask ] -= np.log( dap[mask] )
        output[ ~mask ] -= np.log( np.log(self.K ))
        return output
    def log_marginaliz_phi_jac( self, a, b ) :
        '''.'''
        dap = self.diverg_apriori( a, b )
        dap_jac = self.diverg_apriori_jac(a, b)
        output = np.zeros( shape = np.shape(dap_jac))
        mask = dap < np.log(self.K)
        output[mask,0] -= (dap_jac[:,0] / dap)[mask]
        output[mask,1] -= (dap_jac[:,1] / dap)[mask]
        return output
    def log_marginaliz_phi_hess( self, a, b ) :
        '''.'''
        dap = self.diverg_apriori( a, b )
        dap_jac = self.diverg_apriori_jac(a, b)
        dap_hess = self.diverg_apriori_hess(a, b)
        output = np.zeros( shape = np.shape(dap_hess) )
        mask = dap < np.log(self.K)
        output[mask,0,0] -= (dap_hess[:,0,0] / dap - np.power(dap_jac[:,0] / dap, 2))[mask]
        output[mask,0,1] -= (dap_jac[:,0] * dap_jac[:,1] * np.power(dap, -2))[mask]
        output[mask,1,0] = output[mask,0,1]
        output[mask,1,1] -= (dap_hess[:,1,1] / dap - np.power(dap_jac[:,1] / dap, 2))[mask]
        return output

########################
#  DIVERGENCE WRAPPER  #
########################

class dpm_Kullback_Leibler( dpm_wrapper ) :
    ''' Auxiliary class to compute Kullback Leibler divergence UDM estimator.'''
    def __init__( self, cpct_div, choice="log-uniform", **kwargs ) :
        self.cpct_div = cpct_div
        self.dir_meta_obj = DirKLdiv(cpct_div.K, choice=choice, **kwargs)
        self.meta_likelihood = two_dim_meta_likelihood(self.cpct_div, self.dir_meta_obj)

    def divergence( self, a, b ) :
        return self.cpct_div.kullback_leibler( a, b )
    def squared_divergence( self, a, b ) :
        return self.cpct_div.squared_kullback_leibler( a, b )
    def mean(self, DIV1) :
        return DIV1
    def std(self, DIV1, DIV2) :
        return np.sqrt(DIV2 - np.power(DIV1, 2))
    '''
    # Equal priori case
    def equal_prior( self, a ) :
        return equalDirKLdiv( a, self.cpct_div.K ).aPrioriExpec()

    def optimal_equal_param( self, ) :
        return optimal_equal_KLdiv_param( self.cpct_div )
    
    def log_equal_evidence_hess( self, a ) :
        return log_equal_KLdiv_meta_posterior_hess( a, self.cpct_div)
    '''

'''
#  EQUAL HYPER-PARAMS CASE TO BE UPDATED  #
def log_equal_KLdiv_meta_posterior_hess( var, *args ) :
    hess_LogLike = equalDirKLdiv(args[0].K).logmetapr_hess(var)
    hess_LogLike += Polya(args[0].compact_1).log_hess(var) 
    hess_LogLike += Polya(args[0].compact_2).log_hess(var) 
    return hess_LogLike

def optimal_equal_KLdiv_param( cpct_div ) :
    def myfunc(var, *args) :
        LogLike = equalDirKLdiv(args[0].K).logmetapr(var)
        LogLike += Polya(args[0].compact_1).log(var)
        LogLike += Polya(args[0].compact_2).log(var)
        return - LogLike
    def myjac(var, *args) :
        jac_LogLike = equalDirKLdiv(var, args[0].K).logmetapr_jac(var)
        jac_LogLike += Polya(args[0].compact_1).log_jac(var)
        jac_LogLike += Polya(args[0].compact_2).log_jac(var)
        return - jac_LogLike
    return minimize( myfunc, [INIT_GUESS], args=(cpct_div,), bounds=(BOUND_DIR,), jac=myjac )
'''

################
#  EXECUTABLE  #
################

def main( cpct_div, error=False, n_bins=None, equal_prior=False, n_sigma=3,
    choice="log-uniform", scaling=1., cpu_count=None, verbose=False, logscaled=True ) :
    '''Kullback-Leibler divergence estimation with UDM method.'''

    # FIXME : developer 
    if equal_prior is True :
        raise SystemError( "The options `equal_prior` is not available yet."  )
    
    dpm_wrap = dpm_Kullback_Leibler(cpct_div, choice=choice, scaling=scaling)
    output = dpm_estimator(dpm_wrap, error=error, n_bins=n_bins, n_sigma=n_sigma,
                           cpu_count=cpu_count, verbose=verbose, logscaled=logscaled)
    return output 
 