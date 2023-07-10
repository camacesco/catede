#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Squared Hellinger Divergence Estimator
    Copyright (C) June 2023 Francesco Camaglia, LPENS 
'''

import numpy as np
from .divergence import dpm_estimator, dpm_wrapper
from ..bayesian_calculus import *

#######################################
#  AUXILIARY BHATTACHARRYA METAPRIOR  #
#######################################

class _Dir_Bhatt_auxfunc( one_dim_metapr ) :
    def __init__(self, K) :
        '''Class for `a priori` expected *** under symmetric Dirichlet prior.'''
        self.K = K
        self._sign = 1
    def _log_auxfunc(self, x_i, X, K) :
        return 0.5 * np.log(K) + LogGmm(x_i+0.5) - LogGmm(x_i) + LogGmm(X) - LogGmm(X+0.5)
    def apriori(self, a) :
        '''`a priori` expected <aux_name>.'''
        return np.exp(self._log_auxfunc(a, self.K * a, self.K))
    def drv_1(self, a) :
        '''1st derivative of the `a priori` expected <aux_name>.'''
        tmp = self.K * D_diGmm(self.K * a, self.K * a + 0.5) - D_diGmm(a, a + 0.5)
        return tmp * self.apriori(a)
    def drv_2(self, a) :
        '''2nd derivative of the `a priori` expected <aux_name>.'''
        tmp = np.power(self.K, 2) * D_triGmm(self.K * a, self.K * a + 0.5) - D_triGmm(a, a + 0.5)
        return  tmp * self.apriori(a) + np.power(self.metapr(a), 2) / self.apriori(a)
    def drv_3(self, a) :
        '''3rd derivative of the `a priori` expected <aux_name>.'''
        tmp = np.power(self.K, 3) * D_quadriGmm(self.K * a, self.K * a + 0.5) - D_quadriGmm(a, a + 0.5)
        tmp *= self.apriori(a)
        tmp += 2 * self.metapr_jac(a) * self.metapr(a) / self.apriori(a)
        tmp += - np.power(self.metapr(a) / self.apriori(a), 2)
        return tmp

####################################
#  HELLINGER DIVERGENCE METAPRIOR  #
####################################

class DirHelldiv( two_dim_metapr ) :
    def __init__(self, var, *args, **kwargs ) :
        two_dim_metapr.__init__(self, var, *args, **kwargs)
        self.A = _Dir_Bhatt_auxfunc(self.K)
        self.B = _Dir_Bhatt_auxfunc(self.K)
    
    '''
    A priori divergence < D | a, b >
    --------------------------------
    '''
        
    def diverg_apriori(self, a, b):
        '''.'''
        return self.A.apriori(a) * self.B.apriori(b)
    def diverg_apriori_jac(self, a, b):
        '''.'''
        dap = self.diverg_apriori(a, b)
        output = np.zeros(shape = (np.size(dap), 2,))
        # FIXME : can I predict dap shape just from a and b ?
        output[:,0] += self.A.drv_1(a) * self.B.apriori(b)
        output[:,1] += self.A.apriori(a) * self.B.drv_1(b)
        return output
    def diverg_apriori_hess(self, a, b):
        '''.'''
        dap = self.diverg_apriori(a, b)
        output = np.zeros(shape = (np.size(dap), 2, 2,))
        # FIXME : can I predict dap shape just from a and b ?
        output[:,0,0] += self.A.drv_2(a) * self.B.apriori(b)
        tmp = self.A.drv_1(a) * self.B.drv_1(b)
        output[:,0,1] += tmp
        output[:,1,0] += tmp
        output[:,1,1] += self.A.apriori(a) * self.B.drv_2(b)
        return output
            
    '''
    Marginalizing function phi.
    --------------------------
    '''

    def marginaliz_phi(self, a, b) :
        '''.'''
        return np.exp(self.log_marginaliz_phi(a, b))
    def log_marginaliz_phi(self, a, b) :
        '''.'''
        dap = self.diverg_apriori(a, b)
        return 2 * np.log(1 - dap) - np.log(dap) - np.log(2 - dap)
    def log_marginaliz_phi_jac( self, a, b ) : 
        '''.'''
        dap = self.diverg_apriori(a, b)
        dap_jac = self.diverg_apriori_jac(a, b)
        output = np.zeros( shape = np.shape(dap_jac) )
        tmp = - 2 * np.power(1 - dap, -1) - np.power(dap, -1) + np.power(2 - dap, -1)
        output[:,0] = tmp * dap_jac[:,0]
        output[:,1] = tmp * dap_jac[:,1]    
        return output
    def log_marginaliz_phi_hess(self, a, b) : 
        '''.'''
        dap = self.diverg_apriori(a, b)
        dap_jac = self.diverg_apriori_jac(a, b)
        dap_hess = self.diverg_apriori_hess(a, b)
        output = np.zeros( shape = np.shape(dap_hess) )
        tmp = - 2 * np.power(1 - dap, -1) - np.power(dap, -1) + np.power(2 - dap, -1)
        tmp_2 = 2 * np.power(1 - dap, -2) + np.power(dap, -2) - np.power(2 - dap, -2)
        output[:,0,0] = tmp * dap_hess[:,0,0] + tmp_2 * dap_jac[:,0]
        output[:,0,1] = dap_hess[:,1,0]
        output[:,1,0] = dap_hess[:,0,1]
        output[:,1,1] = tmp * dap_hess[:,1,1] + tmp_2 * dap_jac[:,1]
        return output

########################
#  DIVERGENCE WRAPPER  #
########################

class dpm_squared_Hellinger(dpm_wrapper) :
    ''' Auxiliary class to compute squared Hellinger divergence UDM estimator.'''
    def __init__( self, cpct_div, choice="log-uniform", **kwargs ) :
        self.cpct_div = cpct_div
        self.dir_meta_obj = DirHelldiv(cpct_div.K, choice=choice, **kwargs)
        self.meta_likelihood = two_dim_meta_likelihood(cpct_div, self.dir_meta_obj)

    def divergence(self, a, b) :
        return self.cpct_div.bhattacharyya(a, b)
    def squared_divergence(self, a, b) :
        return self.cpct_div.squared_bhattacharyya(a, b)
    def mean(self, DIV1) :
        return 1. - DIV1
    def std(self, DIV1, DIV2) :
        return np.sqrt(DIV2 - np.power(DIV1, 2))
    '''
    def equal_prior( self, a ) :
        return None
    
    def optimal_equal_param( self, ) :
        return None
    
    def log_equal_evidence_hess( self, a ) :
        return None
    '''

################
#  EXECUTABLE  #
################

def main( cpct_div, error=False, n_bins=None, equal_prior=False, n_sigma=3,
         choice="log-uniform", scaling=1., cpu_count=None, verbose=False, logscaled=True) :
    '''squared Hellinger divergence estimation with UDM method.'''

    # FIXME : developer 
    if equal_prior is True :
        raise SystemError( "The options `equal_prior` is not available yet."  )

    dpm_wrap = dpm_squared_Hellinger(cpct_div, choice=choice, scaling=scaling)
    output = dpm_estimator(dpm_wrap, error=error, n_bins=n_bins, n_sigma=n_sigma,
                           cpu_count=cpu_count, verbose=verbose, logscaled=logscaled)
    return output 
 