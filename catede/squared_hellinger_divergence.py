#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Squared Hellinger Divergence Estimator
    Copyright (C) February 2023 Francesco Camaglia, LPENS 
'''

import numpy as np
from .dpm_divergence import dpm_estimator
from .new_calculus import *

class dpm_Hellinger() :
    ''' Auxiliary class to compute squared Hellinger divergence UDM estimator.'''
    def __init__( self, comp_div, choice="log-uniform", scaling=1 ) :
        self.comp_div = comp_div
        self.choice = choice
        self.scaling = scaling

    def equal_prior( self, a ) :
        return None
    
    def optimal_equal_param( self, ) :
        return None
    
    def log_equal_evidence_hess( self, a ) :
        return None

    def dpm_prior( self, var ) :
        return DirHelldiv( var, self.comp_div.K, self.choice ).Metapr()
    
    def optimal_divergence_params( self ) :
        return optimal_Hellinger_params( self.comp_div, choice=self.choice, scaling=self.scaling )
    
    def log_evidence_hess( self, a, b ) :
        return log_Hellinger_meta_posterior_hess([a, b], self.comp_div, self.choice, {"scaling" : self.scaling})
    
    def divergence( self, a, b ) :
        return self.comp_div.bhattacharrya( a, b )
    
    def squared_divergence( self, a, b ) :
        return self.comp_div.squared_bhattacharrya( a, b )
    
    def estim_mean( self, DIV1 ) :
        return 1 - DIV1
    
    def estim_std( self, DIV1, DIV2 ) :
        return np.sqrt( DIV2 - np.power( DIV1, 2 ) )

def main(
    comp_div, error=False, n_bins="default", equal_prior=False,
    choice="log-uniform", scaling=1.,
    cpu_count=None, verbose=False, 
    ) :
    '''squared Hellinger divergence estimation with UDM method.'''

    # FIXME : developer 
    if equal_prior is True :
        raise SystemError( "The options `equal_prior` is not available yet."  )

    dpm_wrap = dpm_Hellinger( comp_div, choice=choice, scaling=scaling )

    output = dpm_estimator( dpm_wrap, 
                  error=error, n_bins=n_bins, 
                  equal_prior=equal_prior, cpu_count=cpu_count, verbose=verbose,
                  )
    return output 
 