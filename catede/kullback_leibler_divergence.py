#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Kullback-Leibler Divergence Estimator
    Copyright (C) February 2023 Francesco Camaglia, LPENS 
'''

import numpy as np
from .dpm_divergence import dpm_estimator
from .new_calculus import *

class dpm_Kullback_Leibler() :
    ''' Auxiliary class to compute Kullback Leibler divergence UDM estimator.'''
    def __init__( self, comp_div, choice="log-uniform", scaling=1 ) :
        self.comp_div = comp_div
        self.choice = choice
        self.scaling = scaling

    def optimal_equal_param( self, ) :
        return optimal_equal_KLdiv_param( self.comp_div )
    
    def log_equal_evidence_hess( self, a ) :
        return log_equal_KLdiv_meta_posterior_hess( a, self.comp_div)

    def optimal_divergence_params( self ) :
        return optimal_KL_divergence_params( self.comp_div, choice=self.choice, scaling=self.scaling )
    
    def log_evidence_hess( self, a, b ) :
        return log_KL_divergence_meta_posterior_hess([a, b], self.comp_div, self.choice, {"scaling" : self.scaling})
    
    def divergence( self, a, b ) :
        return self.comp_div.kullback_leibler( a, b )
    
    def squared_divergence( self, a, b ) :
        return self.comp_div.squared_kullback_leibler( a, b )
    
    def dpm_prior( self, var ) :
        return DirKLdiv( var, self.comp_div.K, self.choice ).Metapr()

    def equal_prior( self, a ) :
        return equalDirKLdiv( a, self.comp_div.K ).aPrioriExpec()
    
    def estim_mean( self, DIV1 ) :
        return DIV1
    
    def estim_std( self, DIV1, DIV2 ) :
        return np.sqrt( DIV2 - np.power( DIV1, 2 ) )

def main(
    comp_div, error=False, n_bins="default", equal_prior=False,
    choice="log-uniform", scaling=1.,
    cpu_count=None, verbose=False, 
    ) :
    '''Kullback-Leibler divergence estimation with UDM method.'''

    dpm_wrap = dpm_Kullback_Leibler( comp_div, choice=choice, scaling=scaling )

    output = dpm_estimator( dpm_wrap, 
                  error=error, n_bins=n_bins, 
                  equal_prior=equal_prior, cpu_count=cpu_count, verbose=verbose,
                  )
    return output 
 