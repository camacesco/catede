#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Bayesian Calculus
    -----------------
    Copyright (C) January 2024 Francesco Camaglia, LPENS 
'''

import warnings
import numpy as np
from scipy import optimize 
from .dirichlet_multinomial import *

MAX_ITER = 1000
TOL = 1.0e-14
BOUND_DIR = (1.0e-6, 1.0e4)
METHOD='L-BFGS-B'
INIT_GUESS = 1.0
CUTOFFRATIO = 5 # FIXME this can be an option
# Warning : CUTOFFRATIO > 1 
USE_JAC_OPT = True
NUMERICAL_ZERO = 1.0e-14
NUMERICAL_INFTY = 1.0e12

# PRIORS ON THE (2D) DIVERGENCE ESTIMATORS #
_PRIOR_CHOICE = {
    "uniform" : {"use_phi" : True, "extra" : {} },
    "log-uniform" : {"use_phi" : True, "extra" : {"scaling"} },
    "scaled": {"use_phi" : False, "extra" : {"scaling"} },
    }
# FIXME : as it's written, "scaled" should be allowed only for DKL

########################################
#  CONCENTRATION PARAMETER Posterior  #
########################################

class BetaMultivariate_symmDir( ) :
    def __init__(self, K) : 
        '''Multivariate Beta function normalization to symmetric Dirichlet distribution.'''
        self.K = K
    def log(self, a) :
        return self.K * LogGmm(a) - LogGmm(self.K * a)
    def log_jac(self, a) :
        return self.K * diGmm(a) - self.K * diGmm(self.K * a)
    def log_hess(self, a) :
        return self.K * triGmm(a) - np.power(self.K, 2) * triGmm(self.K * a)

class Polya( ) :
    def __init__(self, cpct_exp) : 
        '''Polya distribution or symmetric-Dirichlet-multinomial distribution.'''
        self.ce = cpct_exp
    def log(self, a) :
        '''logarithm'''
        output = self.ce.ff.dot(LogGmm(np.add.outer(self.ce.nn, a)))
        output -= LogGmm(self.ce.N + self.ce.K * a) 
        output -= BetaMultivariate_symmDir(self.ce.K).log(a)                
        return output
    def log_jac(self, a) :
        '''1st derivative of the logarithm'''
        output = self.ce.ff.dot(diGmm(np.add.outer(self.ce.nn, a))) 
        output -= self.ce.K * diGmm(self.ce.N + self.ce.K * a) 
        output -= BetaMultivariate_symmDir(self.ce.K).log_jac(a)                   
        return output
    def log_hess(self, a) :
        '''2nd derivative of the logarithm'''
        output = self.ce.ff.dot(triGmm(np.add.outer(self.ce.nn, a))) 
        output -= np.power(self.ce.K, 2) * triGmm(self.ce.N + self.ce.K * a) 
        output -= BetaMultivariate_symmDir(self.ce.K).log_hess(a)     
        return output

#############################
#  ONE DIM METAPRIOR TERMS  #
#############################

class one_dim_metapr( ) :
    ''' Auxiliary class for transformation factor in determinant of the jacobian.'''
    def metapr( self, a ) :
        '''factor of transformation Jacobian determinant (i.e abs of 1st derivative, NSB metaprior) '''
        a = np.array(a).reshape(-1)
        return self._sign * self.drv_1(a)
    def metapr_jac(self, a) :
        '''1st derivative of the transformation factor.'''
        a = np.array(a).reshape(-1)
        return self._sign * self.drv_2(a)
    def metapr_hess(self, a) :
        '''2nd derivative of the transformation factor.'''
        a = np.array(a).reshape(-1)
        return self._sign * self.drv_3(a)
    def logmetapr(self, a) :
        '''logarithm of the transformation factor.'''
        a = np.array(a).reshape(-1)
        return np.log(self.metapr(a))
    def logmetapr_jac(self, a) :
        '''1st derivative of the logarithm of the transformation factor.'''
        a = np.array(a).reshape(-1)
        return self.metapr_jac(a) / self.metapr(a)
    def logmetapr_hess(self, a) :
        '''2nd derivative of the logarithm of the transformation factor.'''
        a = np.array(a).reshape(-1)
        return self.metapr_hess(a) / self.metapr(a) - np.power(self.logmetapr_jac(a), 2)

###########################
#  2-DIM METAPRIOR TERMS  #
###########################

class two_dim_metapr( ) :
    ''' Auxiliary class for two dimensional metapriors.'''
    def __init__(self, K, choice, **kwargs) :
        # note : get around 0-dimensional numpy scalar arrays
        self.K = K
        self.choice = choice
        self._extra = {}

        # NOTE CUTOFFRATIO > 1.
        # FIXME: add option cutoff here
        if self.choice not in _PRIOR_CHOICE :
            raise IOError(f"unrecognized choice `{self.choice}`.\n Choose amongst: {list(_PRIOR_CHOICE.keys())}")
        else :
            for extra in _PRIOR_CHOICE[self.choice]["extra"] :
                self._extra[extra] = kwargs.get(extra, 1.)
            # Known controls
            if ("scaling" in kwargs) and ("scaling" in _PRIOR_CHOICE[self.choice]["extra"]) :    
                try :
                    self._extra["scaling"] = np.float64(kwargs["scaling"])
                    if self._extra["scaling"] <= 0. :
                        raise ValueError( "The parameter `scaling` should be >0." )
                except :
                    raise TypeError( "The parameter `scaling` should be a scalar." )
        self.use_phi = _PRIOR_CHOICE[self.choice]["use_phi"]

    def metapr(self, var) :
        ''' Complete metaprior '''
        # pre-load variables
        a = np.array(var[0]).reshape(-1)
        b = np.array(var[1]).reshape(-1)
        dap = self.diverg_apriori(a,b)

        # contribution of the marginalization constraint phi
        if self.use_phi is True :
            output = self.marginaliz_phi(a, b)
        else :
            output = np.ones(shape = (np.size(dap),))
        
        # contribution of the prior on the a priori expected divergence rho(D)
        if self.choice in ["scaled"] :
            output *= np.exp(- self._extra["scaling"] * (dap / self.A.apriori(a)))
        elif self.choice in ["uniform"] :
            output[dap >= CUTOFFRATIO * np.log(self.K)] = NUMERICAL_ZERO
            # NOTE : no point in adding the normalization 1. / (CUTOFFRATIO * np.log(self.K))
        elif self.choice in ["log-uniform"] :
            output *= np.power(dap, - self._extra["scaling"])    

        # contribution of the jacobian of the transformation
        output *= self.A.metapr(a) * self.B.metapr(b) 
        return output
    
    def logmetapr( self, var ) :
        ''' logarithm of metaprior '''
        # pre-load variables
        a = np.array(var[0]).reshape(-1)
        b = np.array(var[1]).reshape(-1)
        dap = self.diverg_apriori(a, b)

        # contribution of the marginalization constraint phi
        if self.use_phi is True :
            output = self.log_marginaliz_phi(a, b)
        else :
            output = np.zeros(shape = (np.size(dap),))

        # contribution of the prior on the divergence log rho(D)
        if self.choice in ["scaled"] :
            output -= self._extra["scaling"] * (dap / self.A.apriori(a))
        elif self.choice in ["uniform"] :
            output[dap >= CUTOFFRATIO * np.log(self.K)] = - NUMERICAL_INFTY 

        elif self.choice in ["log-uniform"] :
                output -= self._extra["scaling"] * np.log(dap) 

        # contribution of the jacobian of the transformation
        output += self.A.logmetapr(a) + self.B.logmetapr(b)
        return output
    
    def logmetapr_jac( self, var ) :
        ''' Gradient of the logarihtm of metaprior '''
        # pre-load variables
        a = np.array(var[0]).reshape(-1)
        b = np.array(var[1]).reshape(-1)
        dap = self.diverg_apriori(a, b)
        dap_jac = self.diverg_apriori_jac(a, b)

        # contribution of the marginalization constraint phi
        if self.use_phi is True :
            output = self.log_marginaliz_phi_jac(a, b)
        else :
            output = np.zeros(shape = (np.size(dap), 2,))

        # contribution of the prior on the divergence rho
        if self.choice in ["scaled"] :
            output[:, 0] -= self._extra["scaling"] * (dap_jac[:,0] - dap * self.A.drv_1(a) / self.A.apriori(a))  / self.A.apriori(a)
            output[:, 1] -= self._extra["scaling"] * dap_jac[:,1] / self.A.apriori(a)
        elif self.choice in ["uniform"] :
            mask = dap < np.log(self.K)
            output[ ~mask,: ] = NUMERICAL_ZERO
            output[ dap >= CUTOFFRATIO * np.log(self.K),: ] = - NUMERICAL_INFTY
        elif self.choice in ["log-uniform"] :
            output[:, 0] -= self._extra["scaling"] * dap_jac[:,0] / dap
            output[:, 1] -= self._extra["scaling"] * dap_jac[:,1] / dap

        # contribution of the jacobian of the transformation
        output[:, 0] += self.A.logmetapr_jac(a)
        output[:, 1] += self.B.logmetapr_jac(b)
        return output
    
    def logmetapr_hess( self, var ) :
        ''' Hessian of the logarihtm of metaprior '''
        # pre-load variables
        a = np.array(var[0]).reshape(-1)
        b = np.array(var[1]).reshape(-1)
        dap = self.diverg_apriori(a, b)
        dap_jac = self.diverg_apriori_jac(a, b)
        dap_hess = self.diverg_apriori_hess(a, b)

        # contribution of the marginalization constraint phi
        if self.use_phi is True :
            output = self.log_marginaliz_phi_hess(a, b)
        else :
            output = np.zeros( shape = np.shape(dap_hess) )
        
        # contribution of the prior on the divergence rho
        if self.choice in ["scaled"] :
            tmp = dap_hess[:,0,0] * np.power( self.A.apriori(a), -1)
            tmp -= 2 * dap_jac[:,0] * self.A.drv_1(a) * np.power( self.A.apriori(a), -2)
            tmp += 2 * dap * np.power(self.A.drv_1(a), 2) * np.power( self.A.apriori(a), -3) 
            tmp -= dap * self.A.drv_2(a) * np.power( self.A.apriori(a), -2) 
            output[:,0,0] -= self._extra["scaling"] * tmp
            aux_sym = dap_hess[:,0,1] * np.power( self.A.apriori(a), -1) 
            aux_sym -= dap_jac[:,1] * self.A.drv_1(a) * np.power( self.A.apriori(a), -2)
            output[:,0,1] += self._extra["scaling"] * aux_sym
            output[:,1,0] += self._extra["scaling"] * aux_sym
            output[:,1,1] -= self._extra["scaling"] * dap_hess[:,1,1] / self.A.apriori(a)
        elif self.choice in ["uniform"] :
            mask = dap < np.log(self.K)
            output[ ~mask,:,: ] = NUMERICAL_ZERO
            output[ dap >= CUTOFFRATIO * np.log(self.K),:,: ] = - NUMERICAL_INFTY
        elif self.choice in ["log-uniform"] :
            output[:,0,0] -= self._extra["scaling"] * (dap_hess[:,0,0] / dap - np.power(dap_jac[:,0] / dap, 2))
            aux_sym = self._extra["scaling"] * (dap_hess[:,0,1] / dap - dap_jac[:,0] * dap_jac[:,1] / np.power(dap, 2)) 
            output[:,0,1] -= aux_sym
            output[:,1,0] -= aux_sym
            output[:,1,1] -= self._extra["scaling"] * (dap_hess[:,1,1] / dap - np.power(dap_jac[:,1] / dap, 2))
        
        # contribution of the jacobian of the transformation
        output[:,0,0] += self.A.logmetapr_hess(a)
        output[:,1,1] += self.B.logmetapr_hess(b)
        return output

# <<<<<<<<<<<<<<<<<<<<<<<<
#  MAXIMUM A POSTERIORI  #
# >>>>>>>>>>>>>>>>>>>>>>>>

def minimize(myfunc, var, args=(), bounds=None, jac=None) :
    '''General minimization wrapper for `myfunc`.'''
    
    if USE_JAC_OPT == False : jac = None
    results = optimize.minimize(
        myfunc,
        x0=var, args=args,
        jac=jac,
        method=METHOD, bounds=bounds, 
        options={'maxiter': MAX_ITER}, tol=TOL
        )
    '''
    if np.any([ np.any(np.isclose(x, b, atol=TOL)) for x,b in zip(results.x, bounds) ]) :
        warnings.warn("The optimal parameter(s) saturated to the boundary.")
    '''
    return results.x

def optimal_polya_param( cpct_exp ) :
    '''.'''
    # NOTE : this can be improved using the exact formula
    def myfunc(var, *args) :
        return - Polya(*args).log(var)
    def myjac(var, *args) :
        return - Polya(*args).log_jac(var)
    return minimize(myfunc, [INIT_GUESS], args=(cpct_exp,), bounds=(BOUND_DIR,), jac=myjac)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>
#  ONE dim meta likelihood  #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>

class one_dim_meta_likelihood( ) :
    '''This class deals with the 1D meta-likelihood (or likelihood of the hyperparameter)'''
    def __init__(self, cpct_exp, dir_meta_obj) :
        '''
        Parameters 
        ----------
        cpct_div
        Dir_meta_obj
        '''
        self.dir_meta_obj = dir_meta_obj
        self.polya = Polya( cpct_exp )

    def neglog(self, a) :
        '''Negative of the log-meta-likelihood.'''
        LogLike = self.dir_meta_obj.logmetapr(a)
        LogLike += self.polya.log(a)
        return - LogLike
    def neglog_jac(self, a) :
        '''Negative of the gradient of the log-meta-likelihood.'''
        jac_LogLike = self.dir_meta_obj.logmetapr_jac(a)
        jac_LogLike += self.polya.log_jac(a)
        return - jac_LogLike
    def neglog_hess(self, a) :
        '''Negative of the Hessian of the log-meta-likelihood.'''
        hess_LogLike = self.dir_meta_obj.logmetapr_hess(a)
        hess_LogLike += self.polya.log_hess(a)
        return - hess_LogLike
    def maximize(self, init_var) :
        '''.'''
        return minimize( self.neglog, init_var, jac=self.neglog_jac, bounds=(BOUND_DIR,) )

    '''
    Negative log-meta-likelihood for maximization in logscale.
    '''
    def lgscl_neglog(self, lgscl_var) :
        return self.neglog(np.exp(lgscl_var)) - lgscl_var
    def lgscl_neglog_jac(self, lgscl_var) :
        return np.exp(lgscl_var) * self.neglog_jac(np.exp(lgscl_var)) - 1
    def lgscl_neglog_hess(self, lgscl_var) :
        return self.lgscl_neglog_jac(lgscl_var) + 1 + np.exp(2*lgscl_var) * self.neglog_hess(np.exp(lgscl_var))
    def lgscl_maximize(self, init_var) :
        return minimize(self.lgscl_neglog, init_var, jac=self.lgscl_neglog_jac, bounds=None )
    '''
    Negative log-meta-likelihood for maximization in logscale.
    '''


# >>>>>>>>>>>>>>>>>>>>>>>>>>>
#  TWO dim meta likelihood  #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>

class two_dim_meta_likelihood( ) :
    '''This class deals with the 2D meta-likelihood (or likelihood of the hyperparameters)'''
    def __init__( self, cpct_div, dir_meta_obj, ) :
        '''
        Parameters 
        ----------
        cpct_div
        Dir_meta_obj
        '''
        self.polya_1 = Polya( cpct_div.compact_1 )
        self.polya_2 = Polya( cpct_div.compact_2 )
        self.dir_meta_obj = dir_meta_obj

    '''
    Negative log-meta-likelihood for maximization.
    '''

    def neglog( self, var ) :
        '''Negative of the log-meta-likelihood.'''
        a = np.array(var[0]).reshape(-1)
        b = np.array(var[1]).reshape(-1)
        log_like = np.add.outer(self.polya_1.log(a), self.polya_2.log(b))
        X, Y = np.meshgrid(a, b)
        log_like += self.dir_meta_obj.logmetapr([X, Y]).reshape(len(a), len(b))
        return - log_like
    def neglog_jac( self, var ) :
        '''Negative of the gradient of the log-meta-likelihood.'''
        # FIXME :
        # this computation should be vectorialized
        jac_LogLike = self.dir_meta_obj.logmetapr_jac( var )
        jac_LogLike[:,0] += self.polya_1.log_jac( var[0] )
        jac_LogLike[:,1] += self.polya_2.log_jac( var[1] )
        return - jac_LogLike
    def neglog_hess( self, var ) :
        '''Negative of the Hessian of the log-meta-likelihood.'''
        # FIXME :
        # this computation should be vectorialized
        hess_LogLike = self.dir_meta_obj.logmetapr_hess( var )
        hess_LogLike[:,0,0] += self.polya_1.log_hess( var[0] )
        hess_LogLike[:,1,1] += self.polya_2.log_hess( var[1] )
        return - hess_LogLike
    def maximize( self, init_var ) :
        '''.'''
        return minimize( self.neglog, init_var, jac=self.neglog_jac, bounds=(BOUND_DIR,BOUND_DIR) )
    
    '''
    Negative log-meta-likelihood for maximization in logscale.
    '''
    def lgscl_neglog(self, var) :
        a = np.array(var[0]).reshape(-1)
        b = np.array(var[1]).reshape(-1)
        add_on = np.log(np.add.outer(a, b))
        return self.neglog(var) - add_on
    def lgscl_neglog_jac(self, var) :
        # FIXME :
        # this computation should be vectorialized
        output = self.neglog_jac(var)
        output[:,0] *= var[0]
        output[:,1] *= var[1]
        output -= 1.
        return output
    def lgscl_neglog_hess( self, var ) :
        # FIXME :
        # this computation should be vectorialized
        output = self.neglog_hess(var)
        jac = self.lgscl_neglog_jac(var)
        output[:,0,0] *= np.power(var[0], 2)
        output[:,0,1] *= np.prod(var)
        output[:,1,0] *= np.prod(var).prod()
        output[:,1,1] *= np.power(var[1], 2)
        output[:,0,0] += jac[:,0] + 1.
        output[:,1,1] += jac[:,1] + 1.
        return output
    def lgscl_maximize( self, init_var ) :
        return minimize( self.lgscl_neglog, init_var, jac=self.lgscl_neglog_jac, bounds=((BOUND_DIR),)*2 )

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#   POSTERIOR STANDARD DEVIATION  #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def empirical_n_bins( size, categories, ceil_value=200 ) :
    '''Empirical choice for the number of bins (obsolete).'''
    n_bins = max(1, 10 * np.power(categories / size, 2)) 
    n_bins = min(int(n_bins), ceil_value) 
    return n_bins

# FIXME !!!
def lgscl_binning(loc, lgscl_std, n_bins, bounds=BOUND_DIR, n_sigma=3) :
    '''.'''
    assert (loc > bounds[0]) and (loc < bounds[1])
    assert n_bins%2 == 1
    lgscl_loc = np.log(loc)
    lgscl_bound = np.log(bounds)
    output = np.append(
        np.linspace( max(lgscl_bound[0], lgscl_loc-n_sigma*lgscl_std), lgscl_loc, n_bins//2+1 )[:-1],
        np.linspace( lgscl_loc, min(lgscl_bound[1], lgscl_loc+n_sigma*lgscl_std), n_bins//2+1 )
        )
    return np.exp(output)

def centered_logspaced_binning(loc, std, n_bins, bounds=BOUND_DIR, n_sigma=3) :
    '''.'''
    assert (loc > bounds[0]) and (loc < bounds[1])
    assert n_bins%2 == 1
    output = np.append(
        np.logspace( np.log10(max(bounds[0], loc-n_sigma*std)), np.log10(loc), n_bins//2+1 )[:-1],
        np.logspace( np.log10(loc), np.log10(min(bounds[1], loc+n_sigma*std)), n_bins//2+1 )
        )
    return output

