'''
NEMENMAN_SHAFEE_BIALEK Estimator
Francesco Camaglia, LPENS February 2020
Version 07/01/2020
'''

import numpy as np
from mpmath import mp                   # for more precise exponential
from scipy.special import loggamma
from scipy.special import polygamma as psi
import scipy.optimize as opt
import multiprocessing


###
def NemenmanShafeeBialek( experiment, bins = None, err = False ):
    '''
    NSB_entropy Function Description:
    '''
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>
    
    if bins == None:
        n_csi_bins = int( np.log( experiment.N ) * 1e2 )       
        # EMPIRICAL: choice of the number of bins for numerical integration
    elif type(bins) == int :
        n_csi_bins = bins
    else:
        raise TypeError("The number of bins requires an integer value.")

    
    # >>>>>>>>>>>>>>>>
    #  COMPUTE beta  #
    # >>>>>>>>>>>>>>>>
        
    POOL = multiprocessing.Pool( multiprocessing.cpu_count() )   
    args = [ (x + .5, n_csi_bins, experiment, ) for x in range(n_csi_bins) ]
    beta_vec = POOL.starmap( _beta_, args )
    POOL.close()
    
    beta_vec = np.asarray( beta_vec )
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  estimators vs beta  #
    # >>>>>>>>>>>>>>>>>>>>>>
    
    POOL = multiprocessing.Pool( multiprocessing.cpu_count() ) 
    args = [ ( beta , experiment, err ) for beta in beta_vec ]
    results = POOL.starmap( _estimator_beta, args )
    POOL.close()
    
    results = np.asarray( results )
    
    # >>>>>>>>>>>>>>>
    #   estimators  #
    # >>>>>>>>>>>>>>>
    
    Zeta = np.sum( results[:,0] )        

    integral_S1 = np.dot( results[:,0],  results[:,1] )
    S = mp.fdiv( integral_S1, Zeta )     
    # NOTE: the normalization integral is computed on the same bins 
    #       which simplifies the bin size 

    if err == True :
        integral_S2 = np.dot( results[:,0],  results[:,2] )
        S2 = mp.fdiv( integral_S2, Zeta )
        S_devStd = np.sqrt( S2 - np.power(S,2) )
        shannon_estimate = np.array([S, S_devStd], dtype=np.float)      
    else :       
        shannon_estimate = np.array(S, dtype=np.float)        

    return shannon_estimate
###



###
def _csi_vs_beta( beta, csi, K ):
    '''
    csi=csi(beta) implicit relationto be inverted.
    '''
    
    return psi( 0, K * beta + 1 ) - psi( 0, beta + 1 ) - csi        
###


###
def _beta_( x, n_csi_bins, experiment, maxiter=100 ):
    '''
    Numerical integration on csi in (0, infty) requires to compute
    eta from brentq algorithm on the implicit relation.
    '''
    
    # number of given categories
    K = experiment.usr_categ      
    
    up_bound = np.log(K) * 5e2
    # EMPIRICAL: right edge of the interval as approx of infty (WARNING:)
    
    xtol = ( 1e-2 ) / ( K * n_csi_bins )                    
    # EMPIRICAL: tolerance for brentq (WARNING:)
    
    _csi_ = x * np.log(K) / n_csi_bins
    output = opt.brentq( _csi_vs_beta, 0, up_bound, args=( _csi_, K), xtol=xtol, maxiter=maxiter )
    
    return output
###

###
def _Lgmm( x ):
    return loggamma( x ).real                                       # real part of loggamma function
###

###
def _Del_psi(order, x, y):
    return psi(order,x)-psi(order,y)                                # difference between same order psi functions
###

###
def _estimator_beta( b, experiment, err ):
    '''
    '''
    
    # loading parameters from experiment        
    N = experiment.N                                                # total number of counts
    K = experiment.usr_categ                                        # number of given categories
    nn = np.array( list( experiment.counts_dict.keys( ) ) )         # counts
    ff = np.array( list( experiment.counts_dict.values( ) ) )       # recurrency of counts
    
    # mu computation    
    lnmu = _Lgmm( K*b ) - _Lgmm( N + K*b ) + np.dot( ff, _Lgmm(nn+b) ) - K * _Lgmm( b )
    mu_b = mp.exp( lnmu )
    
    # entropy computation
    S1_temp = np.dot( ff, (nn+b) * _Del_psi( 0, N+K*b+1, nn+b+1 ) )     
    S1_b = mp.fdiv( S1_temp, N + K*b )
    
    # squared entropy computation if required
    # WARNING: double check
    if err == True :
        temp = np.zeros(len(ff))
        for i in range( len(ff) ):   
            temp[i] = np.dot( ff,(nn+b)*(nn[i]+b)*_Del_psi(0,nn+b+1,N+K*b+2)*(_Del_psi(0,nn[i]+b+1,N+K*b+2)-psi(1,N+K*b+2)))
        S2_temp1 = np.dot( ff, temp )
        
        S2_temp1_corr = np.dot( ff, np.power(nn+b,2)*(np.power(_Del_psi(0,nn+b+1,N+K*b+2),2)-psi(1,N+K*b+2)) )
        S2_temp2 = np.dot( ff, (nn+b)*(nn+b+1)*(np.power(_Del_psi(0,nn+b+2,N+K*b+2),2)+_Del_psi(1,nn+b+2,N+K*b+2))) 
        S2_temp = S2_temp1 - S2_temp1_corr + S2_temp2 
        S2_b = mp.fdiv( S2_temp, mp.fmul(N + K*b+1,N + K*b) )

        output_b = np.array( [ mu_b, S1_b, S2_b ] )
    else :
        output_b = np.array( [ mu_b, S1_b ] )

    return output_b
###
