'''
functions for the asymptotic limit of the NSB estimators
'''

import numpy as np
from kamapack.entropy import nsb

###
def NSB_low_coinc_est(experiment):

    # loading quantities from class
    N = experiment.N
    D = experiment.N - experiment.obs_categ
    
    # expansion for small numbers of coincidences
    C_gamma = -nsb.psi( 0, 1 ) 
    S = C_gamma - np.log(2) + 2 * np.log(N) - nsb.psi(0,D)
    S_devStd = np.sqrt( nsb.psi( 1, D ) )

    return np.array([S, S_devStd])
###



### 
def NSB_sad_pnt_est( experiment ):

    # loading parameters from experiment 
    N = experiment.N                                                # total number of counts
    K = experiment.usr_categ                                        # number of given categories
    K_obs = experiment.obs_categ                                    # number of observed categories
    nn = np.array( list( experiment.counts_dict.keys( ) ) )         # counts
    ff = np.array( list( experiment.counts_dict.values( ) ) )       # recurrency of counts
    
    # saddle-point estimation of the beta
    dd = 1 - K_obs / N
    b_m1 = 0.5 * ( N - 1 ) / N
    b_0 = ( 1 - 2*N ) / ( 3*N )
    b_1 = ( N - 1 - 2 / N ) / ( 9 * ( N - 1 ) )
    ki_0 = N * ( b_m1 / dd + b_0 + b_1 * dd )

    # evaluation of the integral at the saddle
    obsMask = nn > 1
    ki_Z = (K_obs/ki_0**2+nsb._Del_psi(1,ki_0+N,ki_0))
    ki_1 = np.dot(ff[obsMask],(nsb._Del_psi(0,nn[obsMask],1)))/ki_Z
    ki_2 = ((K_obs/ki_0**3+0.5*nsb._Del_psi(2,ki_0,ki_0+N))*ki_1**2+ki_0*np.dot(ff[obsMask],(nsb._Del_psi(1,nn[obsMask],1))))/ki_Z
    beta_sad_pnt = ki_0/K+ki_1/K**2+ki_2/K**3
        
    trash, S, S2 = nsb._estimator_beta( beta_sad_pnt, experiment )    
    S_devStd = np.sqrt( S2 - np.power( S ,2 ) )
    
    return np.array([S, S_devStd])
###
