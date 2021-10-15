'''
NEMENMAN_SHAFEE_BIALEK Estimator
Francesco Camaglia, LPENS February 2020
Version 07/01/2020
'''

import numpy as np
from mpmath import mp                   # for more precise exponential
from scipy.special import loggamma
from scipy.special import polygamma
import scipy.optimize as opt
import multiprocessing


###
def NemenmanShafeeBialek( experiment, bins= None, err = False ):
    '''
    NSB_entropy Function Description:
    '''

    # loading parameters from experiment 
    N = experiment.tot_counts                           # total number of counts
    K = experiment.usr_n_categ
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  CHECK user OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>
    
    if bins == None:
        # EMPIRICAL: choice of the number of bins for numerical integration
        n_bins = int( np.log(N) * 1e2 )       
    else :
        try :
            n_bins = int(bins)
        except :
            raise TypeError("The parameter `bins` requires an integer value.")

    # >>>>>>>>>>>>>>>>>
    #  Compute Alpha  #
    # >>>>>>>>>>>>>>>>>

    # multiprocessing (WARNING:)
    POOL = multiprocessing.Pool( multiprocessing.cpu_count() )   
    args = [ ( implicit_S_vs_Alpha, x + 0.5, n_bins, K ) for x in range(n_bins) ]
    Alpha_vec = POOL.starmap( get_from_implicit, args )
    POOL.close()
    Alpha_vec = np.asarray( Alpha_vec )
    
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

##############
#  NOTATION  #
##############

def Delta_polyGmm(order, x, y):
    '''
    Difference between same `order` polygamma functions, computed in `x` and `y`.
    '''
    return polygamma(order,x) - polygamma(order,y)                                                     

def implicit_S_vs_Alpha( alpha, x, K ):
    '''
    implicit relation to be inverted.
    '''
    return Delta_polyGmm( 0, K * alpha + 1, alpha + 1 ) - x   

#################
#  _MEASURE MU  #
#################

def LogGmm( x ): 
    ''' alias '''
    return loggamma( x ).real    

def measureMu( b, experiment ) :
    '''
    Measure Mu term in the posterior estimators computed as the exponent of an exponential.
    '''
        
    # DEAFULT : loading parameters from experiment        
    N = experiment.N                                            # total number of counts
    K = experiment.usr_n_categ                                  # number of given categories
    nn = np.array( list( experiment.counts_dict.keys( ) ) )     # counts
    ff = np.array( list( experiment.counts_dict.values( ) ) )   # recurrency of counts
    
    # mu computation    
    LogMu = LogGmm( K*b ) - K * LogGmm( b )                   # Dirichelet prior normalization contribution
    LogMu += np.dot( ff, LogGmm(nn+b) ) - LogGmm( N + K*b )   # posterior contribution

    return mp.exp( LogMu )

########################
#  get_from_implicit  #
########################

def get_from_implicit( implicit_relation, x, n_bins, K, maxiter=100 ):
    '''
    Numerical integration in (0, log(K)), domain of the explicit variable, 
    requires to solve an implicit relation to get the implicit variable  in(0, infty)
    ''' 

    # bounds for the implicit variable
    lower_bound = 0
    # EMPIRICAL: right edge of the interval as approx of infty (WARNING:)
    upper_bound = np.log(K) * 1.e3    
    # EMPIRICAL: tolerance for brentq (WARNING:)
    xtol = 1.e-2  / ( K * n_bins )        
    # arguments of `implicit_relation` are explicit variable and categories       
    args = ( np.log(K) * x / n_csi_bins , K )

    # brentq algorithm for implicit relation
    # NOTE : the implicit_realtion must have opposite signs in 0 and up_bound
    output = opt.brentq( implicit_relation, lower_bound, upper_bound, 
                        args=args, xtol=xtol, maxiter=maxiter )
    
    return output
###



####################################
#  estimation vs Dirichelet param  #
####################################



###
def estimate_at_alpha( a, experiment, err ):
    '''
    '''
    
    # loading parameters from experiment        
    N = experiment.N                                                # total number of counts
    K = experiment.usr_categ                                        # number of given categories
    nn = np.array( list( experiment.counts_dict.keys( ) ) )         # counts
    ff = np.array( list( experiment.counts_dict.values( ) ) )       # recurrency of counts
    
    mu_a = measureMu( a, experiment )
    
    # entropy computation
    temp = np.dot( ff, (nn+a) * Delta_polyGmm(0, N+K*a+1, nn+a+1) )     
    S1_a = mp.fdiv( temp, N + K*a )
    
    # squared entropy computation if required
    # WARNING: double check
    if err is True :
        S2_temp1 = np.zeros(len(ff))
        for i in range( len(ff) ):   
            temp = (nn+a) * (nn[i]+a) * Delta_polyGmm(0, nn+a+1, N+K*a+2) * ( Delta_polyGmm(0, nn[i]+a+1, N+K*a+2) - polygamma(1, N+K*a+2) )
            S2_temp1[i] = np.dot( ff, temp )
        
        S2_temp1_corr = np.power(nn+a, 2) * ( np.power(Delta_polyGmm(0, nn+a+1, N+K*a+2), 2) - polygamma(1, N+K*a+2) )
        S2_temp2 = (nn+a) * (nn+a+1) * ( np.power(Delta_polyGmm(0, nn+a+2, N+K*a+2), 2) + Delta_polyGmm(1, nn+a+2, N+K*a+2) ) 
        S2_temp = np.dot( ff, S2_temp1 - S2_temp1_corr + S2_temp2 )
        S2_a = mp.fdiv( S2_temp, mp.fmul(N + K*a+1, N + K*a) )

        output = np.array( [ mu_a, S1_a, S2_a ] )
        
    else :
        output = np.array( [ mu_a, S1_a ] )

    return output
###
