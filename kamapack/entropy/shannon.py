'''
Utils Module:
Francesco Camaglia, LPENS February 2020
'''

import numpy as np
from scipy.special import comb

from kamapack.entropy import nsb
from kamapack.utils import dict_generator, hist_generator

# loagirthm unit
_unit_Dict_ = { "log2": 1. / np.log(2), "ln": 1., "log10": 1. / np.log(10) }

######################
#  EXPERIMENT CLASS  #
######################

###
class experiment:
    '''
    A class used as a wrapper to work on single sequences, from which one can easily compute entropy.

    Parameters
    ----------

    data: list/dictionary
        a list of homogeneous observations or a dictionary { observations : counts }.
        TO BE UPDATED
    categories: scalar, optional
        number of categories of the system. If value lower than observed number of categories,
        the observed number is used instead. Default is observed number.
    iscount: bool, optional
        TO BE UPDATED
        
    Attributes
    ----------
    N
    counts_dict
    obs_categ
    usr_categ
    '''
    
    def __init__( self, data, categories=None, iscount=True ):
        
        # initializing data
        if type( data ) == dict :                                       
            # loading dictionary where values represent keys counts
            observations = np.array( list( data.values() ) )
            observations = observations[ np.nonzero(observations) ]
        else :
            try :
                data = np.array( data )                                    
                if iscount == True :
                    # loading list of counts
                    observations = data[ np.nonzero(data) ]
                else :
                    # loading raw list of sequences
                    observations = hist_generator( data ) 
            except :
                raise TypeError('Parameter "data" requires a dictionary or a list.')

        self.N = np.sum( observations )
        self.counts_dict = dict_generator( observations )

        # initializing categories
        self.obs_categ = len( observations )

        if categories == None:
            self.usr_categ = self.obs_categ    
        else :
            try : 
                categories = int( categories )
            except :        
                raise IOError( 'The parameter categories must be an integer.')  
            if categories >= self.obs_categ :
                self.counts_dict[ 0 ] = categories - len( observations )
                self.usr_categ = categories
            else :
                self.usr_categ = self.obs_categ  
                print('Categories cannot be less than observed categories.')
                
    '''
    Methods
    -------
    '''

    def update_categories( self, categories ):
        '''
        To change the number of categories.

        Parameters
        ----------        
        
        categories: scalar, optional
                the new number of categories (i.e. diversity) of the system. 
                If the value is lower than observed number of categories,
                the observed number is used instead. Default is observed number.
        '''

        if  type( categories ) == int:
            if categories < self.obs_categ:
                raise IOError("Categories must be greater than observed categories." )
            else:
                self.usr_categ = categories
                self.counts_dict[0] = categories - self.obs_categ
        else:
            raise TypeError("Categories requires an int value.")    

    # >>>>>>>>
    #  SHOW  #
    # >>>>>>>>
    
    def show( self ):
        '''
        To print a short summar of the experiment.
        '''
        
        print("Number of sequences: ", self.N)
        print("Categories: ", self.usr_categ, ' (a priori) | ', self.obs_categ, ' (observed)')
        print("Recurrencies: ", self.counts_dict )

    # to compute entropy
    def entropy( self, method, unit="ln", **kwargs ):
        '''
        A wrapper for the Shannon entropy estimation over a given experiment class object through a chosen "method".
        The unit of the logarithm can be specified through the parameter "unit".
            
        Parameters
        ----------

        experiment: experiment object
                an experiment class object already initialized with the observation vector.
        method: str
                the name of the entropy estimation method:
                - "ML": Maximum Likelihood entropy estimator;
                - "MM": Miller Madow entropy estimator;
                - "CS": Chao Shen entropy estimator;       
                - "shrink": shrinkage entropy estimator;       
                - "Jeffreys": Jeffreys entropy estimator;
                - "Laplace": Laplace entropy estimator;
                - "SG": Schurmann-Grassberger entropy estimator;
                - "minimax": minimax entropy estimator;
                - "NSB": Nemenman Shafee Bialek entropy estimator.
        unit: str, optional
                the entropy logbase unit:
                - "log2": base 2 logairhtm (default);
                - "ln": natural logarithm;
                - "log10":base 10 logarithm.

        return numpy.array
        '''
        return _entropy( self, method, unit=unit, **kwargs )
###



########################
#  ENTROPY ESTIMATION  #
########################

def _entropy( experiment, method, unit=None, **kwargs ):

    # loading units
    if unit in _unit_Dict_.keys( ) :
        unit_conv = _unit_Dict_[ unit ]
    else:
        raise IOError("Unknown unit, please choose amongst ", _unit_Dict_.keys( ) )

    # choosing entropy estimation method
    if method == "ML":                          # Maximum Likelihood
        shannon_estimate = maxlike( experiment )
    elif method == "MM":                        # Miller Madow
        shannon_estimate = MillerMadow( experiment )
    elif method == "CS":                        # Chao Shen       
        shannon_estimate = ChaoShen( experiment )
    elif method == "Jeffreys":                  # Jeffreys
        a = 0.5
        shannon_estimate = Dirichlet( experiment, a )
    elif method == "Laplace":                   # Laplace
        a = 1.
        shannon_estimate = Dirichlet( experiment, a )
    elif method == "SG":                        # Schurmann-Grassberger
        a = 1. / experiment.obs_categ
        shannon_estimate = Dirichlet( experiment, a )
    elif method == "minimax":                   # minimax
        a = np.sqrt( experiment.N ) / experiment.obs_categ
        shannon_estimate = Dirichlet( experiment, a )
    elif method == "NSB":                       # Nemenman Shafee Bialek
        shannon_estimate = nsb.NemenmanShafeeBialek( experiment, **kwargs )
    else:
        raise IOError("The chosen method is unknown.")

    return unit_conv * shannon_estimate
###



##################################
#  MAXIMUM LIKELIHOOD ESTIMATOR  #
##################################

def maxlike( experiment ):
    '''
    Maximum likelihood estimator.
    '''

    # loading parameters from experiment 
    N = experiment.N                                    # total number of counts
    local_dict = experiment.counts_dict.copy()
    if 0 in local_dict :                                # check 0 counts
        del local_dict[ 0 ]                             # delete locally 0 counts
    nn = np.array( list( local_dict.keys( ) ) )         # counts
    ff = np.array( list( local_dict.values( ) ) )       # recurrency of counts
    
    shannon_estimate = np.array( np.log( N ) - np.dot( ff , np.multiply( nn, np.log( nn ) ) ) / N )
    return shannon_estimate
###



############################
#  MILLER MADOW ESTIMATOR  #
############################

def MillerMadow( experiment ): 
    '''
    Miller-Madow estimator.
    '''

    N = experiment.N                    # total number of counts
    M = experiment.obs_categ            # number of bins with non-zero counts: obs_categ

    shannon_estimate = np.array( maxlike( experiment ) + 0.5 * ( M - 1 ) / N )
    return shannon_estimate 
###



#########################
#  CHAO SHEN ESTIMATOR  #
#########################

def ChaoShen( experiment ):
    '''
    Compute Chao-Shen (2003) entropy estimator 
    WARNING!: TO BE CHECKED
    '''

    def __coverage( nn, ff ) :
        '''
        Good-Turing frequency estimation with Zhang-Huang formulation
        '''
        N = np.dot( nn, ff )
        # Check for the pathological case of all singletons (to avoid coverage = 0)
        # i.e. nn = [1], which means ff = [N]
        if ff[ np.where( nn == 1 )[0] ] == N :  
            # this correpsonds to the correction ff_1=N |==> ff_1=N-1
            GoodTuring = ( N - 1 ) / N                                  
        else :
            sign = np.power( -1, nn + 1 )
            binom = list( map( lambda k : 1. / comb(N,k), nn ) )
            GoodTuring = np.sum( sign * binom * ff )
            
        return 1. - GoodTuring
    ###
    
    # loading parameters from experiment class
    local_dict = experiment.counts_dict.copy()
    if 0 in local_dict :  del local_dict[ 0 ]           # delete 0 counts locally                   
    nn = np.array( list( local_dict.keys() ) )         # counts
    ff = np.array( list( local_dict.values() ) )       # recurrency of counts
    N = np.dot( nn, ff )                                # total number of counts

    C = __coverage( nn, ff )                            
    p_vec = C * nn / N                                # coverage adjusted empirical frequencies
    lambda_vec = 1. - np.power( 1. - p_vec, N )         # probability to see a bin (specie) in the sample

    shannon_estimate = np.array( - np.dot( ff , p_vec * np.log( p_vec ) / lambda_vec ) )
    return shannon_estimate 
###



##########################
#  DIRICHELET ESTIMATOR  #
##########################

def Dirichlet( experiment, a ):
    '''
    Estimate entropy based on Dirichlet-multinomial pseudocount model.
    a:  pseudocount per bin
    a=0          :   empirical estimate
    a=1          :   Laplace
    a=1/2        :   Jeffreys
    a=1/M        :   Schurmann-Grassberger  (M: number of bins)
    a=sqrt(N)/M  :   minimax
    '''

    # loading parameters from experiment 
    N = experiment.N                                    # total number of counts
    nn = np.array( list( experiment.counts_dict.keys( ) ) )         # counts
    ff = np.array( list( experiment.counts_dict.values( ) ) )       # recurrency of counts

    nn_a = nn + a                                       # counts plus pseudocounts
    N_a = N + a * np.sum( ff )                          # total number of counts plus pseudocounts

    shannon_estimate  = np.array(  np.log( N_a ) - np.dot( ff , nn_a * np.log( nn_a ) ) / N_a )
    return shannon_estimate  
###

