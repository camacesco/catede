
'''
Uniform_Histograms Library:
Francesco Camaglia, LPENS June 2020
'''

# coding: utf-8

import numpy as np
import pandas as pd

from kamapack.graphics import easy_plot
from kamapack.utils import dict_generator

#################
#  ALBUM CLASS  #
#################

class Album:
    '''    
    This class defines objects containing histograms/curves/cakes of a single random variable computed at different stages.
    The data are loaded through the function "Load_Func" to be specified. It must be a function of the index k in range(n_sets). 
    One between "n_sets" and "headers" must be specified. If both are specified, they must satisfy len(headers)=n_sets
    V V

    Attributes
    ----------
    Load_Func: function
            The function specified must return an array like variable containing data for the set k;
            its optional arguments can be specified through additional **kwargs.
    n_sets: integer, optional
            The number of different sets tobe loaded.
    headers: list, optional
            A list with the names of the different sets (default is ["Hist 1", "Hist 2", ...]);
            at least one between n_sets and headers must be specified
    '''

    def __init__(self, LoadFunc, n_sets=None, headers=None, **kwargs):

        new_headers = _optional_input_check_( n_sets=n_sets, headers=headers )
        self.headers = new_headers
        self.n_sets = len( new_headers )
    
    # Divergence Method
    def divergence(
        self,
        unit=None, xlabel=None, ylabel=None, title=None, xlim=None, 
        color=None, tags=False, fileout=None, cmap_bounds=None, 
        **owargs
        ):
        
        return easy_plot.divergence(
            self, 
            unit=unit, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, 
            color=color, tags=tags, fileout=fileout, cmap_bounds=cmap_bounds, 
            **owargs
            )
    
    # Display method
    def display(
        self,
        n_cols=None, xlabel=None, ylabel=None, title=None, xlim=None, 
        color=None, tags=False, fileout=None, **owargs
        ):
        
        return easy_plot.display(
            self, 
            n_cols=n_cols, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, 
            color=color, tags=tags, fileout=fileout, **owargs
            )
###

##################
#  ALBUM CURVES  #
##################

class Curves( Album ) :    
    '''
    It returns a pandas data frame with "n_sets" curves meant to be on the same x_array.
    This is useful for homogeneous data that can be loaded alltogether.
    ''' 

    def __init__( self, LoadFunc, x_array=None, n_sets=None, headers=None, **kwargs ):

        # inheritance from Album class
        super().__init__( LoadFunc, n_sets=n_sets, headers=headers, **kwargs )

        try :
            x_array = np.array( x_array )
        except :
            raise IOError('the "x_array" is an array or list of floats to be specified.')

        sizes, df = _uniform_curves_( x_array, LoadFunc, self.headers, **kwargs )   
        self.sizes = dict( zip( headers, sizes ) )     
        self.DataFrame = df
###

###
def _uniform_curves_( x_array, LoadFunc, headers, **kwargs ):
    '''
    It returns a dataframe containing the respective curve for each header
    '''

    n_sets = len( headers )
    
    sizes = []
    # initialize bins and dataframe 
    df = {}
    # assign x_array to bins column 
    df["bins"] = x_array
    
    for k in range( 0, n_sets ):
        loaded = LoadFunc(k, **kwargs)
        sizes.append(len(loaded))
        # if multiple data are loaded together returns the average (used for bootstrap)
        if type( loaded[ 0 ] ) in [ list, np.ndarray ] : 
            if len(loaded) > 1 :
                sum_data_k = np.zeros(len(x_array))
                sum_sq_data_k = np.zeros(len(x_array))
                for data in loaded :
                    sum_data_k += np.array( data )
                    sum_sq_data_k += np.power( data, 2 )
                # average
                df[ headers[k] ] = sum_data_k / len(loaded)    
                # std deviation (of the sample, not of the mean)             
                df[ headers[k] + "-err" ] = devStd_est( len(loaded), sum_data_k, sum_sq_data_k )
            else :
                df[ headers[k] ] = loaded[0]
        else :
            df[ headers[k] ] = loaded
            
    return sizes, pd.DataFrame(df)
###


#################
#  ALBUM HISTS  #
#################

class Hists( Album ) :    
    '''
    It returns a pandas data frame with "n_sets" histograms computed on the same bins of size "bin_size".
    This is useful for homogeneous data that can be loaded alltogether.
    ''' 

    def __init__( self, LoadFunc, yerr=False, bin_size=None, n_sets=None, headers=None, **kwargs ):

        # inheritance from Album class
        super().__init__( LoadFunc, yerr=yerr, n_sets=n_sets, headers=headers, **kwargs )

        try :
            bin_size = float( bin_size )
        except :
            raise IOError('the "bin_size" is a float to be specified.')

        sizes, df = _uniform_histograms_( bin_size, LoadFunc, headers, **kwargs )
        self.sizes = dict( zip( headers, sizes ) )
        self.DataFrame = df
###

###
def _uniform_histograms_(bin_size, LoadFunc, headers, **kwargs):        

    n_sets = len( headers )
    
    # Find uniforming bins
    left = []
    right = []
    for k in range( 0, n_sets ):
        loaded = LoadFunc(k, **kwargs)
        # if multiple data are loaded together returns the average (used for bootstrap)
        if type( loaded[ 0 ] ) in [ list, np.ndarray ] : 
            for data in loaded :
                left.append(np.min(data))
                right.append(np.max(data))
        else :
            left.append(np.min(loaded))
            right.append(np.max(loaded))
    
    sizes = []
    # initialize bins and dataframe 
    df = {}
    # assign bins (left extreme for each interval) for "bins"
    bins = np.min(left) + np.arange(0,np.ceil((np.max(right)-np.min(left))/bin_size), bin_size)
    df["bins"] = bins[:-1]

    for k in range( 0, n_sets ):
        loaded = LoadFunc(k, **kwargs)
        sizes.append(len(loaded))
        # if multiple data are loaded together returns the average (used for bootstrap)
        if type( loaded[ 0 ] ) in [ list, np.ndarray ] : 
            if len(loaded) > 1 :
                sum_data_k = np.zeros(len(bins[:-1]))
                sum_sq_data_k = np.zeros(len(bins[:-1]))
                for data in loaded :
                    hist, _trash = np.histogram( data, density = True, bins = bins )
                    sum_data_k += np.array( hist )
                    sum_sq_data_k += np.power( hist, 2 )
                # average
                df[ headers[k] ] = sum_data_k / len(loaded)    
                # std deviation (of the sample, not of the mean)             
                df[ headers[k] + "-err" ] = devStd_est( len(loaded), sum_data_k, sum_sq_data_k )
            else :
                df[ headers[k] ], _trash  = np.histogram( loaded[0], density = True, bins = bins )
        else :
            df[ headers[k] ], _trash = np.histogram( loaded, density = True, bins = bins )
            
    return sizes, pd.DataFrame(df)
###

 

#################
#  ALBUM RANKS  #
#################

class Ranks( Album ) :    
    '''
    ''' 

    def __init__( self, LoadFunc, ref_head=0, n_sets=None, headers=None, **kwargs ):

        # inheritance from Album class
        super().__init__( LoadFunc, n_sets=n_sets, headers=headers, **kwargs )

        sizes, df, specimens_sorted = _uniform_ranks_( LoadFunc, headers, ref_head=ref_head, **kwargs )
        self.sizes = dict( zip( headers, sizes ) )
        self.DataFrame = df
        self.specimens = specimens_sorted

    # Usage Method
    def usage(
        self,
        xlabel=None, ylabel=None, title=None, thresh=None,
        tags=False, fileout=None, cmap_bounds=None
        ):
        
        return easy_plot.usage(
            self, 
            xlabel=xlabel, ylabel=ylabel, title=title, thresh=thresh,
            tags=tags, fileout=fileout, cmap_bounds=cmap_bounds
            )
###


###
def _uniform_ranks_( LoadFunc, headers, ref_head=0, **kwargs ):        

    n_sets = len( headers )

    # Find list of species
    obs_species = {}
    for k in range( 0, n_sets ):
        loaded = LoadFunc(k, **kwargs)
        # if multiple data are loaded together returns the average (used for bootstrap)
        if type( loaded[ 0 ] ) in [ list, np.ndarray ] : 
            for data in loaded :
                thisDict = dict_generator( data )
                obs_species.update( thisDict )
        else :
            thisDict = dict_generator( loaded )
            obs_species.update( thisDict )

    # the list of all the specimens found to uniform the dataframe
    specimens = list( obs_species.keys() )

    sizes = []
    df = {}
    # temporarily assignment of specimens to bins to ease sorting 
    df["bins"] = specimens

    # STEP k  
    for k in range( 0, n_sets ):
        loaded = LoadFunc(k, **kwargs)
        sizes.append(len(loaded))
        # if multiple data are loaded together returns the average (used for bootstrap)
        # check if list of lists:
        if type( loaded[ 0 ] ) in [ list, np.ndarray ] : 
            if len(loaded) > 1 :
                sum_data_k = np.zeros(len(specimens))
                sum_sq_data_k = np.zeros(len(specimens))
                for data in loaded :
                    usage = dict.fromkeys( specimens, 0 ) 
                    usage.update( dict_generator( data ) )
                    usage_prob = np.array(list(usage.values())) / np.sum(list(usage.values()))
                    sum_data_k += usage_prob
                    sum_sq_data_k += np.power( usage_prob, 2 )
                # average
                df[ headers[k] ] = sum_data_k / len(loaded)    
                # std deviation (of the sample, not of the mean)             
                df[ headers[k] + "-err" ] = devStd_est( len(loaded), sum_data_k, sum_sq_data_k )
            else :
                usage = dict.fromkeys( specimens, 0 )
                usage.update( dict_generator( loaded[0] ) )
                usage_prob = np.array(list(usage.values())) / np.sum(list(usage.values()))
                df[ headers[k] ] = usage_prob
        else :
            usage = dict.fromkeys( specimens, 0 )
            usage.update( dict_generator( loaded ) )
            usage_prob = np.array(list(usage.values())) / np.sum(list(usage.values()))
            df[ headers[k] ] = usage_prob

    df = pd.DataFrame(df).sort_values( by=[ headers[ ref_head ] ], ascending=False ).reset_index( drop=True )
    specimens_sorted = df["bins"]               # savesorted specimens accorded to ranks of ref_head

    # Note: for standard plotting usage, bins can't be specimens names
    df["bins"] = np.arange( 0, len(specimens), 1 )  

    return sizes, df, specimens_sorted
    
#####################################################################################################################

#######################
#  STD.DEV ESTIMATOR  #
#######################

def devStd_est( N, sum_x, sum_sq_x ) :
    return np.sqrt( np.abs( ( sum_sq_x - np.power( sum_x, 2 ) / N ) / ( N - 1 ) ) )


##########################
#  OPTIONAL INPUT CHECK  #
##########################

def _optional_input_check_( n_sets = None, headers = None ) :
    
    if headers == None :
        if n_sets == None :
            raise IOError( "At least one between n_sets and headers list must be specified." )
        elif type(n_sets)!=int:
            raise TypeError( "n_sets requires an integer." )
        elif n_sets < 1:
            raise ValueError( "n_sets must be greater than 0." )
        else:
            new_headers = ["Hist " + str(i) for i in range(0, n_sets)]
    elif type(headers) != list :
        raise TypeError( "headers option requires a list." )
    else : 
        if n_sets == None :
            n_sets = len(headers)   
            new_headers = headers
        elif type(n_sets)!=int:
            raise TypeError( "n_sets requires an integer." )
        elif ( len(headers) != n_sets ):
            raise ValueError( "headers list must have a length equal to n_sets." )    
        else:
            new_headers = headers 
    
    return new_headers
###

    