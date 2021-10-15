#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) October 2021 Francesco Camaglia, LPENS 
'''

import warnings

import numpy as np
import pandas as pd

from utils import dict_generator
from kamapack import default_entropy

######################
#  EXPERIMENT CLASS  #
######################

class Experiment :
    
    def __init__( self, data, categories=None, iscount=False ):
        '''
        TO BE UPDATED
        Parameters
        ----------      
        '''
        #  OPEN : data  #
        if type( data ) == dict :   
            # loading dictionary where values represent keys counts                                    
            data_hist = pd.Series( data )
        elif type( data ) == pd.Series :
            # loading serie where values represent index counts
            data_hist = data
        elif type( data ) == list or type( data ) == np.ndarray :                              
            if iscount == True :
                # loading list of counts
                obs = { k:v for k,v in enumerate(data) }
            else :
                # loading raw list of sequences
                obs = dict_generator( data )   
            data_hist = pd.Series( obs )
        else :
            raise TypeError("The parameter `data` has unsopported type.")

        # check int counts # WARNING! : this try/except doesn't looks nice
        try :
            data_hist = data_hist.astype(int)
        except ValueError :
            raise ValueError("Unrecognized count value in `data`.")    

        self.data_hist = data_hist
        self.tot_counts = np.sum( data_hist.values )
        self.counts_hist = pd.Series( dict_generator( data_hist.values ) )        
        self.obs_n_categ = len( data_hist ) # observed categories

        #  OPEN : categories  #
        if categories is None:
            self.usr_n_categ = self.obs_n_categ    
        else :
            self.update_categories( categories )
                
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
                The new number of categories of the system. If the value is lower than observed number of categories, 
                the observed number is used instead. Default is observed number.
        '''
        try : 
            categories = int( categories )
        except :        
            raise TypeError("The parameter `categories` must be an integer.")
        if categories > self.obs_n_categ :
            self.counts_hist[ 0 ] = categories - self.obs_n_categ
            self.usr_n_categ = categories
        else :
            self.usr_n_categ = self.obs_n_categ  
            if categories < self.obs_n_categ :
                warnings.warn("The parameter `categories` is set equal to the observed number of categories.")
    
    def show( self ):
        '''
        To print a short summary of the Experiment.
        '''
        
        print("Total number of counts: ", self.tot_counts)
        print("Number of Categories: ", self.usr_categ, ' (a priori) | ', self.obs_categ, ' (observed)')
        print("Recurrencies: ", self.counts_hist )
        
    def compact( self )
        
        return Compact( self )

    def entropy( self, method, unit="ln", **kwargs ):
        '''
        A wrapper for the Shannon entropy estimation over a given Experiment class object through a chosen "method".
        The unit of the logarithm can be specified through the parameter "unit".
            
        Parameters
        ----------
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
        
        return default_entropy.switchboard( self.compact(), method, unit=unit, **kwargs )

###################
#  COMPACT CLASS  #
###################

class Compact :
    '''
    It provides aliases useful for computations.
    '''
    def __init__( self, experiment ) :
        self.N = experiment.tot_counts                                   # total number of counts
        self.K = experiment.usr_n_categ                                  # user number of categories
        self.Kobs = experiment.obs_n_categ                               # observed number of categories
        self.nn = experiment.counts_dict.index.values                    # counts
        self.ff = experiment.counts_dict.values                          # recurrency of counts
    
######################
#  DIVERGENCE CLASS  #
######################

#class Divergence :
    
#    def __init__( self, experiment_A, experiment_B ) :
        
    