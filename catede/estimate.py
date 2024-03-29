#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) January 2024 Francesco Camaglia, LPENS 
'''

import warnings
import numpy as np
import pandas as pd
from .dirichlet_multinomial import Experiment_Compact, Divergence_Compact
from . import default_divergence, default_entropy

__entropy_doc__ = '''
        The unit (of the logarithm) can be specified with the parameter `unit`.

        return numpy.array

        Parameters
        ----------
        method: str
            the name of the estimation method:
            - "naive" : naive estimator (default);
            - "MM" : Miller-Madow estimator;
            - "Pe" : Perks pseudocunt estimator;
            - "La" : Laplace pseudocount estimator;
            - "Je" : Jeffreys pseudocount estimator (Krichevsky-Trofimov);
            - "Tr" : Trybula pseudocount estimator (minimax); 
            - "SG" : Schurmann-Grassberger estimator;
            - "CS" : Chao-Shen estimator (coverage-adjusted); 
            - "DC" : Dirichlet categorical estimator;
            - "DP" : Dirichlet prior estimator (maximum evidence);
            - "NSB" : "Nemenmann-Shafee-Bialek".

        unit: str, optional
            the divergence logbase unit:
            - "ln": natural logarithm (default);
            - "log2": base 2 logarihtm;
            - "log10":base 10 logarithm.
'''
        
__diverg_doc__ = '''
        The unit (of the logarithm) can be specified with the parameter `unit`.

        return numpy.array

        Parameters
        ----------
        method: str
            the name of the estimation method:
            - "naive" : naive estimator (default);
            - "La" : Laplace pseudocount estimator;
            - "Je" : Jeffreys pseudocount estimator (Krichevsky-Trofimov);
            - "Tr" : Trybula pseudocount estimator (minimax); 
            - "ZG" : Zhang-Grabchak estimator;
            - "DC" : Dirichlet categorical estimator;
            - "DP" : Dirichlet prior estimator (maximum evidence);
            - "DPM" : Dirichlet prior mixture estimator.
        unit: str, optional
            the divergence logbase unit:
            - "ln": natural logarithm (default);
            - "log2": base 2 logarihtm;
            - "log10":base 10 logarithm.
'''

class Skeleton_Class :
    '''Auxiliary class for Experiment and Divergence.'''

    def update_categories( self, categories ):
        '''Change the number of categories.

        Parameters
        ----------        
        categories : scalar
                The new number of categories of the system (int or float). 
                If the value is lower than the observed number of categories, the observed number is used instead.
        '''

        obs_n_categ = np.max(self.obs_n_categ)
        if categories is None:
            self.usr_n_categ = obs_n_categ    
        else :
            try : 
                categories = int( categories )
            except :        
                raise TypeError("The parameter `categories` must be an integer.")
            if categories > obs_n_categ :
                self.usr_n_categ = categories
            else :
                self.usr_n_categ = obs_n_categ 
                if categories < obs_n_categ :
                    warnings.warn("The parameter `categories` is set equal to the observed number of categories.")
        self._fix_zero_counts()
        pass
        
    def show( self ):
        '''Print a short summary.'''
        
        print("Total n. of counts:")
        if type(self.tot_counts) == pd.Series :
            print(self.tot_counts.to_string())
        else :
            print(self.tot_counts)
            
        print("N. of Categories:")
        print(self.obs_n_categ, ' (observed)')
        print(self.usr_n_categ, ' (a priori)')
        
        print("Multiplicities:")
        print(self.counts_hist)
        pass
    
######################
#  EXPERIMENT CLASS  #
######################

class Experiment( Skeleton_Class ) :
    '''The basic class for estimating entropy of a distribution given a sample.'''
    
    def __init__(self, data_hist, categories=2, ishist=True):
        '''
        Parameters
        ----------    
        data_hist : Union[dict, pd.Series, pd.DataFrame, list, np.array] 
            The number of observations for each category or a raw list of observations (deprecated).
        categories : scalar, optional
            The a priori total number of categories.
        ishist : bool, optional
            When `data_hist` is a list of integers, it specifies wheter it is a histogram (True, default) 
            or a raw list of observations (False, deprecated).
        '''
        
        #  Load data_hist #
        if type( data_hist ) == dict :   
            # loading dictionary where values represent keys counts                                    
            data_hist = pd.Series( data_hist )
        elif type( data_hist ) == pd.DataFrame :
            # loading datafame where values represent keys counts 
            if len( data_hist.columns ) > 1 :
                warnings.warn("The parameter `data` contained multiple columns, only the first one is considered.")
            data_hist = data_hist[ data_hist.columns[0] ]
        elif type( data_hist ) == pd.Series :
            # loading series where values represent index counts
            data_hist = data_hist
        elif type( data_hist ) == list or type( data_hist ) == np.ndarray :                              
            if ishist == True :
                # loading list of counts
                data_hist = pd.Series( data_hist ).astype(int)
            else :
                # loading raw list of sequences
                temp = pd.Series( data_hist )
                data_hist = temp.groupby( temp ).size()
        else :
            raise TypeError("The parameter `data` has unsopported type.")

        # check int counts 
        # FIXME : these try/except for option loading can be messy
        try :
            data_hist = data_hist.astype(int)
        except ValueError :
            raise ValueError("Unrecognized count value in `data`.")    

        # check categories
        try :
            categories = int(categories)
        except :
            raise TypeError("The parameter `categories` must be a scalar.")

        self.data_hist = data_hist
        # total n. of counts
        self.tot_counts = np.sum( data_hist.values )  
        # n. of observed categories
        self.obs_n_categ = (data_hist > 0).sum() 
        # frquencies of observed counts
        temp = pd.Series( data_hist.values ).astype(int)

        self.counts_hist = temp.groupby( temp ).size()  
        self.counts_hist.name = "freq"
        # n. of coincidences
        self.obs_coincedences = self.counts_hist[ self.counts_hist.index > 1 ].sum()

        #  Load categories  #
        categories = np.max([categories, self.obs_n_categ])
        self.update_categories( categories )
        pass

    def _fix_zero_counts( self ) :
        ''' (internal) Add/remove 0 values to counts_hist.'''
        if self.usr_n_categ > self.obs_n_categ :
            self.counts_hist.at[0] = self.usr_n_categ - self.obs_n_categ
            self.counts_hist = self.counts_hist.sort_index(ascending=True) 
        elif self.usr_n_categ == self.obs_n_categ :
            if 0 in self.counts_hist.index :
                self.counts_hist.drop(index=[0], inplace=True)
        else :
            raise ValueError('Interal inconsistecy between n. of categories.')
        pass

    def shannon(self, method="naive", unit="ln", **kwargs):
        '''Estimate Shannon entropy with a chosen `method`.'''
        return default_entropy.switchboard( self.compact(), which="Shannon", method=method, unit=unit, **kwargs )
    
    def simpson( self, method="naive", **kwargs ):
        '''Estimate Simpson index with a chosen `method`.'''
        return default_entropy.switchboard( self.compact(), which="Simpson", method=method, **kwargs )
    
    def compact( self ) :
        '''It provides aliases for computations.'''
        return Experiment_Compact( source=self )

    def save_compact( self, filename ) :
        '''It saves the compact features of Experiment to `filename`.'''
        self.compact( ).save( filename )
        pass

# fix docstring
Experiment.shannon.__doc__ += __entropy_doc__
Experiment.simpson.__doc__ += __entropy_doc__

######################
#  DIVERGENCE CLASS  #
######################

class Divergence( Skeleton_Class ) :
    '''The basic class for estimating divergence between two distributions given two samples.'''

    def __init__( self, my_exp_1, my_exp_2, categories=2, ishist=True ) :
        '''
        Parameters
        ----------    
        my_exp_1 : class Experiment
            the first dataset.
        my_exp_2 : class Experiment
            the second dataset.
        '''

        if type( my_exp_1 ) != Experiment :
            my_exp_1 = Experiment( my_exp_1, categories=categories, ishist=ishist )
        if type( my_exp_2 ) != Experiment :
            my_exp_2 = Experiment( my_exp_2, categories=categories, ishist=ishist )

        self.tot_counts = pd.Series(
            {"Exp-1": my_exp_1.tot_counts,
            "Exp-2": my_exp_2.tot_counts}
            )

        df = pd.concat( [my_exp_1.data_hist, my_exp_2.data_hist], axis=1 )
        df = df.replace(np.nan, 0).astype(int)
        df.columns = ["Exp-1", "Exp-2"]
        self.data_hist = df

        # Observed Categories
        self.obs_n_categ = pd.Series(
            {"Exp-1": my_exp_1.obs_n_categ,
            "Exp-2":my_exp_2.obs_n_categ,
            "Union":len(self.data_hist.loc[(df>0).any(axis=1)])}
            )

        # Counts Histogram
        self.counts_hist = df.groupby(by=["Exp-1", "Exp-2"]).size()
        self.counts_hist.name = "freq"

        # User Stated Categories
        categories = np.max([my_exp_1.usr_n_categ, my_exp_2.usr_n_categ, self.obs_n_categ["Union"]])
        self.update_categories( categories )

        # WARNING!: is this a deep copy ?
        tmp = self.counts_hist.reset_index(level=[0,1]).copy()
        # experiment 1 copy
        self.exp_1 = Experiment(my_exp_1.data_hist, categories=self.usr_n_categ)
        counts_1 = tmp[["freq", "Exp-1"]].set_index("Exp-1", drop=True)
        self.exp_1.counts_hist = counts_1["freq"] # convert to series
        # experiment 2 copy
        self.exp_2 = Experiment(my_exp_2.data_hist, categories=self.usr_n_categ)
        counts_2 = tmp[["freq", "Exp-2"]].set_index("Exp-2", drop=True)
        self.exp_2.counts_hist = counts_2["freq"]
        pass

    def kullback_leibler( self, method="naive", unit="ln", **kwargs ):
        '''Estimate the Kullback-Leibler divergence with a chosen `method`.'''
        return default_divergence.switchboard( self.compact(), which="Kullback-Leibler", method=method, unit=unit, **kwargs )

    def jensen_shannon( self, method="naive", unit="ln", **kwargs ):
        '''Estimate the Jensen-Shannon divergence with a chosen `method`.'''
        return default_divergence.switchboard( self.compact(), method=method, which="Jensen-Shannon", unit=unit, **kwargs )

    def symmetrized_KL( self, method="naive", unit="ln", **kwargs ):
        '''Estimate the symmetrized Kullback-Leibler divergence with a chosen `method`.'''
        return default_divergence.switchboard( self.compact(), method=method, which="symmetrized-KL", unit=unit, **kwargs )

    def squared_hellinger( self, method="naive", **kwargs ):
        '''Estimate the (squared) Hellinger divergence with a chosen `method`.'''
        return default_divergence.switchboard( self.compact(), method=method, which="squared-Hellinger", **kwargs )
    
    def compact( self ) :
        '''It provides aliases for computations.'''
        return Divergence_Compact( self )

    def save( self, filename, compression="gzip" ) : 
        '''It saves the Divergence object to `filename`.'''
        outFrame = self.data_hist.copy()
        add_ons = pd.Series(
            {"Exp-1" : self.exp_1.usr_n_categ, "Exp-2" : self.exp_2.usr_n_categ}, 
            name="__usr_n_categ__"
            )
        outFrame = outFrame.append( add_ons )
        outFrame.to_csv( filename, sep=' ', mode='w', header=True, index=True, compression=compression )
        pass

    def save_compact( self, filename ) :
        '''It saves the compact version of Divergence to `filename`.'''
        self.compact( ).save( filename )
        pass

    def _fix_zero_counts( self ) :
        ''' (internal) add/remove (0,0) to counts_hist.'''
        if self.usr_n_categ > self.obs_n_categ["Union"] :
            self.counts_hist.at[(0,0)] = self.usr_n_categ - self.obs_n_categ["Union"]
            self.counts_hist = self.counts_hist.sort_index(ascending=True)
        elif self.usr_n_categ == self.obs_n_categ["Union"] :
            if (0,0) in self.counts_hist.index :
                self.counts_hist.drop(index=(0,0), inplace=True)
        else :
            raise ValueError('Interal inconsistecy between n. of categories.')
        pass

# fix docstring
Divergence.kullback_leibler.__doc__ += __diverg_doc__
Divergence.jensen_shannon.__doc__ += __diverg_doc__
Divergence.symmetrized_KL.__doc__ += __diverg_doc__
Divergence.squared_hellinger.__doc__ += __diverg_doc__

def load_diver( filename ) :
    '''  Load the Divergence object stored in `filename`. '''
    df = pd.read_csv( filename, sep=" ", index_col=0, compression="infer", na_filter=False )
    this_categories = df.loc["__usr_n_categ__"].max()
    df = df.drop(index=["__usr_n_categ__"])
    Div = Divergence( df["Exp-1"], df["Exp-2"], categories=this_categories )
    return Div
